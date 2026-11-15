"""Skeleton punch detector trained as Positive-Unlabelled (PU) learning, leave-one-video-out.

Positives = frames inside labelled punches. Everything else is UNLABELLED (mostly no punch, but
annotators missed some punches). We follow Elkan & Noto (2008):
  1. Train g(x) = P(labelled | x), "labelled vs unlabelled", per frame. Under the SCAR assumption
     (labelled punches are a random sample of all punches), g(x) = c * P(punch | x) where
     c = P(labelled | punch) is a constant.
  2. Estimate c as the mean of g over held-out labelled frames, then P(punch | x) = g(x) / c.
  3. Expected number of hidden punch frames = sum of P(punch | x) over unlabelled frames.
For fairness, every video's scores come from a model trained only on the OTHER 9 videos.

Model: non-causal temporal conv net over the whole frame stream (offline use only: dataset
preparation). Input per frame: MediaPipe 3D (hip-centred, torso-scaled, rotated to face the
camera with a smoothed hip angle), velocities per second, and joint visibility = 119 channels.

Outputs:
  detect/scores/pu_skeleton/V{v}.npz: frame_g (F,), frame_p (F,) = P(punch|x), start (windows),
      score (window P(punch), per detect/WINDOWS.md), c (calibration constant)
  results to RUN_DIR: pu_summary.json (per-video AUC, recall, hidden-punch estimate)
    python detect/pu_detector.py [--steps 2500]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mem_guard  # noqa: F401,E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from progress import progress  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
FRAMES = ROOT / "detect" / "frames"
OUT = ROOT / "detect" / "scores" / "pu_skeleton"
VIDEOS = list(range(1, 11))
W, S = 16, 4                                   # shared window definition (detect/WINDOWS.md)


# ------------------------------------------------------------------ features
def features(d):
    world = d["world"].astype(np.float32); vis = d["vis"].astype(np.float32); present = d["present"]
    fps = float(d["fps"])
    hip = (world[:, 11] + world[:, 12]) / 2
    sh = (world[:, 5] + world[:, 6]) / 2
    torso = np.linalg.norm(sh - hip, axis=1)
    scale = np.median(torso[present]) if present.any() else 1.0
    Z = (world - hip[:, None]) / max(scale, 1e-3)
    # rotate about the vertical axis so the (smoothed) hip line faces the camera
    hv = world[:, 11] - world[:, 12]
    ang = np.arctan2(hv[:, 2], hv[:, 0])
    k = 25
    c_, s_ = np.convolve(np.cos(ang), np.ones(k) / k, "same"), np.convolve(np.sin(ang), np.ones(k) / k, "same")
    th = np.arctan2(s_, c_)
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    x, z = Z[..., 0].copy(), Z[..., 2].copy()
    Z[..., 0], Z[..., 2] = x * c + z * s, -x * s + z * c
    V = np.diff(Z, axis=0, prepend=Z[:1]) * (fps / 25.0)         # per-frame motion normalised to 25 fps
    F = np.concatenate([Z.reshape(len(Z), -1), V.reshape(len(Z), -1), vis], axis=1)
    F[~present] = 0
    return F.astype(np.float32)


def load_all():
    data = {}
    for v in VIDEOS:
        d = np.load(FRAMES / f"V{v}.npz")
        data[v] = dict(x=features(d), pos=(d["label"] >= 0), present=d["present"].copy(), label=d["label"])
    return data


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, ch, dil, p):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.bn = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(torch.relu(self.bn(self.conv(x))))


class TCN(nn.Module):
    """Non-causal dilated TCN; receptive field ~ 1 + 2*(1+2+4+8+16+1) = 65 frames (~2.5 s)."""
    def __init__(self, c_in, ch=128, p=0.2):
        super().__init__()
        self.inp = nn.Conv1d(c_in, ch, 1)
        self.blocks = nn.Sequential(*[Block(ch, d, p) for d in (1, 2, 4, 8, 16, 1)])
        self.out = nn.Conv1d(ch, 1, 1)

    def forward(self, x):                                       # x: (B, T, C)
        return self.out(self.blocks(self.inp(x.transpose(1, 2)))).squeeze(1)   # (B, T) logits


def train_fold(data, test_v, steps, seed, crop=256, batch=32):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    train = [v for v in VIDEOS if v != test_v]
    X = np.concatenate([data[v]["x"] for v in train]); mu, sd = X.mean(0), X.std(0) + 1e-6
    lens = np.array([len(data[v]["x"]) for v in train], float)
    n_pos = sum(data[v]["pos"][data[v]["present"]].sum() for v in train)
    n_unl = sum((~data[v]["pos"])[data[v]["present"]].sum() for v in train)
    pos_weight = torch.tensor(float(n_unl / max(n_pos, 1)), dtype=torch.float32, device=DEVICE)
    model = TCN(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    model.train()
    for step in range(steps):
        xb, yb, mb = [], [], []
        for _ in range(batch):
            v = train[rng.choice(len(train), p=lens / lens.sum())]
            s = rng.integers(0, len(data[v]["x"]) - crop)
            xb.append((data[v]["x"][s:s + crop] - mu) / sd)
            yb.append(data[v]["pos"][s:s + crop]); mb.append(data[v]["present"][s:s + crop])
        xb = torch.tensor(np.stack(xb), device=DEVICE)
        yb = torch.tensor(np.stack(yb), dtype=torch.float32, device=DEVICE)
        mb = torch.tensor(np.stack(mb), dtype=torch.float32, device=DEVICE)
        loss = (nn.functional.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight,
                                                                reduction="none") * mb).sum() / mb.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            progress.tick(50 / steps, note=f"fold V{test_v} step {step}")
    model.eval()
    return model, mu, sd, pos_weight.item()


@torch.no_grad()
def predict(model, x, mu, sd, chunk=4096, margin=64):
    xs = (x - mu) / sd; F = len(xs); out = np.zeros(F, np.float32)
    for s in range(0, F, chunk):
        a, b = max(0, s - margin), min(F, s + chunk + margin)
        logit = model(torch.tensor(xs[a:b][None], device=DEVICE))[0].cpu().numpy()
        out[s:min(F, s + chunk)] = logit[s - a: s - a + min(chunk, F - s)]
    return out


def windows(frame_p, label, present):
    F = len(frame_p)
    starts = np.arange(0, F - W + 1, S, dtype=np.int32)
    sm = np.convolve(frame_p, np.ones(3) / 3, "same")
    score = np.array([sm[s:s + W].max() for s in starts], np.float32)
    lab = label >= 0
    cov = np.array([lab[s:s + W].mean() for s in starts])
    near = np.convolve(lab.astype(float), np.ones(2 * 8 + 1), "same") > 0          # within 8 frames of a punch
    gap = np.array([not near[s:s + W].any() for s in starts])
    pres = np.array([present[s:s + W].mean() for s in starts])
    return starts, score, cov, gap, pres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(os.environ.get("RUN_DIR", ROOT / "runs" / "pu_detector")); out_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"device {DEVICE}")
    data = load_all()
    progress.start("PU detector", total=len(VIDEOS))
    summary = {}
    for v in VIDEOS:
        t0 = time.time()
        model, mu, sd, pw = train_fold(data, v, args.steps, args.seed)
        logit = predict(model, data[v]["x"], mu, sd)
        g = 1 / (1 + np.exp(-logit))
        # pos_weight re-balances classes in training, so g is P(labelled | x) on a rebalanced scale;
        # Elkan-Noto only needs a monotone score + the constant c measured on labelled frames.
        pres, pos = data[v]["present"], data[v]["pos"]
        c = float(np.mean(g[pres & pos]))
        p = np.clip(g / c, 0, 1)
        starts, score, cov, gap, wp = windows(p, data[v]["label"], pres)
        np.savez_compressed(OUT / f"V{v}.npz", frame_g=g.astype(np.float16), frame_p=p.astype(np.float16),
                            start=starts, score=score, c=c)
        lab_win = cov >= 0.5
        gap_win = gap & (wp > 0.5)
        thr95 = np.quantile(score[lab_win], 0.05) if lab_win.any() else np.nan
        lab = data[v]["label"]
        n_punch = int(((lab >= 0) & np.r_[True, lab[1:] != lab[:-1]]).sum())   # new punch at every label change
        lab_len = max(1.0, pos.sum() / max(1, n_punch))                        # mean labelled punch length (frames)
        hidden_frames = float(p[pres & ~pos].sum())
        summary[f"V{v}"] = dict(
            frames=int(len(g)), c=round(c, 4),
            auc_labelled_vs_gap=round(float(roc_auc_score(np.r_[np.ones(lab_win.sum()), np.zeros(gap_win.sum())],
                                                          np.r_[score[lab_win], score[gap_win]])), 4),
            threshold_95pct_recall=round(float(thr95), 4),
            gap_windows=int(gap_win.sum()),
            gap_windows_above_thr95=round(float((score[gap_win] >= thr95).mean()), 4),
            hidden_punch_frames_est=round(hidden_frames, 1),
            hidden_punches_est=round(hidden_frames / lab_len, 1),
            labelled_punches=n_punch, mean_punch_frames=round(lab_len, 1),
            seconds=round(time.time() - t0, 1))
        print(f"V{v}: " + ", ".join(f"{k} {val}" for k, val in summary[f"V{v}"].items()), flush=True)
        del model
    (out_dir / "pu_summary.json").write_text(json.dumps(summary, indent=1))
    progress.finish()


if __name__ == "__main__":
    main()
