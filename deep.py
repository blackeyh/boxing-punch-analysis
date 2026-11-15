"""Deep sequence models (1D-CNN, LSTM) on the Apple GPU (MPS).

Same data and same evaluation as the random-forest baselines, so numbers are
directly comparable with B4c:
  * orthodox-canonicalised clips (V4 mirrored)
  * normalised (hip centre, torso scale), resampled to K=10 frames
  * each frame = 34 positions + 34 velocities = 68 channels  -> input (N, 10, 68)
  * scored by `evaluate()`: random 80/20 + leave-one-video-out balanced accuracy

The torch model is wrapped as a scikit-learn estimator so `evaluate()` can
clone/fit/predict it exactly like the random forest.

Run:  /opt/anaconda3/bin/python deep.py                 # cnn + lstm, seeds 0 1 2
      /opt/anaconda3/bin/python deep.py cnn --seeds 0   # one model, one seed
"""
import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path

import mem_guard  # noqa: F401  MUST come before torch: MPS memory cap + RSS watchdog
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

from baselines import FITS_PER_EVAL, evaluate, normalise, resample, to_orthodox
from progress import progress
from punch_data import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available()          # HPC GPU
                      else "mps" if torch.backends.mps.is_available()  # Mac GPU
                      else "cpu")


# ---------------------------------------------------------------- features
N_FACE = 5  # COCO keypoints 0-4: nose, eyes, ears


def frontalness(X):
    """Signed camera-angle cue from the punch's FIRST frame (the guard, before the punch
    rotates the shoulders): (left shoulder x - right shoulder x) / torso length.
    About +-0.35-0.45 facing the camera, near 0 side-on; the sign says which way the
    fighter is turned. Computable live from any single frame of pose."""
    f = X[:, 0]
    torso = np.linalg.norm((f[:, 5] + f[:, 6]) / 2 - (f[:, 11] + f[:, 12]) / 2, axis=-1)
    return (f[:, 5, 0] - f[:, 6, 0]) / np.maximum(torso, 1e-3)


LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]  # upper arms, forearms, thighs, shins


def seq_features(X, K=10, face=True, duration=False, frontal=False, foreshort=False):
    """(N, 25, 17, 2) -> (N, K, C). Channel layout (the augmentation relies on it):
        [positions (P) | velocities (P) | duration (1, opt) | frontalness (1, opt) | foreshortening (8, opt)]
    P = 34 with face keypoints, 24 without (face=False drops nose/eyes/ears).
    duration = log(real frame count), repeated on every frame (speed info that
    resampling otherwise removes). frontalness = see frontalness(), repeated."""
    lengths = (np.abs(X).sum(axis=(2, 3)) > 0).sum(axis=1)
    R = resample(normalise(X, "per_frame"), K)                       # (N, K, 17, 2)
    if not face:
        R = R[:, :, N_FACE:]
    R = R.reshape(len(X), K, -1)                                     # (N, K, P)
    V = np.diff(R, axis=1, prepend=R[:, :1])                         # (N, K, P), first = 0
    parts = [R, V]
    if duration:
        parts.append(np.repeat(np.log(lengths)[:, None, None], K, axis=1))
    if frontal:
        parts.append(np.repeat(frontalness(X)[:, None, None], K, axis=1))
    if foreshort:   # depth cue: a limb pointing at the camera looks short. Each limb's on-screen
        # length divided by its longest length in the clip (1 = fully across the image plane)
        Rk = resample(normalise(X, "per_frame"), K)                    # (N, K, 17, 2), uses all 17 joints
        seg = np.stack([np.linalg.norm(Rk[:, :, a] - Rk[:, :, b], axis=-1) for a, b in LIMBS], axis=2)
        parts.append(seg / np.maximum(seg.max(axis=1, keepdims=True), 1e-3))      # (N, K, 8)
    return np.concatenate(parts, axis=2).astype(np.float32)


# ---------------------------------------------------------------- models
class CNN1D(nn.Module):
    """Slides small windows (3 frames) over time to detect short motion patterns."""
    def __init__(self, in_ch, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        def block(i, o):
            return [nn.Conv1d(i, o, kernel_size=3, padding=1), nn.BatchNorm1d(o), nn.ReLU()]
        self.net = nn.Sequential(*block(in_ch, hidden), *block(hidden, hidden),
                                 *block(hidden, hidden), nn.AdaptiveAvgPool1d(1),
                                 nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):                 # x: (B, T, C)
        return self.net(x.transpose(1, 2))  # Conv1d wants (B, C, T)


class ManualLSTM(nn.Module):
    """One-direction LSTM layer built from plain Linear ops.

    Why not nn.LSTM: on MPS (torch 2.9) nn.LSTM leaks driver memory that is never
    returned, even by torch.mps.empty_cache(). A full run reached 46.7 GB and
    kernel-panicked the Mac (see PROJECT_LOG.md, D1). Same maths, no leak.
    """
    def __init__(self, in_ch, hidden):
        super().__init__()
        self.hidden = hidden
        self.x2g = nn.Linear(in_ch, 4 * hidden)              # input  -> 4 gates
        self.h2g = nn.Linear(hidden, 4 * hidden, bias=False)  # hidden -> 4 gates

    def forward(self, x):                 # x: (B, T, C) -> (B, T, H)
        B, T, _ = x.shape
        gx = self.x2g(x)                  # all input projections at once
        h = x.new_zeros(B, self.hidden); c = x.new_zeros(B, self.hidden)
        outs = []
        for t in range(T):
            i, f, g, o = (gx[:, t] + self.h2g(h)).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outs.append(h)
        return torch.stack(outs, dim=1)


class LSTMNet(nn.Module):
    """Reads frames in order, both directions, 2 layers; averages outputs over time."""
    def __init__(self, in_ch, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.fw1, self.bw1 = ManualLSTM(in_ch, hidden), ManualLSTM(in_ch, hidden)
        self.fw2, self.bw2 = ManualLSTM(2 * hidden, hidden), ManualLSTM(2 * hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden, n_classes)

    @staticmethod
    def _bi(fw, bw, x):                   # bidirectional = forward pass + pass on reversed time
        return torch.cat([fw(x), bw(x.flip(1)).flip(1)], dim=2)

    def forward(self, x):                 # x: (B, T, C)
        h = self.drop(self._bi(self.fw1, self.bw1, x))
        h = self._bi(self.fw2, self.bw2, h)
        return self.head(self.drop(h.mean(dim=1)))


# ---------------------------------------------------------------- skeleton graph network
# COCO-17 bones. The graph network uses them so information flows along the body:
# wrist <-> elbow <-> shoulder <-> hip ... instead of treating 34 numbers as unrelated.
COCO_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4),            # head
              (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),    # shoulders, arms
              (5, 11), (6, 12), (11, 12),                 # torso
              (11, 13), (13, 15), (12, 14), (14, 16)]     # legs


def coco_adjacency(V=17):
    """Symmetrically normalised adjacency D^-1/2 (A + I) D^-1/2."""
    A = np.eye(V)
    for a, b in COCO_EDGES:
        A[a, b] = A[b, a] = 1
    d = A.sum(1) ** -0.5
    return (d[:, None] * A * d[None, :]).astype(np.float32)


class GraphConv(nn.Module):
    """Spatial graph convolution: each joint mixes features from itself and its
    bone neighbours (fixed skeleton graph) plus a LEARNED joint-to-joint matrix,
    so the network can also link joints that aren't physically connected
    (e.g. the two wrists)."""
    def __init__(self, c_in, c_out, A):
        super().__init__()
        self.register_buffer("A", torch.tensor(A))
        self.B = nn.Parameter(torch.zeros_like(self.A))      # learned extra links, start at 0
        self.self_w = nn.Conv2d(c_in, c_out, 1)              # joint's own features
        self.nbr_w = nn.Conv2d(c_in, c_out, 1)               # neighbours' features

    def forward(self, x):                                    # x: (B, C, T, V)
        nbr = torch.einsum("bctv,vw->bctw", x, self.A + self.B)
        return self.self_w(x) + self.nbr_w(nbr)


class STGCNBlock(nn.Module):
    """Graph conv over joints, then a temporal conv over 3 frames, with a residual."""
    def __init__(self, c_in, c_out, A, dropout):
        super().__init__()
        self.gcn = GraphConv(c_in, c_out, A)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.tcn = nn.Conv2d(c_out, c_out, kernel_size=(3, 1), padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(c_out)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        y = torch.relu(self.bn1(self.gcn(x)))
        y = self.drop(self.bn2(self.tcn(y)))
        return torch.relu(y + self.res(x))


class STGCN(nn.Module):
    """Spatio-temporal graph network. Input is the same (B, T, 68) features as the CNN,
    regrouped per joint: (B, 4, T, 17) with channels x, y, vx, vy."""
    def __init__(self, in_ch, n_classes, hidden=128, dropout=0.3):
        super().__init__()
        assert in_ch == 68, "stgcn needs the default 68-channel input (17 joints x pos+vel)"
        A = coco_adjacency()
        h = hidden // 2
        self.bn_in = nn.BatchNorm1d(4 * 17)
        self.blocks = nn.Sequential(STGCNBlock(4, h, A, dropout), STGCNBlock(h, h, A, dropout),
                                    STGCNBlock(h, hidden, A, dropout), STGCNBlock(hidden, hidden, A, dropout))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):                                    # x: (B, T, 68)
        B, T, _ = x.shape
        pos = x[..., :34].reshape(B, T, 17, 2)
        vel = x[..., 34:68].reshape(B, T, 17, 2)
        z = torch.cat([pos, vel], dim=3)                     # (B, T, 17, 4)
        z = self.bn_in(z.permute(0, 2, 3, 1).reshape(B, 17 * 4, T))      # (B, 68, T)
        z = z.reshape(B, 17, 4, T).permute(0, 2, 3, 1)       # (B, 4, T, 17)
        z = self.blocks(z)
        return self.head(z.mean(dim=(2, 3)))                 # pool over time and joints


ARCHS = {"cnn": CNN1D, "lstm": LSTMNet, "stgcn": STGCN}


# ---------------------------------------------------------------- sklearn wrapper
class TorchSeqClassifier(ClassifierMixin, BaseEstimator):
    reports_progress = True  # ticks the status-line gauge once per epoch

    def __init__(self, arch="cnn", epochs=60, batch_size=128, lr=1e-3,
                 weight_decay=1e-2, hidden=128, dropout=0.3, seed=0,
                 class_weight=False, augment=False, n_pos=34, view_aug=False):
        self.arch, self.epochs, self.batch_size, self.lr = arch, epochs, batch_size, lr
        self.weight_decay, self.hidden, self.dropout, self.seed = weight_decay, hidden, dropout, seed
        self.class_weight, self.augment, self.n_pos = class_weight, augment, n_pos
        self.view_aug = view_aug

    def _view(self, xb):
        """Training-only VIEW augmentation: squash every x-coordinate (positions are
        hip-centred) by a random factor 0.4-1.0 in half the clips, a crude stand-in for
        seeing the same punch from a more side-on camera. 2D can't truly rotate the body
        (depth is unknown), so this widens the range of apparent shoulder widths and
        horizontal arm travel the model trains on. Velocities are recomputed."""
        P = self.n_pos
        s = torch.empty(len(xb), 1, 1, device=DEVICE).uniform_(0.4, 1.0)
        s = torch.where(torch.rand(len(xb), 1, 1, device=DEVICE) < 0.5, s, torch.ones_like(s))
        xmask = torch.zeros(P, device=DEVICE); xmask[0::2] = 1           # x is every even channel
        pos = xb[..., :P] * (1 - xmask + xmask * s)
        vel = torch.diff(pos, dim=1, prepend=pos[:, :1])
        return torch.cat([pos, vel, xb[..., 2 * P:]], dim=2)

    def _augment(self, xb):
        """Training-only augmentation on UN-standardised features [pos | vel | rest]:
        random body scale (0.9-1.1) + small joint jitter (0.02 torso lengths),
        then velocities are recomputed from the augmented positions."""
        P = self.n_pos
        pos = xb[..., :P] * torch.empty(len(xb), 1, 1, device=DEVICE).uniform_(0.9, 1.1)
        pos = pos + 0.02 * torch.randn_like(pos)
        vel = torch.diff(pos, dim=1, prepend=pos[:, :1])
        return torch.cat([pos, vel, xb[..., 2 * P:]], dim=2)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        # standardise each channel with TRAINING statistics only
        self.mu_ = X.mean(axis=(0, 1), keepdims=True)
        self.sd_ = X.std(axis=(0, 1), keepdims=True) + 1e-6
        Xraw = torch.tensor(X, device=DEVICE)
        mu, sd = torch.tensor(self.mu_, device=DEVICE), torch.tensor(self.sd_, device=DEVICE)
        yt = torch.tensor(y_idx, device=DEVICE)

        self.model_ = ARCHS[self.arch](X.shape[2], len(self.classes_),
                                       self.hidden, self.dropout).to(DEVICE)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        steps = self.epochs * int(np.ceil(len(Xraw) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        weight = None
        if self.class_weight:  # 'balanced': rare classes (Rear Hook) count more in the loss
            counts = np.bincount(y_idx, minlength=len(self.classes_))
            weight = torch.tensor(len(y_idx) / (len(counts) * counts), dtype=torch.float32, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xraw), device=DEVICE)
            for i in range(0, len(Xraw), self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = Xraw[idx]
                if self.view_aug:
                    xb = self._view(xb)
                if self.augment:
                    xb = self._augment(xb)
                loss = loss_fn(self.model_((xb - mu) / sd), yt[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
            progress.tick(1 / self.epochs)  # one fit = 1 unit, split across its epochs
        self.model_.eval()
        del Xraw, mu, sd, yt, opt, sched
        if DEVICE.type == "mps":
            torch.mps.empty_cache()       # hand cached GPU memory back after every fit
        elif DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        Xt = torch.tensor((X - self.mu_) / self.sd_, device=DEVICE)
        logits = torch.cat([self.model_(Xt[i:i + 1024]) for i in range(0, len(Xt), 1024)])
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("archs", nargs="*", default=["cnn", "lstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--duration", action="store_true", help="add log(real frame count) channel")
    ap.add_argument("--no-face", action="store_true", help="drop nose/eyes/ears keypoints")
    ap.add_argument("--class-weight", action="store_true", help="balanced class-weighted loss")
    ap.add_argument("--augment", action="store_true", help="scale + jitter augmentation")
    ap.add_argument("--view-aug", action="store_true", help="random horizontal squash (camera-angle augmentation)")
    ap.add_argument("--frontal", action="store_true", help="add frontalness (camera angle) input channel")
    ap.add_argument("--foreshort", action="store_true", help="add limb foreshortening (depth cue) channels")
    ap.add_argument("--input", choices=["2d", "lift", "lift-rot", "2d+lift-rot", "mp3d", "mp3d-rot", "2d+mp3d-rot"], default="2d",
                    help="2d skeletons, VideoPose3D-lifted 3D, lifted 3D rotated to face the camera, or 2D + rotated 3D")
    ap.add_argument("--smooth3d", type=int, default=0, help="moving-average window (frames) for lifted 3D")
    ap.add_argument("--conf-fill", type=float, default=0, help="MediaPipe: interpolate joints with confidence below this")
    ap.add_argument("--conf-input", action="store_true", help="MediaPipe: add the 17 joint confidences as input channels")
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", "."), help="where results.json goes")
    args = ap.parse_args()

    variant = "+".join(f for f, on in [("dur", args.duration), ("noface", args.no_face),
                                       ("cw", args.class_weight), ("aug", args.augment),
                                       ("viewaug", args.view_aug), ("frontal", args.frontal), ("foreshort", args.foreshort),
                                       (args.input, args.input != "2d"), (f"smooth{args.smooth3d}", args.smooth3d > 1),
                                       (f"fill{args.conf_fill}", args.conf_fill > 0), ("confin", args.conf_input)] if on) or "base"
    print(f"device: {DEVICE}  variant: {variant}")
    X, y, video_id, _ = load_dataset()
    X_orth = to_orthodox(X, video_id)
    feats = lambda X: seq_features(X, face=not args.no_face, duration=args.duration, frontal=args.frontal,
                                   foreshort=args.foreshort)
    n_pos = (17 - (N_FACE if args.no_face else 0)) * 2
    if args.input != "2d":   # lifted 3D (made from the same orthodox 2D clips, row-for-row)
        from lift3d.features import load_lifted, seq_features_3d
        assert not (args.no_face or args.duration or args.frontal or args.view_aug or args.foreshort), "2D-only options"
        from lift3d.features import fill_low_conf, load_mediapipe3d
        mp = args.input.endswith("mp3d") or "mp3d" in args.input
        layout = "coco" if mp else "h36m"
        conf = None
        if mp:
            X3, conf_all = load_mediapipe3d(y, video_id, with_conf=True)
            if args.conf_fill > 0:
                X3, frac = fill_low_conf(X3, conf_all, args.conf_fill)
                print(f"conf-fill < {args.conf_fill}: replaced {frac:.1%} of joint-frames")
            conf = conf_all if args.conf_input else None
        else:
            assert not (args.conf_fill or args.conf_input), "confidence options are MediaPipe-only"
            X3 = load_lifted(y, video_id)
        rot = args.input.endswith("-rot")
        if args.input.startswith("2d+"):
            # both views per frame: [2D pos 34 | 2D vel 34 | 3D pos 51 | 3D vel 51]. Features are
            # computed per clip with no fitting, so precomputing them is leak-free.
            X_orth = np.concatenate([seq_features(X_orth), seq_features_3d(X3, rotate=True, smooth=args.smooth3d, layout=layout, conf=conf)], axis=2)
            feats = lambda F: F
            n_pos = 34   # augmentation (if used) only touches the 2D part
        else:
            # precompute (no fitting involved, so leak-free); keeps conf aligned with rows
            X_orth = seq_features_3d(X3, rotate=rot, smooth=args.smooth3d, layout=layout, conf=conf)
            feats = lambda F: F
            n_pos = 17 * 3

    progress.start("deep.py", total=len(args.archs) * len(args.seeds) * FITS_PER_EVAL)
    results = {"variant": variant, "args": vars(args), "device": str(DEVICE), "runs": []}
    for arch in args.archs:
        scores = []
        for seed in args.seeds:
            progress.tick(0, note=f"{arch} {variant} seed {seed}")
            t0 = time.time()
            model = TorchSeqClassifier(arch=arch, epochs=args.epochs, seed=seed, n_pos=n_pos,
                                       class_weight=args.class_weight, augment=args.augment,
                                       view_aug=args.view_aug)
            quiet = seed != args.seeds[0]  # full report for the first seed only
            with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
                r = evaluate(f"D-{arch} {variant} seed {seed}", model, feats,
                             X_orth, y, video_id, seed=seed)
            scores.append(r["lovo_mean"])
            results["runs"].append({"arch": arch, "seed": seed, "random": r["random"],
                                    "lovo_mean": r["lovo_mean"],
                                    "lovo_per_video": {f"V{k}": v for k, v in r["lovo"].items()},
                                    "confusion": r["confusion"].tolist(), "seconds": time.time() - t0,
                                    "lovo_pred": r["lovo_pred"].tolist(),
                                    "lovo_proba": np.round(r["lovo_proba"], 4).tolist()})
            print(f"[{arch} {variant} seed {seed}] LOVO {r['lovo_mean']:.1%}  random {r['random']:.1%}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        print(f"==> {arch} {variant}: LOVO {np.mean(scores):.1%} ± "
              f"{np.std(scores, ddof=1) if len(scores) > 1 else 0:.1%} over seeds {args.seeds}\n", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results.json").write_text(json.dumps(results, indent=1))
    progress.finish()
