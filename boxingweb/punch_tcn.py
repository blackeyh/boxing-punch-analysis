"""Punch detection + type on RTMPose skeletons: a per-frame temporal network (dense spotting), two-person input.

Data: external/punch_ds (boxingweb/punch_data.py). Train = BoxingWeb data_train minus VAL rounds; VAL (the last 5
training rounds) picks the decision threshold; TEST = data_test (the authors' split). The boxers are the truth-matched
skeletons here, so this measures detection alone.
Model: 1D dilated residual convolutions over time (sees about +-1 s), one output per frame: no punch / 6 types.
Events: per boxer, stretches where smoothed P(punch) > threshold (>= 3 frames, gaps <= 2 bridged, split where the
punching hand changes); type = the class with the most probability mass.
Scoring per boxer, one-to-one: a counted punch matches a labelled one if they overlap in time (label grown by 3 frames).
  found (recall), real (precision), F1; type right and hand right among matches; chance = labels shifted by 7 s.
Labelled punches while the boxer has no skeleton count as missed.
    python boxingweb/punch_tcn.py [--no-opp] [--seed 0] [--epochs 40] [--name NAME]
"""
import argparse, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mem_guard  # noqa: F401
import numpy as np, torch, torch.nn as nn
from progress import progress
import punch_data as pd
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
OPP = slice(70, 138)                                    # opponent pose + change channels
L, VAL_N, TOL = 512, 5, 3

class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__(); self.a = nn.Conv1d(c, c, 3, padding=d, dilation=d); self.b = nn.Conv1d(c, c, 1); self.n = nn.BatchNorm1d(c); self.dp = nn.Dropout(0.2)
    def forward(self, x):
        return x + self.dp(self.b(torch.relu(self.n(self.a(x)))))

class TCN(nn.Module):
    def __init__(self, cin, c=128, ncls=7):
        super().__init__(); self.inp = nn.Conv1d(cin, c, 5, padding=2); self.blocks = nn.Sequential(*[Block(c, d) for d in (1, 2, 4, 8, 1, 2, 4, 8)]); self.out = nn.Conv1d(c, ncls, 1)
    def forward(self, x):                                # x (B, T, C)
        return self.out(self.blocks(torch.relu(self.inp(x.transpose(1, 2))))).transpose(1, 2)

def load(ids):
    return [np.load(os.path.join(ROOT, "external", "punch_ds", c + ".npz")) for c in ids]

@torch.no_grad()
def predict(model, X):
    model.eval(); return torch.softmax(model(torch.from_numpy(X).to(DEV)), -1).cpu().numpy()        # (S, F, 7)

def events(P, M, thr):
    """P (F,7), M (F,) -> [(start, end, class)]."""
    p = 1 - P[:, 0]; p = np.convolve(p, np.ones(3) / 3, mode="same") * M; on = p > thr; F = len(p); out = []; i = 0
    while i < F:
        if not on[i]:
            i += 1; continue
        j = i
        while j < F and (on[j] or (j + 2 < F and on[j + 1:j + 3].any())):
            j += 1
        if j - i >= 3:
            hand = (P[i:j, [2, 4, 6]].sum(1) > P[i:j, [1, 3, 5]].sum(1)).astype(int)      # 1 = right
            cuts = [i] + [i + k for k in range(2, j - i - 2) if hand[k] != hand[k - 1] and (hand[k:k + 3] == hand[k]).all() and (hand[k - 3:k] == hand[k - 1]).all()] + [j]
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a >= 3:
                    out.append((a, b - 1, int(P[a:b, 1:].sum(0).argmax()) + 1))
        i = j
    return out

def labels_of(y):
    out = []; i = 0; F = len(y)
    while i < F:
        if y[i] == 0:
            i += 1; continue
        j = i
        while j < F and y[j] == y[i]:
            j += 1
        out.append((i, j - 1, int(y[i]))); i = j
    return out

def score(Ps, data, thr, shift=0):
    s = dict(lab=0, cnt=0, hit=0, typ=0, hand=0)
    for P, d in zip(Ps, data):
        for b in (0, 1):
            ev = events(P[b], d["M"][b], thr); F = d["Y"].shape[1]
            lab = [((a + shift) % F, (a + shift) % F + (e - a), c) for a, e, c in labels_of(d["Y"][b])]; used = set()
            for a, e, c in lab:
                best, bo = None, 0
                for k, (pa, pe, pc) in enumerate(ev):
                    if k in used:
                        continue
                    o = min(e + TOL, pe) - max(a - TOL, pa) + 1
                    if o > bo:
                        best, bo = k, o
                if best is not None:
                    used.add(best); s["hit"] += 1; s["typ"] += ev[best][2] == c; s["hand"] += (ev[best][2] % 2) == (c % 2)
            s["lab"] += len(lab); s["cnt"] += len(ev)
    r, p = s["hit"] / max(s["lab"], 1), s["hit"] / max(s["cnt"], 1)
    return dict(found=r, real=p, f1=2 * r * p / max(r + p, 1e-9), type=s["typ"] / max(s["hit"], 1), hand=s["hand"] / max(s["hit"], 1), labels=s["lab"], counted=s["cnt"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-opp", action="store_true"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--name", default=None)
    a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ids = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "punch_ds", pre + "*.npz")))
    tr_ids, te_ids = ids("W_train"), ids("W_test"); va_ids, tr_ids = tr_ids[-VAL_N:], tr_ids[:-VAL_N]
    tr, va, te = load(tr_ids), load(va_ids), load(te_ids)
    def X_of(d, n):
        X = d["X"][:n].copy()
        if a.no_opp:
            X[..., OPP] = 0; X[..., -1] = 0
        return X
    Xtr = [X_of(d, 4) for d in tr]; Ytr = [np.where(d["M"], d["Y"], -100) for d in tr]
    model = TCN(Xtr[0].shape[-1]).to(DEV); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2)
    steps = a.epochs * sum(x.shape[1] * 4 for x in Xtr) // (L * 32); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    w = torch.tensor([1.0] + [2.0] * 6, device=DEV); lossf = nn.CrossEntropyLoss(weight=w, ignore_index=-100)
    name = a.name or ("punch-tcn-noopp" if a.no_opp else "punch-tcn"); progress.start(name, total=steps); t0 = time.time()
    for st in range(steps):
        model.train(); xb, yb = [], []
        for _ in range(32):
            k = rng.integers(len(Xtr)); s = rng.integers(4); F = Xtr[k].shape[1]; i = rng.integers(0, F - L)
            xb.append(Xtr[k][s, i:i + L]); yb.append(Ytr[k][s, i:i + L])
        x = torch.from_numpy(np.stack(xb)).to(DEV); y = torch.from_numpy(np.stack(yb)).to(DEV)
        loss = lossf(model(x).reshape(-1, 7), y.reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if st % 20 == 0:
            progress.tick(20, note=f"step {st}/{steps} loss {loss.item():.3f}")
    progress.finish()
    Pva = [predict(model, X_of(d, 2)) for d in va]; Pte = [predict(model, X_of(d, 2)) for d in te]
    thr = max(np.arange(0.2, 0.81, 0.05), key=lambda t: score(Pva, va, t)["f1"])
    res = dict(name=name, seed=a.seed, threshold=float(thr), val=score(Pva, va, thr), test=score(Pte, te, thr), chance=score(Pte, te, thr, shift=210),
               train_seconds=time.time() - t0, device=DEV, steps=steps)
    cm = np.zeros((7, 7), int)
    for P, d in zip(Pte, te):
        for b in (0, 1):
            v = d["M"][b]; np.add.at(cm, (d["Y"][b][v], P[b][v].argmax(1)), 1)
    res["frame_confusion"] = cm.tolist()
    out = os.path.join(ROOT, "runs", name); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"results_seed{a.seed}.json"), "w"), indent=1); torch.save(model.state_dict(), os.path.join(out, f"model_seed{a.seed}.pt"))
    t, c = res["test"], res["chance"]
    print(f"{name} seed {a.seed}: threshold {thr:.2f} (from validation, F1 {res['val']['f1']:.3f}) | {res['train_seconds']:.0f}s on {DEV}")
    print(f"  TEST {t['labels']} labelled punches, {t['counted']} counted: found {t['found']:.1%}, real {t['real']:.1%}, F1 {t['f1']:.3f} | type right {t['type']:.1%}, hand right {t['hand']:.1%}")
    print(f"  chance (labels shifted 7 s): found {c['found']:.1%}, real {c['real']:.1%}, F1 {c['f1']:.3f}")

if __name__ == "__main__":
    main()
