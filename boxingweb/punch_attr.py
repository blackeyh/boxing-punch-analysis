"""LANDED or not (and target), per punch, from two-person RTMPose skeletons.  BoxingWeb labels:
  effect : effective (landed cleanly, 26%) vs ineffective        target: head vs torso (chest + abdomen)
Input per punch: the puncher's stream from 8 frames before the start to 12 frames after the end (the aftermath shows
the target's reaction), resampled to 40 steps: own pose + change, OPPONENT pose + change (punch_data channels),
plus which hand punched and a "punch in progress" marker. Punch segments and hand come from the labels here, so this
measures the landed/target decision alone.
Train: data_train minus the last 5 rounds (validation: picks the decision threshold); test: data_test.
Reference, BoxMind Table S3 (F1): effect 0.621 pose-only, 0.645 best; target 0.747 pose-only, 0.764 best.
    python boxingweb/punch_attr.py [--no-opp] [--seed 0]
"""
import argparse, glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mem_guard  # noqa: F401
import numpy as np, torch, torch.nn as nn
from progress import progress
import punch_data as pd, punch_tcn as pt
PRE, POST, T = 8, 12, 40

def windows(cid, mirror_too):
    d = np.load(os.path.join(ROOT, "external", "punch_ds", cid + ".npz")); F = d["X"].shape[1]
    ev = [e for e in json.load(open(os.path.join(HERE, "events.json"))) if e["video"] == cid and pd.usable(e, F)]
    X, ye, yt = [], [], []
    for e in ev:
        b = 0 if e["side"] == "red" else 1; a, z = max(e["start"] - PRE, 0), min(e["end"] + POST, F - 1)
        if d["M"][b, e["start"]:e["end"] + 1].mean() < 0.5:
            continue                                         # the puncher has no skeleton during the punch
        src = np.linspace(a, z, T); lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, F - 1); w = (src - lo)[:, None]
        act = ((src >= e["start"]) & (src <= e["end"])).astype(np.float32)[:, None]
        for mir in ((False, True) if mirror_too else (False,)):
            x = d["X"][b + (2 if mir else 0)]; hand = float((e["hand"] == "right") != mir)
            X.append(np.concatenate([x[lo] * (1 - w) + x[hi] * w, np.full((T, 1), hand, np.float32), act], 1).astype(np.float32))
            ye.append(int(e["landed"])); yt.append(int(e["target"] != "head"))
    return np.array(X), np.array(ye), np.array(yt)

class Net(nn.Module):
    def __init__(self, cin, c=128):
        super().__init__(); self.inp = nn.Conv1d(cin, c, 5, padding=2); self.blocks = nn.Sequential(*[pt.Block(c, d) for d in (1, 2, 4, 1, 2, 4)])
        self.dp = nn.Dropout(0.3); self.eff = nn.Linear(2 * c, 2); self.tgt = nn.Linear(2 * c, 2)
    def forward(self, x):
        h = self.blocks(torch.relu(self.inp(x.transpose(1, 2)))); h = self.dp(torch.cat([h.mean(2), h.amax(2)], 1)); return self.eff(h), self.tgt(h)

def f1s(y, p, thr):
    yh = (p >= thr).astype(int); out = []
    for k in (0, 1):
        tp = ((y == k) & (yh == k)).sum(); pr = tp / max((yh == k).sum(), 1); rc = tp / max((y == k).sum(), 1); out.append(2 * pr * rc / max(pr + rc, 1e-9))
    tp = ((y == 1) & (yh == 1)).sum()
    return float(np.mean(out)), float(tp / max((yh == 1).sum(), 1)), float(tp / max((y == 1).sum(), 1)), float((yh == y).mean())

def auc(y, p):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(1, len(p) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-opp", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ids = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "punch_ds", pre + "*.npz")))
    tr_ids, te_ids = ids("W_train"), ids("W_test"); va_ids, tr_ids = tr_ids[-5:], tr_ids[:-5]
    cat = lambda idl, m: [np.concatenate(z) for z in zip(*(windows(c, m) for c in idl))]
    (Xtr, etr, ttr), (Xva, eva, tva), (Xte, ete, tte) = cat(tr_ids, True), cat(va_ids, False), cat(te_ids, False)
    def prep(X):
        X = X.copy()
        if a.no_opp:
            X[..., pt.OPP] = 0; X[..., 156] = 0
        return X
    Xtr, Xva, Xte = prep(Xtr), prep(Xva), prep(Xte)
    model = Net(Xtr.shape[-1]).to(pt.DEV); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=5e-2)
    nb = len(Xtr) // 64; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.epochs * nb)
    we = torch.tensor([1.0, float((etr == 0).sum() / max((etr == 1).sum(), 1))], device=pt.DEV); wt = torch.tensor([1.0, float((ttr == 0).sum() / max((ttr == 1).sum(), 1))], device=pt.DEV)
    name = "punch-attr-noopp" if a.no_opp else "punch-attr"; progress.start(name, total=a.epochs * nb); t0 = time.time()
    Xt, Et, Tt = torch.from_numpy(Xtr), torch.from_numpy(etr), torch.from_numpy(ttr)
    for ep in range(a.epochs):
        model.train(); perm = torch.from_numpy(rng.permutation(len(Xtr)))
        for k in range(nb):
            i = perm[k * 64:(k + 1) * 64]; x = Xt[i].to(pt.DEV)
            x = x + 0.02 * torch.randn_like(x) * (torch.arange(x.shape[-1], device=pt.DEV) < 138)       # jitter the pose channels
            le, lt = model(x); loss = nn.functional.cross_entropy(le, Et[i].to(pt.DEV), weight=we) + 0.5 * nn.functional.cross_entropy(lt, Tt[i].to(pt.DEV), weight=wt)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        progress.tick(nb, note=f"epoch {ep + 1}/{a.epochs} loss {loss.item():.3f}")
    progress.finish()
    @torch.no_grad()
    def pred(X):
        model.eval(); le, lt = model(torch.from_numpy(X).to(pt.DEV)); return torch.softmax(le, 1)[:, 1].cpu().numpy(), torch.softmax(lt, 1)[:, 1].cpu().numpy()
    pev, ptv = pred(Xva); pe, ptt = pred(Xte); grid = np.arange(0.1, 0.91, 0.05)
    the = max(grid, key=lambda t: f1s(eva, pev, t)[0]); tht = max(grid, key=lambda t: f1s(tva, ptv, t)[0])
    fe, ft = f1s(ete, pe, the), f1s(tte, ptt, tht)
    res = dict(name=name, seed=a.seed, n_train=int(len(Xtr) // 2), n_test=int(len(Xte)), landed_share_test=float(ete.mean()),
               effect=dict(macro_f1=fe[0], precision_landed=fe[1], recall_landed=fe[2], accuracy=fe[3], auc=auc(ete, pe), threshold=float(the)),
               target=dict(macro_f1=ft[0], accuracy=ft[3], auc=auc(tte, ptt), torso_share=float(tte.mean())), seconds=time.time() - t0)
    out = os.path.join(ROOT, "runs", name); os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"results_seed{a.seed}.json"), "w"), indent=1); torch.save(model.state_dict(), os.path.join(out, f"model_seed{a.seed}.pt"))
    print(f"{name} seed {a.seed}: train {res['n_train']} punches, test {res['n_test']} ({ete.mean():.0%} landed) | {res['seconds']:.0f}s")
    print(f"  LANDED: macro-F1 {fe[0]:.3f}, AUC {res['effect']['auc']:.3f}, accuracy {fe[3]:.1%}; of those called landed {fe[1]:.1%} are, and {fe[2]:.1%} of landed punches are caught")
    print(f"  TARGET head/torso: macro-F1 {ft[0]:.3f}, AUC {res['target']['auc']:.3f}, accuracy {ft[3]:.1%} (torso {tte.mean():.0%})")

if __name__ == "__main__":
    main()
