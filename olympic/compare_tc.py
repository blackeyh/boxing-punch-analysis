"""Glove-centred vs target-centred clips for the contact + landed model, and the two averaged.
Target-centred = crop on the opponent's shoulder midpoint (olympic/landed_clips_tc.py), which needs no hand.
A. referee's exact moment (held-out gym)   C. behind our own detector (verdict at the scan peak / weighted)
    python olympic/compare_tc.py
"""
import glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MAP = {"head": 0, "body": 0, "block": 1, "miss": 2, None: -1}

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

def load(pre, ny, N):
    P = np.full((ny, 3), np.nan); S = np.full((N, 13, 4), np.nan)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", pre + "_kam*.npz"))):
        z = np.load(f); P[z["idx"]] = z["P3"]; S[z["chain_idx"]] = z["chain_scan"]
    return P, S

def main():
    L = os.path.join(ROOT, "hpc_rtm", "lists")
    ref = json.load(open(os.path.join(L, "landed_clips.json"))); y = np.array([c["y"] for c in ref])
    reft = json.load(open(os.path.join(L, "landed_clips_tc.json"))); assert (np.array([c["y"] for c in reft]) == y).all()
    clips = json.load(open(os.path.join(L, "chain_clips_v3.json"))); lab = np.array([MAP[c["label"]] for c in clips]); N = len(clips)
    c4 = json.load(open(os.path.join(L, "chain_clips_v4.json"))); assert (np.array([MAP[c["label"]] for c in c4]) == lab).all()
    n_landed = sum(e["outcome"] in ("head", "body") for e in json.load(open(os.path.join(HERE, "events.json"))))
    G = load("cl_large", len(y), N); T = load("cl_tc_large", len(y), N); out = {}
    for name, (P, S) in (("glove-centred", G), ("target-centred", T), ("both averaged", ((G[0] + T[0]) / 2, (G[1] + T[1]) / 2))):
        cm = np.zeros((3, 3), int); np.add.at(cm, (y, P.argmax(1)), 1); f1 = [2 * cm[k, k] / max(cm[k].sum() + cm[:, k].sum(), 1) for k in range(3)]
        print(f"\n{name}\n  A. referee's moment: 3-way right {np.trace(cm) / cm.sum():.1%}, macro-F1 {np.mean(f1):.3f}; caught: landed {cm[0, 0] / cm[0].sum():.0%}, blocked {cm[1, 1] / cm[1].sum():.0%}, missed {cm[2, 2] / cm[2].sum():.0%}; AUC {auc((y == 0).astype(int), P[:, 0]):.3f}")
        w = np.nan_to_num(1 - S[:, :, 3], nan=0.0); k = w.argmax(1); m = lab >= 0
        for how, p3 in (("peak", S[np.arange(N), k, :3]), ("weighted", (np.nan_to_num(S[:, :, :3]) * w[:, :, None]).sum(1))):
            c2 = np.zeros((3, 3), int); np.add.at(c2, (lab[m], p3[m].argmax(1)), 1)
            called = p3.argmax(1) == 0; tp = (called & (lab == 0)).sum(); rc, pr = tp / n_landed, tp / called.sum()
            print(f"  C. behind our detector ({how}): 3-way right {np.trace(c2) / c2.sum():.1%} (landed {c2[0, 0] / c2[0].sum():.0%}, blocked {c2[1, 1] / c2[1].sum():.0%}, missed {c2[2, 2] / c2[2].sum():.0%}); landed end to end: found {rc:.1%}, real {pr:.1%}, F1 {2 * rc * pr / (rc + pr):.3f}")
            out[f"{name} {how}"] = dict(A_acc=float(np.trace(cm) / cm.sum()), C_acc=float(np.trace(c2) / c2.sum()), C_f1=float(2 * rc * pr / (rc + pr)))
    json.dump(out, open(os.path.join(ROOT, "runs", "olympic_compare_tc.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
