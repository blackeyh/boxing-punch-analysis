"""Does adding BoxingWeb punches (contact found by the pixel model, "effective" as a second task) improve the
landed model?  Same folds, epochs and seed; only the training data differs.
  without : external/landed_ft/cl_large_kam*.npz          with : external/landed_ft/cl_bw_large_kam*.npz
A. referee's exact moment and hand (held-out gym)   B. pixel timing on referee clips   C. behind our own detector
    python olympic/compare_bw.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MAP = {"head": 0, "body": 0, "block": 1, "miss": 2, None: -1}

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

def main():
    ref = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips.json"))); y = np.array([c["y"] for c in ref])
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); lab = np.array([MAP[c["label"]] for c in clips]); N = len(clips)
    n_landed = sum(e["outcome"] in ("head", "body") for e in json.load(open(os.path.join(HERE, "events.json")))); out = {}
    for name, pre in (("WITHOUT BoxingWeb", "cl_large"), ("WITH BoxingWeb", "cl_bw_large")):
        fs = sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", pre + "_kam*.npz")))
        if len(fs) < 2:
            print(name, "not complete"); continue
        P = np.full((len(y), 3), np.nan); S = np.full((N, 13, 4), np.nan); tim = []; ep = []
        for f in fs:
            z = np.load(f); P[z["idx"]] = z["P3"]; S[z["chain_idx"]] = z["chain_scan"]; offs = z["scan_offsets"]; tim.append(np.abs(offs[z["scan"][:, :, 3].argmin(1)])); ep.append(int(z["best_epoch"]))
        cm = np.zeros((3, 3), int); np.add.at(cm, (y, P.argmax(1)), 1); f1 = [2 * cm[k, k] / max(cm[k].sum() + cm[:, k].sum(), 1) for k in range(3)]; tim = np.concatenate(tim)
        k = np.nan_to_num(1 - S[:, :, 3], nan=-1).argmax(1); p3 = S[np.arange(N), k, :3]; m = lab >= 0; c2 = np.zeros((3, 3), int); np.add.at(c2, (lab[m], p3[m].argmax(1)), 1)
        called = p3.argmax(1) == 0; tp = (called & (lab == 0)).sum(); rc, pr = tp / n_landed, tp / called.sum()
        print(f"\n{name} (best epochs {ep})")
        print(f"  A. referee's moment: 3-way right {np.trace(cm) / cm.sum():.1%}, macro-F1 {np.mean(f1):.3f} (landed {f1[0]:.2f}, blocked {f1[1]:.2f}, missed {f1[2]:.2f}); caught: landed {cm[0, 0] / cm[0].sum():.0%}, blocked {cm[1, 1] / cm[1].sum():.0%}, missed {cm[2, 2] / cm[2].sum():.0%}; landed-vs-not AUC {auc((y == 0).astype(int), P[:, 0]):.3f}")
        print(f"  B. pixel timing on referee clips: within +-2 frames {np.mean(tim <= 2):.1%}, +-4 {np.mean(tim <= 4):.1%}")
        print(f"  C. behind our detector: 3-way right {np.trace(c2) / c2.sum():.1%} (landed {c2[0, 0] / c2[0].sum():.0%}, blocked {c2[1, 1] / c2[1].sum():.0%}, missed {c2[2, 2] / c2[2].sum():.0%}); landed end to end: found {rc:.1%}, real {pr:.1%}, F1 {2 * rc * pr / (rc + pr):.3f}")
        out[name] = dict(A_acc=float(np.trace(cm) / cm.sum()), A_f1=float(np.mean(f1)), A_auc=float(auc((y == 0).astype(int), P[:, 0])), B_pm2=float(np.mean(tim <= 2)), C_acc=float(np.trace(c2) / c2.sum()), C_f1=float(2 * rc * pr / (rc + pr)))
    json.dump(out, open(os.path.join(ROOT, "runs", "olympic_compare_bw.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
