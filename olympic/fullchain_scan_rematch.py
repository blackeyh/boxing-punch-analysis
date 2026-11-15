"""Re-score the landed chain with a fairer label-to-punch matching. The referee names no boxer; the original score gave
each label to the FIRST counted punch (either boxer) whose span contains it. In exchanges both boxers have a counted
punch there (48% of labels), so the label often lands on the other boxer's punch. Here: one-to-one assignment over all
time-compatible pairs, as many matches as possible, then the smallest distance between the counted punch's glove and the
referee's box. The counted punches, clips and model outputs are unchanged; only which label each punch is compared with.
    python olympic/fullchain_scan_rematch.py [cl_large]
"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MAP = {"head": 0, "body": 0, "block": 1, "miss": 2}; TOL = 10

def main():
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); ev = json.load(open(os.path.join(HERE, "events.json"))); N = len(clips)
    S = np.full((N, 13, 4), np.nan)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", (sys.argv[1] if len(sys.argv) > 1 else "cl_large") + "_kam*.npz"))):
        z = np.load(f); S[z["chain_idx"]] = z["chain_scan"]; offs = z["scan_offsets"]
    cen = 1 - S[:, :, 3]; k = np.nan_to_num(cen, nan=-1).argmax(1); d = offs[k]; verdict = S[np.arange(N), k, :3].argmax(1); have = ~np.isnan(S[:, 0, 0])
    by = {}; idx = {}
    for e in ev:
        by.setdefault("O_" + e["video"], []).append(e)
    for n, c in enumerate(clips):
        idx.setdefault(c["id"], []).append(n)
    n_landed = sum(e["outcome"] in ("head", "body") for e in ev); out = {}
    for name in ("first counted punch in time (original)", "nearest glove to the referee's box"):
        rows = []                                                  # (verdict, label, hand ok, timing error, glove distance in crop sides)
        for cid, ns in idx.items():
            L = [e for e in by[cid]]; C = np.full((len(L), len(ns)), 1e6)
            for i, e in enumerate(L):
                m = (e["start"] + e["end"]) / 2; b = np.array(e["boxes"]); bc = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
                for j, n in enumerate(ns):
                    c = clips[n]
                    if c["f0"] - TOL <= m <= c["f1"] + TOL and have[n]:
                        C[i, j] = np.hypot(c["cx"] - bc[0], c["cy"] - bc[1]) / max(c["side"], 1.0) if name.startswith("nearest") else j * 1e-3
            if name.startswith("nearest"):
                ri, ci = linear_sum_assignment(C); pairs = [(i, j) for i, j in zip(ri, ci) if C[i, j] < 1e5]
            else:
                used = set(); pairs = []
                for i in range(len(L)):
                    for j in range(len(ns)):
                        if j not in used and C[i, j] < 1e5:
                            used.add(j); pairs.append((i, j)); break
            for i, j in pairs:
                e, n = L[i], ns[j]; c = clips[n]; m = (e["start"] + e["end"]) / 2; b = np.array(e["boxes"]); bc = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
                rows.append((verdict[n], MAP[e["outcome"]], c["hand"] == e["hand"], c["coarse"] + d[n] - m, np.hypot(c["cx"] - bc[0], c["cy"] - bc[1]) / max(c["side"], 1.0), c["coarse"] - m))
        r = np.array(rows, float); y, yh = r[:, 1].astype(int), r[:, 0].astype(int); cm = np.zeros((3, 3), int); np.add.at(cm, (y, yh), 1)
        tp = ((yh == 0) & (y == 0)).sum(); called = (have & (verdict == 0)).sum(); rc, pr = tp / n_landed, tp / called
        print(f"\n{name}: {len(r)} labels matched to a counted punch ({len(r) / len(ev):.1%} of labels)")
        print(f"  3-way right {np.trace(cm) / cm.sum():.1%} (always 'landed' {np.mean(y == 0):.1%}); missed caught {cm[2, 2] / cm[2].sum():.0%}, blocked caught {cm[1, 1] / cm[1].sum():.0%}, landed kept {cm[0, 0] / cm[0].sum():.0%}")
        print(f"  hand right {r[:, 2].mean():.1%} | timing within +-2 frames {np.mean(np.abs(r[:, 3]) <= 2):.1%}, +-5 {np.mean(np.abs(r[:, 3]) <= 5):.1%} (rough moment +-12: {np.mean(np.abs(r[:, 5]) <= 12):.1%}) | glove within half a crop of the referee's box {np.mean(r[:, 4] <= 0.5):.1%}")
        print(f"  landed end to end: found {rc:.1%}, real {pr:.1%}, F1 {2 * rc * pr / (rc + pr):.3f}")
        good = (r[:, 2] > 0) & (np.abs(r[:, 5]) <= 12); print(f"  hand right AND rough moment within the scan's reach: {good.mean():.1%} of matches, verdict right {np.mean(yh[good] == y[good]):.1%}; the rest {np.mean(yh[~good] == y[~good]):.1%}")
        out[name] = dict(n=len(r), acc=float(np.trace(cm) / cm.sum()), cm=cm.tolist(), hand=float(r[:, 2].mean()), pm2=float(np.mean(np.abs(r[:, 3]) <= 2)), found=float(rc), real=float(pr))
    json.dump(out, open(os.path.join(ROOT, "runs", "olympic_fullchain_scan_rematch.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
