"""BoxingWeb punches as clips for the pixel contact + landed model (user's idea: expand the landed training data).
BoxingWeb gives each punch's start/end, hand, target and effective/ineffective, but NOT the contact moment.
Per usable punch whose puncher has a skeleton: 44 frames centred on the middle of the punch, crop centred on the
labelled hand's glove there (side 2.4 body scales). The contact moment is then found by scanning these clips with
the pixel model.   -> hpc_rtm/lists/bw_clips.json      python boxingweb/contact_clips.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import punch_data as pd

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); out = []; skipped = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_*.npz"))):
        if os.path.basename(f).count(".") != 1:
            continue
        cid = os.path.basename(f)[:-4]; R = np.load(f); role = np.load(os.path.join(ROOT, "external", "boxers_gt", cid + ".npz"))["role"]
        kp, sc = R["kp"].astype(np.float64), R["score"]; F = len(R["frame"]); K = {r: pd.stream(kp, sc, role, r)[0] for r in (1, 2)}
        for e in [e for e in ev if e["video"] == cid and pd.usable(e, F)]:
            r = 1 if e["side"] == "red" else 2; mid = (e["start"] + e["end"]) // 2; w = 9 if e["hand"] == "left" else 10; k = K[r][mid]
            g = k[w] + 0.4 * (k[w] - k[w - 2])
            if not (np.isfinite(g).all() and np.isfinite(k[[5, 6, 11, 12]]).all()):
                skipped += 1; continue
            S = max(np.linalg.norm(k[[5, 6]].mean(0) - k[[11, 12]].mean(0)), 20.0)
            out.append(dict(id=cid, gym="bw", frame=int(mid), cx=float(g[0]), cy=float(g[1]), side=float(2.4 * S), start=e["start"], end=e["end"], landed=bool(e["landed"]),
                            target=e["target"], hand=e["hand"], technique=e["technique"], split=e["split"], fps=float(R["fps"])))
    json.dump(out, open(os.path.join(ROOT, "hpc_rtm", "lists", "bw_clips.json"), "w"))
    L = np.array([o["end"] - o["start"] + 1 for o in out])
    print(f"{len(out)} BoxingWeb punch clips ({skipped} skipped: no skeleton at the middle of the punch); effective {np.mean([o['landed'] for o in out]):.1%}; "
          f"train {sum(o['split'] == 'train' for o in out)}, test {sum(o['split'] == 'test' for o in out)}; punch length frames median {np.median(L):.0f}, 95th pct {np.percentile(L, 95):.0f}; crop side median {np.median([o['side'] for o in out]):.0f} px")

if __name__ == "__main__":
    main()
