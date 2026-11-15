"""TARGET-centred clip lists (same punches and order as the glove-centred lists):
  hpc_rtm/lists/landed_clips_tc.json  referee punches: crop centred on the OPPONENT's shoulder midpoint at contact,
                                      side 3 x the opponent's torso length (head to hips in view)
  hpc_rtm/lists/chain_clips_v4.json   the chain's punches: the other boxer's stream at the rough contact frame
Why: the clip no longer depends on which hand punched (the detector's hand is right only 80% of the time) and does
not jump with wrist jitter.     python olympic/landed_clips_tc.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import landed_cues as lc

def target(k):
    sh = k[[5, 6]].mean(0); S = np.linalg.norm(sh - k[[11, 12]].mean(0))
    return (sh, float(3.0 * max(S, 20.0))) if np.isfinite(sh).all() and np.isfinite(S) else (None, None)

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); old = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips.json"))); out = []; n = 0; fb = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "pose_rtm", "O_*.npz"))):
        if os.path.basename(f).count(".") != 1:
            continue
        cid = os.path.basename(f)[:-4]; R = np.load(f); pick = np.load(os.path.join(ROOT, "external", "boxers_rtm", cid + ".npz"))["pick"]; kp, fr = R["kp"].astype(np.float64), R["frame"]
        for e in [e for e in ev if "O_" + e["video"] == cid]:
            if lc.cues(R, pick, e) is None:
                continue
            o = old[n]; n += 1; assert o["id"] == cid and o["frame"] == (e["start"] + e["end"]) // 2
            i = int(np.clip(np.searchsorted(fr, o["frame"]), 0, len(fr) - 1)); ps = [p for p in pick[i] if p >= 0]
            w = 9 if e["hand"] == "left" else 10; g = [kp[i, p, w] + 0.4 * (kp[i, p, w] - kp[i, p, w - 2]) for p in ps]
            a = int(np.argmin([np.linalg.norm(x - np.array([o["bx"], o["by"]])) for x in g])); c, side = target(kp[i, ps[1 - a]]); d = dict(o)
            if c is None:
                fb += 1
            else:
                d.update(cx=float(c[0]), cy=float(c[1]), side=side)
            out.append(d)
    assert len(out) == len(old); json.dump(out, open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips_tc.json"), "w"))
    print(f"{len(out)} referee clips, target-centred ({fb} fell back to the glove crop); crop side median {np.median([o['side'] for o in out]):.0f} px")
    v3 = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); cache = {}; v4 = []; fb = 0
    for c in v3:
        if c["id"] not in cache:
            z = np.load(os.path.join(ROOT, "external", "chain_streams", c["id"] + ".npz")); cache[c["id"]] = (z["K"].astype(np.float64), z["frame"])
        K, fr = cache[c["id"]]; step = int(fr[1] - fr[0]); t = int(np.clip(round(c["frame"] / step), 0, len(fr) - 1)); s = "AB".index(c["boxer"]); ctr, side = target(K[1 - s, t]); d = dict(c)
        if ctr is None:
            fb += 1
        else:
            d.update(cx=float(ctr[0]), cy=float(ctr[1]), side=side)
        v4.append(d)
    json.dump(v4, open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v4.json"), "w")); print(f"{len(v4)} chain clips, target-centred ({fb} fell back to the glove crop: no opponent skeleton)")

if __name__ == "__main__":
    main()
