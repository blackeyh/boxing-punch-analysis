"""List of contact clips for the pixel landed/blocked/missed model -> hpc_rtm/lists/landed_clips.json.
Per usable Olympic punch: video, contact frame (video frame, 50 fps), crop centre = the PUNCHER'S GLOVE at contact
(from the skeleton, available at run time), crop side = 2.4 body scales, label, gym; plus the referee's box centre
for an upper-bound crop.   python olympic/landed_clips.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import landed_cues as lc

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "pose_rtm", "O_*.npz"))):
        if os.path.basename(f).count(".") != 1:
            continue
        cid = os.path.basename(f)[:-4]; R = np.load(f); pick = np.load(os.path.join(ROOT, "external", "boxers_rtm", cid + ".npz"))["pick"]
        kp, fr = R["kp"].astype(np.float64), R["frame"]
        for e in [e for e in ev if "O_" + e["video"] == cid]:
            if lc.cues(R, pick, e) is None:
                continue
            i = int(np.clip(np.searchsorted(fr, (e["start"] + e["end"]) // 2), 0, len(fr) - 1)); ps = [p for p in pick[i] if p >= 0]
            w = 9 if e["hand"] == "left" else 10; g = [kp[i, p, w] + 0.4 * (kp[i, p, w] - kp[i, p, w - 2]) for p in ps]
            b = np.array(e["boxes"]); c = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
            a = int(np.argmin([np.linalg.norm(x - c) for x in g])); p = ps[a]
            S = max(np.linalg.norm(kp[i, p, [5, 6]].mean(0) - kp[i, p, [11, 12]].mean(0)), 20.0)
            out.append(dict(id=cid, frame=int((e["start"] + e["end"]) // 2), cx=float(g[a][0]), cy=float(g[a][1]), side=float(2.4 * S), bx=float(c[0]), by=float(c[1]),
                            y={"head": 0, "body": 0, "block": 1, "miss": 2}[e["outcome"]], outcome=e["outcome"], hand=e["hand"], gym=cid[2:6]))
    json.dump(out, open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips.json"), "w"))
    d = np.array([np.hypot(o["cx"] - o["bx"], o["cy"] - o["by"]) / o["side"] for o in out])
    print(len(out), "clips; glove-to-referee-box distance / crop side: median", round(float(np.median(d)), 2), "90th pct", round(float(np.percentile(d, 90)), 2), "| crop side px median", int(np.median([o["side"] for o in out])))

if __name__ == "__main__":
    main()
