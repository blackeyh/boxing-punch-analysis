"""Step 3: pose ceiling on real bouts, judged at the human-labelled punches.

For every labelled punch (olympic/events.json) inside the frames a pose dump covers, on the processed frame(s)
within the event (+-1 processed frame):
  puncher seen : some skeleton has a WRIST inside the referee's box (grown by MARGIN px)
  both seen    : (head/body/block events only) a SECOND skeleton has a head/torso keypoint inside the grown box
A punch nobody's skeleton reaches cannot be counted by any classifier on top.
    python olympic/pose_ceiling.py [ids...]          (compares external/pose (MediaPipe) and external/pose_rtm on common frames)
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MARGIN = 30
WR, HEAD, TORSO = [9, 10], [0, 1, 2, 3, 4], [5, 6, 11, 12]

def load(tool, cid):
    if tool == "mediapipe":
        p = ROOT / "external" / "pose" / f"{cid}.npz"
        if not p.exists(): return None
        d = np.load(p); W, H = int(d["width"]), int(d["height"])
        img = d["img"][:, :, MP2COCO]
        kp = img[..., :2] * [W, H]; sc = np.nan_to_num(img[..., 3]); n = d["npose"]
        return dict(frame=d["frame"], kp=kp, sc=sc, n=n, thr=0.5)
    p = ROOT / "external" / "pose_rtm" / f"{cid}.npz"
    if not p.exists(): return None
    d = np.load(p)
    return dict(frame=d["frame"], kp=d["kp"], sc=d["score"], n=d["n"], thr=0.3)

def inside(pts, box):
    x1, y1, x2, y2 = box
    return (pts[..., 0] >= x1 - MARGIN) & (pts[..., 0] <= x2 + MARGIN) & (pts[..., 1] >= y1 - MARGIN) & (pts[..., 1] <= y2 + MARGIN)

def judge(P, e):
    """-> (puncher seen, both seen) for one event, or None if not covered."""
    fr = P["frame"]
    i0, i1 = np.searchsorted(fr, e["start"]) - 1, np.searchsorted(fr, e["end"], "right") + 1
    i0, i1 = max(i0, 0), min(i1, len(fr))
    if e["end"] > fr[-1] or e["start"] < fr[0] or i1 <= i0:
        return None
    bx = np.array(e["boxes"]); box = [bx[:, 1].min(), bx[:, 2].min(), bx[:, 3].max(), bx[:, 4].max()]
    punch = both = False
    for i in range(i0, i1):
        n = int(P["n"][i])
        if n == 0: continue
        kp, sc = P["kp"][i, :n], P["sc"][i, :n]
        ok = (sc >= P["thr"]) & ~np.isnan(kp[..., 0])
        w = (inside(kp[:, WR], box) & ok[:, WR]).any(1)                       # per person: wrist in box
        b = (inside(kp[:, HEAD + TORSO], box) & ok[:, HEAD + TORSO]).any(1)   # per person: head/torso in box
        if w.any():
            punch = True
            for p in np.flatnonzero(w):
                if np.delete(b, p).any(): both = True
    return punch, both

def main():
    ev = json.load(open(ROOT / "olympic" / "events.json"))
    ids = sys.argv[1:] or sorted(p.stem for p in (ROOT / "external" / "pose_rtm").glob("O_*.npz"))
    tot = {}
    for cid in ids:
        P = {t: load(t, cid) for t in ("mediapipe", "rtmpose")}
        if any(v is None for v in P.values()): print(cid, "missing a dump"); continue
        last = min(int(v["frame"][-1]) for v in P.values())
        es = [e for e in ev if "O_" + e["video"] == cid and e["end"] <= last]
        row = []
        for t in P:
            r = [judge(P[t], e) for e in es]
            contact = [x for x, e in zip(r, es) if e["outcome"] != "miss"]
            a, b = np.mean([x[0] for x in r]), np.mean([x[1] for x in contact])
            people = np.median(P[t]["n"][P[t]["frame"] <= last])
            tot.setdefault(t, []).append((len(r), a, len(contact), b))
            row.append(f"{t}: puncher seen {a:.0%}, both seen {b:.0%}, people/frame {people:.0f}")
        print(f"{cid} ({len(es)} punches, first {last/50:.0f}s) | " + " | ".join(row))
    for t, v in tot.items():
        v = np.array(v)
        print(f"ALL {t}: puncher seen {np.average(v[:,1], weights=v[:,0]):.1%} of {int(v[:,0].sum())} punches; "
              f"both seen {np.average(v[:,3], weights=v[:,2]):.1%} of {int(v[:,2].sum())} contact punches")

if __name__ == "__main__":
    main()
