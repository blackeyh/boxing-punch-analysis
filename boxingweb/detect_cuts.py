"""Camera cuts in the BoxingWeb broadcast rounds, from the pixels (a fixed phone camera has none).
Per frame: mean absolute difference of 64x36 grey thumbnails to the previous frame. Cut = a spike: difference
> ABS and > REL x the local median (+-12 frames). Output external/cuts/{id}.npz: diff (F,), cut (F,) bool.
    python boxingweb/detect_cuts.py
"""
import glob, os, sys
from multiprocessing import Pool
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABS, REL = 18.0, 4.0

def run(cid):
    out = os.path.join(ROOT, "external", "cuts", cid + ".npz")
    if not os.path.exists(out):
        cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); prev = None; d = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(im, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            d.append(0.0 if prev is None else float(np.abs(g - prev).mean())); prev = g
        d = np.array(d, np.float32)
        med = np.array([np.median(d[max(0, i - 12):i + 13]) for i in range(len(d))])
        np.savez_compressed(out, diff=d, cut=(d > ABS) & (d > REL * np.maximum(med, 1.0)))
    z = np.load(out); return cid, int(z["cut"].sum()), len(z["cut"])

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "external", "cuts"), exist_ok=True)
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as p:
        res = p.map(run, ids)
    n = sum(r[1] for r in res); f = sum(r[2] for r in res)
    print(f"{len(res)} rounds, {f} frames, {n} cuts = {n / (f / 30 / 60):.1f} per minute; per round min {min(r[1] for r in res)} median {int(np.median([r[1] for r in res]))} max {max(r[1] for r in res)}")
