"""Per-boxer frame streams for punch detection on BoxingWeb (RTMPose skeletons).

For each round and each boxer (red, blue; the RTMPose person matched to the truth pose), per frame:
  own pose (17x2, hip-centred, body-scale units) + its frame-to-frame change + hip speed
  OPPONENT's pose in the same coordinates + its change          (two-person input)
  own keypoint confidences, "own skeleton present", "opponent present"
Short gaps (<= 8 frames) in a boxer's skeleton are filled by interpolation.
Labels per frame: 0 no punch, 1 l-straight, 2 r-straight, 3 l-hook, 4 r-hook, 5 l-uppercut, 6 r-uppercut.
Label cleaning: punches shorter than 2 frames or longer than 45 (1.5 s) or out of range are unusable and dropped
(reported); when two punches of one boxer overlap, the later one takes over from its start.
A mirrored copy (left <-> right) of every stream is stored for training.
-> external/punch_ds/{id}.npz: X (4, F, C) [red, blue, red mirrored, blue mirrored], Y (4, F), M (4, F) own present.
    python boxingweb/punch_data.py
"""
import glob, json, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import median_filter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CLASSES = ["none", "lstraight", "rstraight", "lhook", "rhook", "luppercut", "ruppercut"]
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
MAXGAP, MINLEN, MAXLEN = 8, 2, 45

def usable(e, n):
    L = e["end"] - e["start"] + 1
    return MINLEN <= L <= MAXLEN and e["end"] < n and e["start"] >= 0

def stream(kp, sc, role, r):
    F = len(role); K = np.full((F, 17, 2), np.nan); C = np.zeros((F, 17), np.float32)
    fs, ps = np.nonzero(role == r); K[fs, ps * 0 + np.arange(1)[:, None].repeat(len(fs), 1)[0]] if False else None
    K[fs] = kp[fs, ps]; C[fs] = sc[fs, ps]
    have = ~np.isnan(K[:, 0, 0]); idx = np.flatnonzero(have); M = have.copy()
    if len(idx) > 1:
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= MAXGAP:
                w = ((np.arange(a + 1, b) - a) / (b - a))[:, None, None]
                K[a + 1:b] = K[a] * (1 - w) + K[b] * w; C[a + 1:b] = np.minimum(C[a], C[b]) * 0.5; M[a + 1:b] = True
    return K, C, M

def feats(K, C, M, Ko, Mo):
    hip = K[:, [11, 12]].mean(1); sh = K[:, [5, 6]].mean(1); torso = np.linalg.norm(sh - hip, axis=1)
    t = np.where(np.isnan(torso), np.nanmedian(torso) if np.isfinite(np.nanmedian(torso)) else 50.0, torso)
    S = np.maximum(median_filter(t, size=25, mode="nearest"), 8.0)[:, None, None]
    own = (K - hip[:, None]) / S; opp = (Ko - hip[:, None]) / S
    d = lambda a: np.r_[np.zeros((1,) + a.shape[1:]), np.diff(a, axis=0)]
    hv = d(hip / S[:, 0]); X = np.concatenate([own.reshape(len(K), -1), d(own).reshape(len(K), -1), hv, opp.reshape(len(K), -1), d(opp).reshape(len(K), -1),
                                               C, M[:, None].astype(np.float32), Mo[:, None].astype(np.float32)], 1)
    return np.clip(np.nan_to_num(X, nan=0.0), -8, 8).astype(np.float32)

def mirror(K, C, W):
    K2 = K[:, FLIP].copy(); K2[..., 0] = W - K2[..., 0]; return K2, C[:, FLIP]

def build(cid):
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); role = np.load(os.path.join(ROOT, "external", "boxers_gt", cid + ".npz"))["role"]
    kp, sc = R["kp"].astype(np.float64), R["score"]; F = len(R["frame"]); W = float(R["width"])
    ev = [e for e in json.load(open(os.path.join(HERE, "events.json"))) if e["video"] == cid]
    st = {r: stream(kp, sc, role, r) for r in (1, 2)}; X, Y, Ms = [], [], []; dropped = 0
    for mir in (False, True):
        for r, side in ((1, "red"), (2, "blue")):
            K, C, M = st[r]; Ko, Co, Mo = st[3 - r]
            if mir:
                K, C = mirror(K, C, W); Ko, Co = mirror(Ko, Co, W)
            y = np.zeros(F, np.int64)
            for e in sorted((e for e in ev if e["side"] == side), key=lambda e: e["start"]):
                if not usable(e, F):
                    dropped += not mir; continue
                c = CLASSES.index(e["technique"]); c = c + (1 if c % 2 else -1) if mir else c
                y[e["start"]:e["end"] + 1] = c
            X.append(feats(K, C, M, Ko, Mo)); Y.append(y); Ms.append(M)
    out = os.path.join(ROOT, "external", "punch_ds"); os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, cid + ".npz"), X=np.stack(X), Y=np.stack(Y), M=np.stack(Ms))
    return cid, F, len(ev), dropped, float(np.mean([st[r][2].mean() for r in (1, 2)])), float((np.stack(Y[:2]) > 0).mean())

if __name__ == "__main__":
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as p:
        res = p.map(build, ids)
    n = sum(r[2] for r in res); d = sum(r[3] for r in res)
    print(f"{len(res)} rounds, {sum(r[1] for r in res)} frames, {n} punches, {d} dropped as unusable ({d / n:.1%}); boxer skeleton present in {np.mean([r[4] for r in res]):.1%} of boxer-frames; punch frames {np.mean([r[5] for r in res]):.1%}")
    print("lowest skeleton coverage:", sorted((round(r[4], 2), r[0][2:40]) for r in res)[:4])
