"""Skeleton-only features per person and frame for the learned boxer selector (no pixels, no colour, no absolute
image position, so nothing is tied to one camera set-up).

From a +-1 s window of the person's tracklet:
  pose   : mean and spread of 12 joints (shoulders..ankles) relative to the hips, in body-scale units   (48)
  motion : wrist speeds, hip speed, hip mobility                                                          (5)
  quality: mean keypoint confidence, tracklet length                                                       (2)
  size   : body scale relative to the largest in the frame, rank among the people in the frame, people count (3)
  partner: the nearest person of similar apparent size (= same distance from the camera): distance, size ratio,
           their wrist heights and arm speed, how many similar-sized people are within 4 body scales       (7)
    python boxingweb/feats.py [PREFIX ...]        (caches external/pose_rtm/{id}.f2.npz; needs select_boxers tracklets)
"""
import glob, os, sys, warnings
from multiprocessing import Pool
import numpy as np
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "olympic"))
import select_boxers as sb
WIN = 25
JOINTS = list(range(5, 17))
NAMES = ([f"m_j{j}{a}" for j in JOINTS for a in "xy"] + [f"s_j{j}{a}" for j in JOINTS for a in "xy"] +
         ["wspd_l", "wspd_r", "hip_spd", "mobility", "arm_spd_max", "kp_score", "lifetime", "size_rel", "size_rank", "n_people",
          "p_dist", "p_ratio", "p_wl", "p_wr", "p_arm", "p_count", "p_none"])

def compute(D):
    kp, sc, n, tid = D["kp"], D["sc"], D["n"], D["tid"]; F, K = tid.shape
    sh = (kp[:, :, 5] + kp[:, :, 6]) / 2; hip = (kp[:, :, 11] + kp[:, :, 12]) / 2
    torso = np.linalg.norm(sh - hip, axis=-1)
    d = lambda a, b: np.linalg.norm(kp[:, :, a] - kp[:, :, b], axis=-1)
    leg = np.maximum(d(11, 13) + d(13, 15), d(12, 14) + d(14, 16)); legok = (sc[:, :, 13:17] > 0.3).all(-1)
    raw_scale = np.maximum(np.where(legok, np.maximum(torso, 0.55 * leg), torso), 8.0)
    rel = (kp[:, :, JOINTS] - hip[:, :, None]) / raw_scale[..., None, None]            # (F,K,12,2)
    rel = np.clip(np.nan_to_num(rel), -6, 6).reshape(F, K, 24)
    W = np.full((F, K, 48 + 7), np.nan, np.float32); S = np.full((F, K), np.nan, np.float32); AR = np.full((F, K), np.nan, np.float32)
    fs, ks = np.nonzero(tid >= 0); t = tid[fs, ks]
    order = np.lexsort((fs, t)); fs, ks, t = fs[order], ks[order], t[order]
    bounds = np.flatnonzero(np.diff(t)) + 1
    for f, k in zip(np.split(fs, bounds), np.split(ks, bounds)):
        m = len(f); a = np.searchsorted(f, f - WIN); b = np.searchsorted(f, f + WIN, "right"); cnt = (b - a)[:, None].astype(np.float64)
        win = lambda v: ((np.r_[np.zeros((1, v.shape[1])), np.cumsum(v, 0)])[b] - (np.r_[np.zeros((1, v.shape[1])), np.cumsum(v, 0)])[a]) / cnt
        r = rel[f, k].astype(np.float64); mu = win(r); sd = np.sqrt(np.maximum(win(r * r) - mu * mu, 0))
        scl = raw_scale[f, k]; hp = hip[f, k]; dt = np.maximum(np.diff(f), 1)[:, None]
        hv = np.r_[np.zeros((1, 2)), np.diff(hp, axis=0) / dt] / scl[:, None]
        wr = (kp[f, k][:, [9, 10]] - hp[:, None]) / scl[:, None, None]
        wv = np.r_[np.zeros((1, 2)), np.linalg.norm(np.diff(wr, axis=0), axis=-1) / dt]
        hpn = hp / scl.mean(); mob = np.sqrt(np.maximum(win(hpn * hpn) - win(hpn) ** 2, 0)).sum(1, keepdims=True)
        mot = np.c_[win(wv), win(np.linalg.norm(hv, axis=1, keepdims=True)), mob, win(wv.max(1, keepdims=True))]
        q = np.c_[win(sc[f, k].mean(1, keepdims=True)), np.full((m, 1), min((f[-1] - f[0]) / 25.0, 20.0))]
        W[f, k] = np.c_[mu, sd, mot, q]
        S[f, k] = win(scl[:, None])[:, 0]; AR[f, k] = mot[:, 4]
    X = np.full((F, K, len(NAMES)), np.nan, np.float32); X[..., :55] = W
    for f in range(F):
        m = int(n[f])
        if m == 0:
            continue
        s = S[f, :m]; X[f, :m, 55] = s / np.nanmax(s); X[f, :m, 56] = (np.argsort(np.argsort(-s))) / max(m - 1, 1); X[f, :m, 57] = m
        dist = np.linalg.norm(hip[f, :m, None] - hip[f, None, :m], axis=-1) / np.maximum(s[:, None], s[None])
        ratio = s[:, None] / s[None]; ok = (ratio > 1 / 1.5) & (ratio < 1.5) & ~np.eye(m, dtype=bool)
        dm = np.where(ok, dist, np.inf); j = dm.argmin(1); has = np.isfinite(dm.min(1))
        X[f, :m, 58] = np.where(has, np.minimum(dm.min(1), 20), 20); X[f, :m, 59] = np.where(has, ratio[np.arange(m), j], 0)
        X[f, :m, 60] = np.where(has, W[f, j, 9], 0); X[f, :m, 61] = np.where(has, W[f, j, 11], 0)      # partner's wrist heights (y of joints 9, 10)
        X[f, :m, 62] = np.where(has, AR[f, j], 0); X[f, :m, 63] = (ok & (dist < 4)).sum(1); X[f, :m, 64] = ~has
    return X, S

def load(cid):
    D = sb.load(cid); cache = os.path.join(ROOT, "external", "pose_rtm", cid + ".f2.npz")
    if os.path.exists(cache):
        c = np.load(cache); D["X2"], D["S"] = c["X"], c["S"]
    else:
        D["X2"], D["S"] = compute(D); np.savez_compressed(cache, X=D["X2"], S=D["S"])
    return D

def _run(cid):
    D = load(cid); return f"{cid[:50]} frames {len(D['frame'])} tracklets {D['tid'].max() + 1}"

if __name__ == "__main__":
    pre = sys.argv[1:] or ["W_", "O_"]
    ids = sorted(os.path.basename(p)[:-4] for q in pre for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", q + "*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as p:
        for r in p.imap_unordered(_run, ids):
            print(r, flush=True)
    print("ALL DONE")
