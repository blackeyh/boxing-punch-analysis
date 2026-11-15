"""Training-free stance detector: orthodox (left side forward) or southpaw (right side forward).

Per frame, from the MediaPipe 3D skeleton (metres, hip-centred, y down, smaller z = nearer the camera):
  forward  = up x (right side - left side), using the hip and shoulder lines, flattened to horizontal
             (for a boxer facing the camera this points at the camera);
  lead     = for each left/right joint pair (ankles, knees, hips, shoulders), (left - right) . forward
             > 0 means the LEFT joint is further forward.
Score per frame = mean over the pairs that are visible (visibility >= 0.5); the stance over a
stretch of time is the sign of the MEDIAN score (robust to single-frame flips). Southpaw clips are
mirrored before the model sees them, exactly like V4 in training.
    /opt/anaconda3/bin/python external/stance.py          # check on our own 10 videos (V4 = southpaw)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PAIRS = [(15, 16), (13, 14), (11, 12), (5, 6)]          # COCO left/right: ankles, knees, hips, shoulders
UP = np.array([0.0, -1.0, 0.0])


def stance_scores(world, vis, present, thr=0.5):
    """world (F,17,3), vis (F,17), present (F,). Returns per-frame score (F,), NaN where unknown.
    > 0: left side forward (orthodox), < 0: right side forward (southpaw)."""
    w = world.astype(np.float32); v = vis.astype(np.float32)
    lat = (w[:, 12] - w[:, 11]) + (w[:, 6] - w[:, 5])               # left -> right, hips + shoulders
    fwd = np.cross(UP, lat); fwd[:, 1] = 0
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9
    num = np.zeros(len(w)); den = np.zeros(len(w))
    for l, r in PAIRS:
        ok = (v[:, l] >= thr) & (v[:, r] >= thr)
        d = np.einsum("fc,fc->f", w[:, l] - w[:, r], fwd)
        num += np.where(ok, d, 0); den += ok
    s = np.where((den > 0) & present, num / np.maximum(den, 1), np.nan)
    return s


def stance_scores_2d(xy, vis, present, W, H, thr=0.5, min_turn=0.15):
    """Image cue for side-on views: the boxer faces the side the nose is on (relative to the
    shoulder centre); the lead joint is the one further that way. Frames where the boxer squarely
    faces the camera (|nose - shoulders| < min_turn torso lengths) give no vote. Units: torso lengths."""
    p = xy.astype(np.float32) * [W, H]; v = vis.astype(np.float32)
    sh = (p[:, 5] + p[:, 6]) / 2
    torso = np.linalg.norm(sh - (p[:, 11] + p[:, 12]) / 2, axis=1) + 1e-6
    turn = (p[:, 0, 0] - sh[:, 0]) / torso
    num = np.zeros(len(p)); den = np.zeros(len(p))
    for l, r in PAIRS:
        ok = (v[:, l] >= thr) & (v[:, r] >= thr)
        num += np.where(ok, (p[:, l, 0] - p[:, r, 0]) * np.sign(turn) / torso, 0); den += ok
    return np.where((den > 0) & present & (np.abs(turn) > min_turn), num / np.maximum(den, 1), np.nan)


SCALE_3D, SCALE_2D = 0.05, 0.4          # typical size of each cue (metres / torso lengths) on our videos


def combined(s3, s2):
    """One number per stretch: each cue's median divided by its typical size, summed (missing = 0)."""
    m3 = np.nanmedian(s3) / SCALE_3D if np.isfinite(s3).any() else 0.0
    m2 = np.nanmedian(s2) / SCALE_2D if np.isfinite(s2).any() else 0.0
    return m3 + m2


def stance_track(xy, world, vis, present, W, H, fps, sec=10.0):
    """Stance for each frame, decided per stretch of `sec` seconds (fighters switch stance now and
    then); a stretch with no evidence inherits the whole-clip call. Returns (+1 orthodox / -1 southpaw
    per frame, whole-clip score, per-stretch scores)."""
    s3 = stance_scores(world, vis, present); s2 = stance_scores_2d(xy, vis, present, W, H)
    whole = combined(s3, s2); n = max(1, int(sec * fps)); per = np.zeros(len(s3)); wins = []
    for a in range(0, len(s3), n):
        c = combined(s3[a:a + n], s2[a:a + n]); wins.append(c)
        per[a:a + n] = np.sign(c) if c != 0 else (1 if whole >= 0 else -1)
    return per, whole, np.array(wins)


if __name__ == "__main__":
    from baselines import MIRROR_IDX, SOUTHPAW_VIDEOS
    for v in range(1, 11):
        fr = np.load(ROOT / "detect" / "frames" / f"V{v}.npz")
        world, vis = fr["world"].astype(np.float32), fr["vis"].astype(np.float32)
        if v in SOUTHPAW_VIDEOS:                                     # frames were mirrored to orthodox: undo
            world, vis = world[:, MIRROR_IDX].copy(), vis[:, MIRROR_IDX]; world[..., 0] *= -1
        xy = fr["xy"].astype(np.float32)
        if v in SOUTHPAW_VIDEOS:
            xy = xy[:, MIRROR_IDX].copy(); xy[..., 0] = np.where(fr["present"][:, None], 1 - xy[..., 0], 0)
        per, whole, wins = stance_track(xy, world, vis, fr["present"], float(fr["W"]), float(fr["H"]), float(fr["fps"]))
        truth = -1 if v in SOUTHPAW_VIDEOS else 1
        print(f"V{v:<2} truth {'southpaw' if truth < 0 else 'orthodox':9s} whole-video call "
              f"{'RIGHT' if np.sign(whole) == truth else 'WRONG'} (score {whole:+.1f}) | "
              f"10-s stretches right {np.mean(np.sign(wins) == truth):.0%} | frames right {np.mean(per == truth):.0%}")
