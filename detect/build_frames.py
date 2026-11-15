"""Frame-level dataset for punch detection: every frame of every video, not just punch clips.

For each dataset video v (file mapping from video3d/common.VIDEO_OF, see detect/WINDOWS.md):
  * person tracking: MediaPipe returns up to 3 people per frame; keep the one whose hip
    midpoint (image coords) is nearest the previously tracked one (94-100% of frames have
    exactly one person anyway). Frames with nobody detected are marked present=False.
  * V4 (southpaw) mirrored to orthodox: image x -> 1 - x, world x -> -x, left/right swapped.
  * world 3D: joints with visibility < 0.5 are linearly interpolated in time from that joint's
    confident frames (the step-9 "conf-fill" that rescued MediaPipe 3D: 57.9 -> 72.5%).
  * label per frame: -1 = not inside any labelled punch, else class index (punch_data.CLASSES),
    from the annotation rows punch_data.load_dataset() keeps (valid label, 0 < end-start < 25).

Output detect/frames/V{v}.npz: world (F,17,3) f16 metres, xy (F,17,2) f16 image 0-1,
vis (F,17) f16, present (F,) bool, label (F,) i8, fps, W, H.
    /opt/anaconda3/bin/python detect/build_frames.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baselines import MIRROR_IDX, SOUTHPAW_VIDEOS  # noqa: E402
from punch_data import CLASSES, T  # noqa: E402
from video3d.common import VIDEO_OF, annotations, load_pose  # noqa: E402

OUT = ROOT / "detect" / "frames"
L_HIP, R_HIP = 11, 12


def track(xy, npose):
    """xy: (F, 3, 17, 2). Returns chosen person index per frame (-1 if none)."""
    F = len(npose)
    pick = np.full(F, -1, np.int8)
    last = np.array([0.5, 0.6])
    for f in range(F):
        n = int(npose[f])
        if n == 0:
            continue
        hips = (xy[f, :n, L_HIP] + xy[f, :n, R_HIP]) / 2          # (n, 2)
        p = int(np.argmin(np.linalg.norm(hips - last, axis=1))) if n > 1 else 0
        pick[f] = p
        last = hips[p]
    return pick


def fill_low_vis(world, vis, present, thr=0.5):
    out = world.copy()
    t = np.arange(len(world))
    for j in range(world.shape[1]):
        good = present & (vis[:, j] >= thr)
        if good.sum() < 2:
            continue
        bad = present & ~good
        for c in range(3):
            out[bad, j, c] = np.interp(t[bad], t[good], world[good, j, c])
    return out


def labels_for(v, F):
    lab = np.full(F, -1, np.int8)
    ann = annotations(v)
    dur = ann["end"] - ann["start"]
    keep = ann["label"].isin(CLASSES) & (dur > 0) & (dur < T)
    for _, r in ann[keep].iterrows():
        s, e = int(r["start"]), int(r["end"])
        lab[max(s, 0): min(e, F - 1) + 1] = CLASSES.index(r["label"])
    return lab, int(keep.sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for v in range(1, 11):
        P = load_pose(VIDEO_OF[v])
        F = len(P["npose"])
        pick = track(P["xy"], P["npose"])
        present = pick >= 0
        idx = np.clip(pick, 0, None)
        f = np.arange(F)
        xy, world, vis = P["xy"][f, idx], P["world"][f, idx], P["vis"][f, idx]
        xy[~present] = 0; world[~present] = 0; vis[~present] = 0
        if v in SOUTHPAW_VIDEOS:
            xy, world, vis = xy[:, MIRROR_IDX], world[:, MIRROR_IDX], vis[:, MIRROR_IDX]
            xy[..., 0] = np.where(present[:, None], 1 - xy[..., 0], 0)
            world[..., 0] = -world[..., 0]
        world = fill_low_vis(world, vis, present)
        label, n_ann = labels_for(v, F)
        np.savez_compressed(OUT / f"V{v}.npz", world=world.astype(np.float16), xy=xy.astype(np.float16),
                            vis=vis.astype(np.float16), present=present, label=label,
                            fps=P["fps"], W=P["W"], H=P["H"])
        lab_frac = (label >= 0).mean()
        print(f"V{v:<2} ({VIDEO_OF[v]:7s}) frames {F:6d} | person present {present.mean():.1%} | "
              f"{n_ann} punches cover {lab_frac:.1%} of frames | multi-person frames {(P['npose'] > 1).mean():.1%}")


if __name__ == "__main__":
    main()
