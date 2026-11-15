"""Which downloaded video does each dataset video's 2D skeleton come from?

For every (dataset video d, downloaded video file v) pair: take 40 random annotated clips of d,
find for each clip the start frame in video v where MediaPipe's 2D best matches the clip
(sliding over the WHOLE video), then count how many clips agree on one straight line
video_frame = slope * dataset_frame + offset with slope in [0.95, 1.05] (RANSAC, +-4 frames).
A true source gives most clips on one line (slope 1, offset 0 here); a wrong video gives a
handful of chance hits.

    python video3d/crossmatch.py      -> video3d/crossmatch.json, printed matrix
"""
import json

import numpy as np

from common import ALIGN_J, HERE, annotations, clip_frames, load_pose

N_CLIPS = 40


def best_starts(E, fl, clips, sel):
    F = len(E); out = []
    for i in sel:
        ds = clips[i, :len(fl[i])][:, ALIGN_J]; n = len(ds)
        D = np.sqrt(((E[None] - ds[:, None, None]) ** 2).sum(-1)).mean(-1)
        D = np.nanmin(np.where(np.isnan(D), 9, D), axis=2)
        S = np.zeros(F - n)
        for k in range(n):
            S += D[k, k:F - n + k]
        out.append((fl[i][0], int(np.argmin(S))))
    return np.array(out, float)


def ransac(p, rng):
    best = (0, np.nan, np.nan)
    for _ in range(3000):
        a, b = p[rng.choice(len(p), 2, replace=False)]
        if a[0] == b[0]:
            continue
        s = (b[1] - a[1]) / (b[0] - a[0])
        if not 0.95 <= s <= 1.05:
            continue
        d = a[1] - s * a[0]
        n = int((np.abs(p[:, 1] - (s * p[:, 0] + d)) < 4).sum())
        if n > best[0]:
            best = (n, s, d)
    return best


def main():
    rng = np.random.default_rng(0)
    poses = {v: load_pose(v)["xy"][:, :, ALIGN_J] for v in range(1, 11)}
    res = {}
    for d in range(1, 11):
        fl, clips = clip_frames(d, annotations(d))
        idx = [i for i in range(len(fl)) if len(fl[i]) >= 6]
        sel = rng.choice(idx, N_CLIPS, replace=False)
        row = {}
        for v in range(1, 11):
            n, s, o = ransac(best_starts(poses[v], fl, clips, sel), rng)
            row[f"V{v}"] = dict(inliers=n, slope=None if np.isnan(s) else round(float(s), 4),
                                offset=None if np.isnan(o) else round(float(o), 1))
        res[f"V{d}"] = row
        print(f"dataset V{d}: " + "  ".join(f"V{v}:{row[f'V{v}']['inliers']:2d}" for v in range(1, 11)), flush=True)
    (HERE / "crossmatch.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
