"""Build the gap-window review set: which windows the user reviews, plus video crops and skeletons.

Selection per video (non-overlapping windows, person present in > 50% of frames):
  * 12 'B-high'   : gap windows with the highest skeleton-PU score (likely missed punches)
  *  6 'B-border' : gap windows with score closest to 0.5
  *  8 'random'   : random gap windows (estimates how clean "no punch" really is)
  *  2 'control'  : LABELLED punch windows, shown blind (checks the review itself)
Groups and detector scores are stored for analysis but NOT shown while reviewing.
When the video-pixel detector (A) arrives, rerun with --with-a to add its scores/groups.

Outputs (detect/review/):
  review_data.json       candidates: video, start, group, scores, crop box, 16-frame skeleton (crop coords)
  sprites/V{v}.jpg       one row per candidate, 16 frames of 160x160 crops (git-ignored, rebuildable)
    /opt/anaconda3/bin/python detect/review/build_review.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from baselines import MIRROR_IDX, SOUTHPAW_VIDEOS  # noqa: E402
from detect.pu_detector import windows  # noqa: E402
from progress import progress  # noqa: E402
from punch_data import CLASSES  # noqa: E402
from video3d.common import VIDEO_OF  # noqa: E402

HERE = Path(__file__).resolve().parent
VIDEOS_DIR = Path("~/Downloads/Dataset_video3d/video3d/raw/videos")
W, TILE, SEED = 16, 160, 0
CTX = 8          # context frames shown before AND after the judged 16-frame window
PLAN = [("B-high", 12), ("B-border", 6), ("random", 8), ("control", 2)]


def pick(cands, n, taken, key=None, rng=None):
    order = list(cands)
    if key is not None:
        order = sorted(order, key=key)
    elif rng is not None:
        rng.shuffle(order)
    out = []
    for s in order:
        if all(abs(s - t) >= W for t in taken):
            out.append(s); taken.append(s)
            if len(out) == n:
                break
    return out


def main():
    rng = np.random.default_rng(SEED)
    (HERE / "sprites").mkdir(exist_ok=True)
    items = []
    progress.start("review sprites", total=10)
    for v in range(1, 11):
        fr = np.load(ROOT / "detect" / "frames" / f"V{v}.npz")
        sb = np.load(ROOT / "detect" / "scores" / "pu_skeleton" / f"V{v}.npz")
        starts, score, cov, gap, wp = windows(sb["frame_p"].astype(np.float32), fr["label"], fr["present"])
        sc = dict(zip(starts.tolist(), score.tolist()))
        ok_gap = [int(s) for s, g, p in zip(starts, gap, wp) if g and p > 0.5]
        ok_lab = [int(s) for s, c, p in zip(starts, cov, wp) if c >= 0.5 and p > 0.5]
        taken, chosen = [], []
        chosen += [("B-high", s) for s in pick(ok_gap, 12, taken, key=lambda s: -sc[s])]
        chosen += [("B-border", s) for s in pick(ok_gap, 6, taken, key=lambda s: abs(sc[s] - 0.5))]
        chosen += [("random", s) for s in pick(ok_gap, 8, taken, rng=rng)]
        chosen += [("control", s) for s in pick(ok_lab, 2, taken, rng=rng)]

        # skeleton in ORIGINAL video orientation (frames file has V4 mirrored to orthodox)
        xy = fr["xy"].astype(np.float32).copy(); pres = fr["present"]
        if v in SOUTHPAW_VIDEOS:
            xy = xy[:, MIRROR_IDX]; xy[..., 0] = np.where(pres[:, None], 1 - xy[..., 0], 0)
        Wpx, Hpx = int(fr["W"]), int(fr["H"])

        boxes = {}
        for g, s in chosen:
            pts = xy[s:s + W][pres[s:s + W]].reshape(-1, 2)
            pts = pts[(pts > 0).all(1)] * [Wpx, Hpx]
            cx, cy = (pts.min(0) + pts.max(0)) / 2 if len(pts) else (Wpx / 2, Hpx / 2)
            side = max((pts.max(0) - pts.min(0)).max() * 1.35 if len(pts) else 0, 0.45 * Hpx)
            side = min(side, Hpx)
            x0 = int(np.clip(cx - side / 2, 0, Wpx - side)); y0 = int(np.clip(cy - side / 2, 0, Hpx - side))
            boxes[s] = (x0, y0, int(side))

        NT = W + 2 * CTX
        need = {}
        for row, (g, s) in enumerate(chosen):
            for k in range(NT):
                need.setdefault(s - CTX + k, []).append((row, k))
        sheet = np.zeros((len(chosen) * TILE, NT * TILE, 3), np.uint8)
        cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{VIDEO_OF[v]}.mp4"))
        last = max(need)
        f = 0
        while f <= last:
            if f in need:
                ok, img = cap.read()
                if ok:
                    for row, k in need[f]:
                        s = chosen[row][1]; x0, y0, side = boxes[s]
                        crop = img[y0:y0 + side, x0:x0 + side]
                        sheet[row * TILE:(row + 1) * TILE, k * TILE:(k + 1) * TILE] = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_AREA)
            else:
                cap.grab()
            f += 1
        cap.release()
        cv2.imwrite(str(HERE / "sprites" / f"V{v}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 72])

        for row, (g, s) in enumerate(chosen):
            x0, y0, side = boxes[s]
            idx = np.clip(np.arange(s - CTX, s + W + CTX), 0, len(xy) - 1)
            sk = xy[idx].copy()
            sk[..., 0] = (sk[..., 0] * Wpx - x0) / side; sk[..., 1] = (sk[..., 1] * Hpx - y0) / side
            sk[~pres[idx]] = -1
            lab = fr["label"][s:s + W]
            true = CLASSES[int(np.bincount(lab[lab >= 0]).argmax())] if (lab >= 0).any() else None
            items.append(dict(id=f"V{v}_{s}", video=v, start=s, row=row, group=g, score_B=round(sc[s], 4),
                              label_in_data=true, crop=[x0, y0, side],
                              skel=np.round(sk, 3).tolist(), fps=float(fr["fps"])))
        progress.tick(1, note=f"V{v}")
        print(f"V{v}: {len(chosen)} windows ({', '.join(f'{g} {sum(1 for x in chosen if x[0] == g)}' for g, _ in PLAN)})", flush=True)
    progress.finish()
    order = rng.permutation(len(items)).tolist()           # review order: shuffled across videos and groups
    (HERE / "review_data.json").write_text(json.dumps(dict(items=items, order=order, tile=TILE, frames=W + 2 * CTX, ctx=CTX, core=W,
                                                           classes=CLASSES), separators=(",", ":")))
    print(f"{len(items)} review items -> {HERE / 'review_data.json'}")


if __name__ == "__main__":
    main()
