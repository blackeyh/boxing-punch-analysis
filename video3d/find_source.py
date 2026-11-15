"""Search candidate YouTube videos for the true source of a dataset video whose metadata link
does not match (used for V1). For each candidate: download a low-resolution copy, run
MediaPipe on N sampled annotated frame indices (offset 0, since every aligned video has
offset 0), and score the 2D agreement with the dataset skeleton (plain and mirrored).

    python video3d/find_source.py --video 1 --ids video3d/source_search/candidates_V1.txt --workers 8

Writes video3d/source_search/V{v}_scores.jsonl (one line per candidate). Videos are deleted
after scoring.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ALIGN_J, HERE, MP2COCO, RAW, SWAP_LR, dataset_frame_map  # noqa: E402

OUT = Path(os.environ.get("RUN_DIR", HERE / "source_search"))


def sample_frames(v, n, seed=0):
    m = dataset_frame_map(v)
    fr = np.array(sorted(m))
    pick = np.sort(np.random.default_rng(seed).choice(fr, n, replace=False))
    return pick, np.stack([m[f] for f in pick])


def score_one(job):
    vid_id, frames, ds = job
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    res = dict(id=vid_id)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        out = Path(td) / "v.mp4"
        r = subprocess.run(["yt-dlp", "--no-warnings", "--no-progress", "-q", "-f",
                            "bv[height<=360][vcodec^=avc1]/bv[height<=360]/worst", "-o", str(out),
                            f"https://www.youtube.com/watch?v={vid_id}"], capture_output=True, text=True)
        if r.returncode or not out.exists():
            res["error"] = r.stderr.strip()[-300:]
            return res
        cap = cv2.VideoCapture(str(out))
        res.update(fps=cap.get(cv2.CAP_PROP_FPS), nframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                   W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        lm = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(RAW / "models" / "pose_landmarker_heavy.task")),
            running_mode=vision.RunningMode.IMAGE, num_poses=3))
        want = {int(f): i for i, f in enumerate(frames)}
        est = np.full((len(frames), 3, 17, 2), np.nan)
        k, last = 0, max(want)
        while k <= last:
            if k in want:
                ok, bgr = cap.read()
                if not ok:
                    break
                det = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                for p, pl in enumerate(det.pose_landmarks):
                    est[want[k], p] = [[pl[j].x, pl[j].y] for j in MP2COCO]
            elif not cap.grab():
                break
            k += 1
        cap.release(); lm.close()
    W, H = 1280, 720   # score in the reference 1280x720 pixel units
    for name, e in (("plain", est), ("mirror", None)):
        if e is None:
            e = est[:, :, SWAP_LR].copy(); e[..., 0] = 1 - e[..., 0]
        d = e[:, :, ALIGN_J] - ds[:, None, ALIGN_J]
        px = np.sqrt((d[..., 0] * W) ** 2 + (d[..., 1] * H) ** 2).mean(-1)
        best = np.nanmin(np.where(np.isnan(px), np.inf, px), axis=1)
        best[~np.isfinite(best)] = np.nan
        res[f"median_px_{name}"] = float(np.nanmedian(best)) if np.isfinite(best).any() else None
        res[f"frac_below_25px_{name}"] = float(np.mean(best < 25))
    res["detected"] = float(np.isfinite(est[:, 0, 0, 0]).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--control", help="VIDEO:ID of a known-correct pair, scored first (positive control)")
    a = ap.parse_args()
    from progress import progress
    OUT.mkdir(exist_ok=True)
    out = OUT / f"V{a.video}_scores.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    ids = [l.split("|")[0].strip() for l in Path(a.ids).read_text().splitlines() if l.strip()]
    ids = [i for i in ids if i not in done]
    if a.control:
        cv, cid = a.control.split(":")
        fr_c, ds_c = sample_frames(int(cv), a.n)
        r = score_one((cid, fr_c, ds_c)); r["control_for_video"] = int(cv)
        (OUT / "control.json").write_text(json.dumps(r)); print("CONTROL", json.dumps(r), flush=True)
    frames, ds = sample_frames(a.video, a.n)
    progress.start(f"find source V{a.video}", total=len(ids))
    with Pool(a.workers) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(score_one, [(i, frames, ds) for i in ids]):
            f.write(json.dumps(r) + "\n"); f.flush()
            progress.tick(1, note=r["id"])
            print(json.dumps(r), flush=True)
    progress.finish()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent))
    main()
