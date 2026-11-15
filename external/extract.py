"""Skeletons for external test clips: MediaPipe Pose Landmarker (heavy), up to K people per frame.

Same model, settings and output format as video3d/extract_pose.py (our own videos), so external
skeletons are directly comparable:  external/pose/{id}.npz with frame, pts_ms, npose (F,),
img (F,K,33,5: x, y, z, visibility, presence), world (F,K,33,3), fps, width, height, model_sha256.
Person ORDER is NOT stable across frames (measured: swaps 3-18%); identities come from track.py.
    /opt/anaconda3/envs/mediapipe/bin/python external/extract.py [ids...] [--num-poses 3]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "video3d" / "raw" / "models" / "pose_landmarker_heavy.task"
sys.path.insert(0, str(ROOT))


def extract(cid, K, seg):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    src = HERE / "raw" / f"{cid}.mp4"; out = HERE / "pose" / f"{cid}.npz"
    cap = cv2.VideoCapture(str(src)); fps = float(cap.get(cv2.CAP_PROP_FPS))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f0, f1 = seg                                          # original-video frame numbers of the segment
    step = max(1, int(round(fps / 30.0)))                 # > 30 fps (e.g. 60): keep every 2nd frame, like our 24-30 fps training data
    n_hdr = f1 - f0
    for _ in range(f0):                                   # sequential skip: exact frame indexing
        if not cap.grab():
            break
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), running_mode=vision.RunningMode.VIDEO, num_poses=K,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
        output_segmentation_masks=False)
    lm = vision.PoseLandmarker.create_from_options(opts)
    try:
        from progress import progress
    except ImportError:
        progress = None
    if progress:
        progress.start(f"pose {cid}", total=max(n_hdr, 1))
    frames, pts, npose, imgs, worlds = [], [], [], [], []
    k, t0 = f0, time.time()
    while k < f1:
        ok, bgr = cap.read()
        if not ok:
            break
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                                  int(round(k * 1000.0 / fps)))
        img = np.full((K, 33, 5), np.nan, np.float32); wld = np.full((K, 33, 3), np.nan, np.float32)
        for p, (pl, pw) in enumerate(zip(res.pose_landmarks, res.pose_world_landmarks)):
            img[p] = [[q.x, q.y, q.z, q.visibility, q.presence] for q in pl]
            wld[p] = [[q.x, q.y, q.z] for q in pw]
        frames.append(k); npose.append(len(res.pose_landmarks)); imgs.append(img); worlds.append(wld)
        k += 1
        for _ in range(step - 1):                         # skip frames when subsampling
            cap.grab(); k += 1
        if progress and (k - f0) % 100 == 0:
            progress.tick(100, note=f"frame {k - f0}/{n_hdr}")
    cap.release(); lm.close()
    if progress:
        progress.finish()
    out.parent.mkdir(exist_ok=True)
    tmp = str(out) + ".tmp.npz"
    np.savez_compressed(tmp, frame=np.array(frames, np.int32), pts_ms=np.array(pts), npose=np.array(npose, np.int8),
                        img=np.array(imgs).reshape(-1, K, 33, 5), world=np.array(worlds).reshape(-1, K, 33, 3),
                        fps=fps / step, fps_source=fps, step=step, width=W, height=H, model_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                        mediapipe_version=mp.__version__)
    os.replace(tmp, out)
    n = np.array(npose)
    print(f"[{cid}] {k - f0} frames (video frames {f0}-{k - 1}) in {time.time() - t0:.0f}s | people per frame: 0 {np.mean(n == 0):.1%}, "
          f"1 {np.mean(n == 1):.1%}, 2 {np.mean(n == 2):.1%}, 3+ {np.mean(n >= 3):.1%}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--num-poses", type=int, default=3)
    args = ap.parse_args()
    man = json.loads((HERE / "manifest.json").read_text())
    ids = args.ids or [k for k, v in man.items() if v.get("status") in ("ok", "check")]
    for cid in ids:
        if man.get(cid, {}).get("status") not in ("ok", "check") or not (HERE / "raw" / f"{cid}.mp4").exists():
            print(f"[{cid}] skipped: no downloaded video"); continue
        extract(cid, args.num_poses, man[cid]["segment_frames"])


if __name__ == "__main__":
    main()
