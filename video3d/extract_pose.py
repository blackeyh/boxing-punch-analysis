"""Run MediaPipe Pose Landmarker (heavy) on EVERY frame of the source videos.

    python video3d/extract_pose.py --videos 1 2 3 --workers 24

Each video is split into chunks of consecutive frames; each chunk is processed by one
worker process in VIDEO mode (temporal tracking inside the chunk). Frames are decoded
sequentially with OpenCV, so frame index k = the k-th decoded frame (0-based). Up to
--num-poses people are kept per frame; the person is chosen later, in align.py.

Output (git-ignored): video3d/raw/pose/{v}.npz with
    frame    (F,)           0-based decoded frame index (contiguous 0..F-1)
    pts_ms   (F,)           decoder timestamp of each frame (ms), to check constant fps
    npose    (F,)           number of people detected
    img      (F, K, 33, 5)  image landmarks: x/width, y/height, z, visibility, presence (NaN if none)
    world    (F, K, 33, 3)  world landmarks in metres, hip-centred (NaN if none)
    fps, width, height, model_sha256, mediapipe_version
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
RAW = HERE / "raw"
MODEL = RAW / "models" / "pose_landmarker_heavy.task"


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS), width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                nframes_header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def run_chunk(job):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    v, start, stop, K, fps, out = job
    if Path(out).exists():
        return v, start, int(np.load(out)["frame"].size), 0.0
    t0 = time.time()
    cv2.setNumThreads(1)
    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO, num_poses=K,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5, output_segmentation_masks=False)
    lm = vision.PoseLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(RAW / "videos" / f"{v}.mp4"))
    for _ in range(start):                      # sequential skip: exact frame indexing
        if not cap.grab():
            break
    frames, pts, npose, imgs, worlds = [], [], [], [], []
    k = start
    while k < stop:
        ok, bgr = cap.read()
        if not ok:
            break
        pts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                  int(round(k * 1000.0 / fps)))
        img = np.full((K, 33, 5), np.nan, np.float32)
        wld = np.full((K, 33, 3), np.nan, np.float32)
        for p, (pl, pw) in enumerate(zip(res.pose_landmarks, res.pose_world_landmarks)):
            img[p] = [[q.x, q.y, q.z, q.visibility, q.presence] for q in pl]
            wld[p] = [[q.x, q.y, q.z] for q in pw]
        frames.append(k); npose.append(len(res.pose_landmarks)); imgs.append(img); worlds.append(wld)
        k += 1
    cap.release(); lm.close()
    tmp = str(out) + ".tmp.npz"
    np.savez(tmp, frame=np.array(frames, np.int32), pts_ms=np.array(pts), npose=np.array(npose, np.int8),
             img=np.array(imgs).reshape(-1, K, 33, 5), world=np.array(worlds).reshape(-1, K, 33, 3))
    os.replace(tmp, out)
    return v, start, len(frames), time.time() - t0


def merge(v, K, info, chunk_files):
    parts = [np.load(f) for f in chunk_files]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ["frame", "pts_ms", "npose", "img", "world"]}
    assert np.array_equal(d["frame"], np.arange(len(d["frame"]))), f"{v}: frames not contiguous"
    sha = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    import mediapipe as mp
    np.savez_compressed(RAW / "pose" / f"{v}.npz", **d, fps=info["fps"], width=info["width"],
                        height=info["height"], model_sha256=sha, mediapipe_version=mp.__version__)
    for f in chunk_files:
        os.remove(f)
    return len(d["frame"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=[str(i) for i in range(1, 11)],
                    help="video file stems: 1 -> V1.mp4, or e.g. V1_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--num-poses", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=0, help="debug: only the first N frames")
    args = ap.parse_args()
    from progress import progress

    (RAW / "pose").mkdir(parents=True, exist_ok=True)
    jobs, infos, chunks_of = [], {}, {}
    for v in [a if a.startswith("V") else f"V{a}" for a in args.videos]:
        if (RAW / "pose" / f"{v}.npz").exists():
            print(f"{v}: already extracted, skipping", flush=True); continue
        info = infos[v] = video_info(RAW / "videos" / f"{v}.mp4")
        n = info["nframes_header"] if not args.max_frames else args.max_frames
        # The header frame count can be off by a few frames: let the last chunk read to EOF.
        starts = list(range(0, n, args.chunk))
        cdir = RAW / "pose" / f"{v}_chunks"; cdir.mkdir(exist_ok=True)
        chunks_of[v] = []
        for i, s in enumerate(starts):
            stop = s + args.chunk if (i < len(starts) - 1 or args.max_frames) else 10**9
            out = cdir / f"{s:07d}.npz"
            chunks_of[v].append(out)
            jobs.append((v, s, stop, args.num_poses, info["fps"], str(out)))
        print(f"{v}: {info} -> {len(starts)} chunks", flush=True)
    total = sum(min(j[2], infos[j[0]]["nframes_header"]) - j[1] for j in jobs)
    progress.start("video3d pose extraction", total=max(total, 1))
    # longest-first scheduling: big chunks (the last one of each video) don't straggle
    jobs.sort(key=lambda j: -(j[2] - j[1]))
    remaining = {v: len(c) for v, c in chunks_of.items()}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for v, s, n, dt in pool.imap_unordered(run_chunk, jobs):
            progress.tick(n, note=f"{v} chunk {s}")
            print(f"{v} chunk {s}: {n} frames in {dt:.0f}s ({n / max(dt, 1e-9):.1f} fps)  "
                  f"[{time.time() - t0:.0f}s elapsed]", flush=True)
            remaining[v] -= 1
            if remaining[v] == 0:
                nf = merge(v, args.num_poses, infos[v], chunks_of[v])
                print(f"{v}: merged {nf} frames -> raw/pose/{v}.npz", flush=True)
    progress.finish()
    print(json.dumps({v: infos[v] for v in infos}), flush=True)


if __name__ == "__main__":
    main()
