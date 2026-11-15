"""LAPTOP copy (CPU) of hpc_rtm/extract.py for the bag-route clips; run with .venv_rtm/bin/python.
HPC GPU job: YOLOX person detector + RTMPose on a list of videos (standalone: needs only rtmlib,
onnxruntime-gpu, opencv, numpy). Same output format as olympic/extract_rtm.py, K = 20 people per frame.

Writes each result to node-local disk first, then copies it to OUT (shared storage) in one go.
    python extract.py LIST.txt VIDEOS_DIR OUT_DIR [--mode balanced] [--limit-frames N]
"""
import argparse, os, shutil, sys, tempfile, time
import numpy as np

K = 20

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("list"); ap.add_argument("videos"); ap.add_argument("out")
    ap.add_argument("--mode", default="balanced"); ap.add_argument("--limit-frames", type=int)
    ap.add_argument("--step", type=int, default=0, help="keep every STEP-th frame; 0 = automatic (about 30 fps)")
    a = ap.parse_args()
    import onnxruntime as ort
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()                                   # CUDA / cuDNN from the pip nvidia-* wheels
    import cv2
    from rtmlib import Body
    body = Body(mode=a.mode, to_openpose=False, backend="onnxruntime", device="cpu")
    for m in (body.det_model, body.pose_model):
        print("providers:", m.session.get_providers(), flush=True)
    ids = [l.strip() for l in open(a.list) if l.strip()]
    tmpdir = tempfile.mkdtemp(prefix="rtm_", dir=os.environ.get("TMPDIR", "/tmp"))
    for n, cid in enumerate(ids):
        dst = os.path.join(a.out, cid + ".npz")
        if os.path.exists(dst):
            print(f"[{cid}] already done", flush=True); continue
        cap = cv2.VideoCapture(os.path.join(a.videos, cid + ".mp4")); fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); step = a.step or max(1, int(round(fps / 30.0)))
        frames, ns, kps, scs, bxs = [], [], [], [], []
        k, t0 = 0, time.time()
        while True:
            ok, bgr = cap.read()
            if not ok or (a.limit_frames and len(frames) >= a.limit_frames):
                break
            boxes = np.asarray(body.det_model(bgr)).reshape(-1, 4)
            kp = np.full((K, 17, 2), np.nan, np.float32); sc = np.zeros((K, 17), np.float32); bx = np.full((K, 4), np.nan, np.float32)
            if len(boxes):
                boxes = boxes[np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:K]]
                p, s = body.pose_model(bgr, bboxes=boxes.tolist())
                m = len(boxes); kp[:m], sc[:m], bx[:m] = p, s, boxes
            frames.append(k); ns.append(min(len(boxes), K)); kps.append(kp); scs.append(sc); bxs.append(bx)
            k += 1
            for _ in range(step - 1):
                cap.grab(); k += 1
            if len(frames) % 1000 == 0:
                print(f"[{cid}] {k}/{total} video frames, {len(frames) / (time.time() - t0):.1f} fps  ({n + 1}/{len(ids)})", flush=True)
        tmp = os.path.join(tmpdir, cid + ".npz")
        np.savez_compressed(tmp, frame=np.array(frames, np.int32), n=np.array(ns, np.int8), kp=np.array(kps), score=np.array(scs),
                            box=np.array(bxs), fps=fps / step, fps_source=fps, step=step,
                            width=int(cap.get(3)), height=int(cap.get(4)), mode=a.mode)
        shutil.copy(tmp, dst + ".part"); os.replace(dst + ".part", dst); os.remove(tmp)
        dt = time.time() - t0
        print(f"[{cid}] DONE {len(frames)} frames in {dt:.0f}s ({len(frames) / max(dt, 1e-9):.1f} fps) | people per frame: median {np.median(ns):.0f}, max {max(ns)}", flush=True)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
