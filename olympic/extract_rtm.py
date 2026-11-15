"""Second pose tool for sparring: person detector (YOLOX) + RTMPose (top-down), via rtmlib / onnxruntime.

Output external/pose_rtm/{id}.npz: frame (F,), n (F,), kp (F,K,17,2 pixels), score (F,K,17), box (F,K,4),
fps. Same frame subsampling as external/extract.py (50 fps -> every 2nd frame). No identities.
    /opt/anaconda3/envs/rtmpose/bin/python olympic/extract_rtm.py ID [--seconds N] [--mode balanced|performance] [--device cpu|mps]
"""
import argparse, json, os, time
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parent.parent
K = 12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id"); ap.add_argument("--seconds", type=float); ap.add_argument("--mode", default="balanced")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from rtmlib import Body
    body = Body(mode=a.mode, to_openpose=False, backend="onnxruntime", device=a.device)
    man = json.loads((ROOT / "external" / "manifest.json").read_text())[a.id]
    cap = cv2.VideoCapture(str(ROOT / "external" / "raw" / f"{a.id}.mp4")); fps = float(cap.get(cv2.CAP_PROP_FPS))
    f0, f1 = man["segment_frames"]
    if a.seconds:
        f1 = min(f1, f0 + int(a.seconds * fps))
    step = max(1, int(round(fps / 30.0)))
    for _ in range(f0):
        cap.grab()
    frames, ns, kps, scs, bxs = [], [], [], [], []
    k, t0 = f0, time.time()
    while k < f1:
        ok, bgr = cap.read()
        if not ok:
            break
        boxes = np.asarray(body.det_model(bgr)).reshape(-1, 4)
        kp = np.full((K, 17, 2), np.nan, np.float32); sc = np.zeros((K, 17), np.float32); bx = np.full((K, 4), np.nan, np.float32)
        if len(boxes):
            order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:K]
            boxes = boxes[order]
            p, s = body.pose_model(bgr, bboxes=boxes.tolist())
            n = len(boxes); kp[:n], sc[:n], bx[:n] = p, s, boxes
        frames.append(k); ns.append(min(len(boxes), K)); kps.append(kp); scs.append(sc); bxs.append(bx)
        k += 1
        for _ in range(step - 1):
            cap.grab(); k += 1
    out = ROOT / "external" / "pose_rtm"; out.mkdir(exist_ok=True)
    tmp = str(out / f"{a.id}{a.tag}.tmp.npz")
    np.savez_compressed(tmp, frame=np.array(frames, np.int32), n=np.array(ns, np.int8), kp=np.array(kps), score=np.array(scs),
                        box=np.array(bxs), fps=fps / step, width=int(cap.get(3)), height=int(cap.get(4)), mode=a.mode)
    os.replace(tmp, out / f"{a.id}{a.tag}.npz")
    dt = time.time() - t0
    print(f"[{a.id}] {len(frames)} frames in {dt:.0f}s ({len(frames)/dt:.1f} fps) | people per frame: median {np.median(ns):.0f}", flush=True)

if __name__ == "__main__":
    main()
