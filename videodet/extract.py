"""Pixel-only features + zero-shot scores for every 16-frame window (detect/WINDOWS.md).

    python videodet/extract.py --videos 1 2 3 [--models base,xclip] [--limit 2000]

Per dataset video v (file VIDEO_OF[v]), two sequential decoding passes (OpenCV, frame k = k-th
decoded frame, as in video3d):
  1. PERSON BOXES: torchvision Faster R-CNN (COCO) every DET_STRIDE frames on a 640x360 copy;
     keep the largest confident person, median-smooth over time, interpolate to every frame.
  2. WINDOWS: for each window (start, start+15) one fixed square crop around the union of the
     main person's boxes over the 16 frames (side = 1.25 x the larger box side), resized to
     224x224, the same crop for all 16 frames (so it doesn't jitter). Then
       - VideoMAE-B (Kinetics-400, 16 frames): 400 logits + 768-d pooled feature
       - VideoMAE-L (optional, `large`): 400 logits + 1024-d feature
       - X-CLIP B/32 (8 frames = every 2nd): logits vs PROMPTS + 512-d video embedding
No skeleton / pose data is read: the only inputs are the video pixels.

Output $RUN_DIR/feats/V{v}.npz (start, box, crop, + per-model arrays) and boxes/V{v}.npz.
"""
import argparse
import os
import sys
import threading
import queue
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from progress import progress  # noqa: E402
import windows as Wd  # noqa: E402

VIDEOS = HERE / "raw" / "videos"
OUT = Path(os.environ.get("RUN_DIR", HERE / "cache"))
DET_STRIDE = 8
SIZE = 224
MARGIN = 1.25

# X-CLIP prompts: first block = punch, second = not a punch. Kept fixed so scores are reproducible.
POS_PROMPTS = [
    "a person throwing a punch",
    "a boxer throwing a jab",
    "a boxer throwing a straight cross punch",
    "a boxer throwing a hook punch",
    "a boxer throwing an uppercut",
    "a person punching a heavy bag",
    "a person shadow boxing and throwing punches",
]
NEG_PROMPTS = [
    "a boxer standing still in guard position",
    "a boxer moving their feet",
    "a boxer bouncing on their feet with hands up",
    "a boxer bobbing and weaving",
    "a person standing with arms down resting",
    "a person talking to the camera",
    "a person stretching",
    "a person walking",
    "a person doing squats",
]
PROMPTS = POS_PROMPTS + NEG_PROMPTS

MODEL_IDS = {  # pinned commits (videodet/SOURCES.md)
    "base": ("MCG-NJU/videomae-base-finetuned-kinetics", "488eb9a0565f257b32866000305c8178965eb9f6"),
    "large": ("MCG-NJU/videomae-large-finetuned-kinetics", "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"),
    "xclip": ("microsoft/xclip-base-patch32", "a2e27a78a2b5d802e894b8a1ef14f3a8ce490963"),
}
IMNET = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
CLIPN = (np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711]))


# ---------------------------------------------------------------- pass 1: person boxes
def person_boxes(path, n, dev, detector):
    cap = cv2.VideoCapture(str(path))
    det_frames, raw, batch, bidx = [], [], [], []

    def flush():
        with torch.no_grad():
            ims = [torch.from_numpy(b).to(dev).permute(2, 0, 1).float() / 255 for b in batch]
            out = detector(ims)
        for f, o in zip(bidx, out):
            keep = (o["labels"] == 1) & (o["scores"] > 0.5)
            raw.append(np.concatenate([o["boxes"][keep].cpu().numpy() * 2,      # back to 1280x720
                                       o["scores"][keep].cpu().numpy()[:, None]], 1))
            det_frames.append(f)
        batch.clear(); bidx.clear()

    for f in range(n):
        if f % DET_STRIDE:
            if not cap.grab():
                break
            continue
        ok, im = cap.read()
        if not ok:
            break
        batch.append(cv2.cvtColor(cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        bidx.append(f)
        if len(batch) == 32:
            flush()
    if batch:
        flush()
    cap.release()
    det_frames = np.array(det_frames)
    main = np.full((len(det_frames), 4), np.nan)
    nper = np.array([len(r) for r in raw])
    for i, r in enumerate(raw):
        if len(r):
            area = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
            main[i] = r[np.argmax(area), :4]                 # main boxer = largest person
    # fill gaps by interpolation, then median-smooth (+-2 detections = +-16 frames)
    ok = ~np.isnan(main[:, 0])
    if ok.sum() == 0:
        main[:] = [280, 0, 1000, 720]                         # no person at all: central square
    else:
        for c in range(4):
            main[:, c] = np.interp(det_frames, det_frames[ok], main[ok, c])
    pad = np.pad(main, ((2, 2), (0, 0)), mode="edge")
    smooth = np.median(np.stack([pad[k:k + len(main)] for k in range(5)]), 0)
    per_frame = np.stack([np.interp(np.arange(n), det_frames, smooth[:, c]) for c in range(4)], 1)
    raw_flat = np.concatenate([np.c_[np.full(len(r), f), r] for f, r in zip(det_frames, raw) if len(r)] or [np.zeros((0, 6))])
    return per_frame.astype(np.float32), dict(det_frames=det_frames, main=main.astype(np.float32),
                                               smooth=smooth.astype(np.float32), n_person=nper, raw=raw_flat.astype(np.float32))


def window_crop(per_frame, s):
    b = per_frame[s:s + Wd.W]
    x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = max(MARGIN * max(x1 - x0, y1 - y0), 160)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.array([cx - side / 2, cy - side / 2, side], np.float32)


def cut(im, crop):
    """Square crop (x, y, side) of a 720p BGR frame -> 224x224 RGB, zero-padded outside the frame."""
    x, y, side = crop
    x0, y0, s = int(round(x)), int(round(y)), int(round(side))
    H, Wi = im.shape[:2]
    canvas = np.zeros((s, s, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(Wi, x0 + s), min(H, y0 + s)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = im[sy0:sy1, sx0:sx1]
    return cv2.cvtColor(cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- pass 2: windows
def window_batches(path, starts, crops, bs):
    """Producer thread: decode sequentially, keep the last 16 frames, emit (idx, clips uint8)."""
    q = queue.Queue(maxsize=6)

    def run():
        cap = cv2.VideoCapture(str(path))
        buf, wi, idx, clips = {}, 0, [], []
        last = starts[-1] + Wd.W - 1
        for f in range(last + 1):
            ok, im = cap.read()
            if not ok:
                raise RuntimeError(f"{path}: decode failed at frame {f}")
            buf[f] = im
            buf.pop(f - Wd.W, None)
            while wi < len(starts) and starts[wi] + Wd.W - 1 == f:
                s = starts[wi]
                clips.append(np.stack([cut(buf[k], crops[wi]) for k in range(s, s + Wd.W)]))
                idx.append(wi); wi += 1
                if len(clips) == bs:
                    q.put((np.array(idx), np.stack(clips))); idx, clips = [], []
        if clips:
            q.put((np.array(idx), np.stack(clips)))
        cap.release()
        q.put(None)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while (item := q.get()) is not None:
        yield item
    t.join()


def norm(clips, dev, stats):
    x = torch.from_numpy(clips).to(dev).float().div_(255)          # B,T,H,W,C
    m = torch.tensor(stats[0], device=dev, dtype=torch.float32)
    s = torch.tensor(stats[1], device=dev, dtype=torch.float32)
    return ((x - m) / s).permute(0, 1, 4, 2, 3).contiguous().half()  # B,T,C,H,W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--models", default="base,xclip")
    ap.add_argument("--limit", type=int, default=0, help="only the first N windows per video (smoke test)")
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    models = args.models.split(",")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # weights pre-fetched by fetch_weights.py
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (OUT / "feats").mkdir(parents=True, exist_ok=True)
    (OUT / "boxes").mkdir(parents=True, exist_ok=True)

    import torchvision
    from transformers import VideoMAEForVideoClassification, XCLIPModel, AutoTokenizer
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights="DEFAULT", min_size=360, max_size=640).eval().to(dev)
    nets = {}
    for m in models:
        rid, rev = MODEL_IDS[m]
        if m == "xclip":
            nets[m] = XCLIPModel.from_pretrained(rid, revision=rev, use_safetensors=True).half().eval().to(dev)
            tok = AutoTokenizer.from_pretrained(rid, revision=rev)
            text = tok(PROMPTS, padding=True, return_tensors="pt").to(dev)
        else:
            nets[m] = VideoMAEForVideoClassification.from_pretrained(
                rid, revision=rev, use_safetensors=True, attn_implementation="sdpa").half().eval().to(dev)

    work = {v: len(Wd.starts(v)) if not args.limit else min(args.limit, len(Wd.starts(v))) for v in args.videos}
    progress.start(f"videodet extract {','.join(models)}", total=sum(work.values()) * 1.1)
    for v in args.videos:
        path = VIDEOS / f"{Wd.VIDEO_OF[v]}.mp4"
        n = Wd.n_frames(v)
        starts = Wd.starts(v)[:work[v]]
        t0 = time.time()
        per_frame, binfo = person_boxes(path, int(starts[-1]) + Wd.W if args.limit else n, dev, detector)
        np.savez_compressed(OUT / "boxes" / f"V{v}.npz", per_frame=per_frame, **binfo)
        progress.tick(work[v] * 0.1, note=f"V{v} boxes")
        t1 = time.time()
        print(f"V{v}: person boxes {time.time() - t0:.0f}s, detection frames {len(binfo['det_frames'])}, "
              f"no person {np.mean(binfo['n_person'] == 0):.1%}, >1 person {np.mean(binfo['n_person'] > 1):.1%}", flush=True)
        crops = np.stack([window_crop(per_frame, s) for s in starts])
        res = {k: [] for k in ["base_logits", "base_feat", "large_logits", "large_feat", "xclip_logits", "xclip_emb"]}
        done = 0
        for idx, clips in window_batches(path, starts, crops, args.bs):
            with torch.no_grad():
                if "base" in nets or "large" in nets:
                    x = norm(clips, dev, IMNET)
                    for m in ("base", "large"):
                        if m in nets:
                            net = nets[m]
                            seq = net.videomae(x)[0]
                            feat = net.fc_norm(seq.mean(1)) if net.fc_norm is not None else seq[:, 0]
                            res[f"{m}_logits"].append(net.classifier(feat).float().cpu().numpy())
                            res[f"{m}_feat"].append(feat.float().cpu().numpy().astype(np.float16))
                if "xclip" in nets:
                    xc = norm(clips[:, ::2], dev, CLIPN)
                    o = nets["xclip"](input_ids=text["input_ids"], attention_mask=text["attention_mask"], pixel_values=xc)
                    res["xclip_logits"].append(o.logits_per_video.float().cpu().numpy())
                    res["xclip_emb"].append(o.video_embeds.float().cpu().numpy().astype(np.float16))
            done += len(idx)
            progress.tick(len(idx), note=f"V{v} {done}/{len(starts)}")
        out = {k: np.concatenate(x) for k, x in res.items() if x}
        np.savez_compressed(OUT / "feats" / f"V{v}.npz", start=starts, crop=crops, prompts=np.array(PROMPTS),
                            n_pos_prompts=len(POS_PROMPTS), **out)
        dt = time.time() - t1
        print(f"V{v}: {len(starts)} windows in {dt:.0f}s ({len(starts) / dt:.0f}/s); "
              + ", ".join(f"{k} {x.shape}" for k, x in out.items()), flush=True)
    progress.finish()


if __name__ == "__main__":
    main()
