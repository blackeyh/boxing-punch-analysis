"""Bag route, feasibility test 1: open-vocabulary detection (OWLv2) of bag / dummy on sampled frames.

    python bagroute/detect_target.py CLIP_ID [--every 2.0] [--limit N] [--device mps]
Writes runs/bagroute/{id}_det.json : per sampled frame, all boxes with score >= MIN_SCORE per prompt.
Test-only footage; nothing is trained here.
"""
import argparse, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mem_guard  # noqa: F401  (must come before torch)
import cv2, numpy as np, torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

MODEL = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["a punching bag", "a heavy bag", "a free standing punching bag", "a boxing dummy",
           "a training dummy torso", "a mannequin", "a person"]
MIN_SCORE, TOPK = 0.05, 15

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip"); ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--limit", type=int); ap.add_argument("--device", default="mps")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "runs", "bagroute", a.clip + "_det.json")
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL, use_safetensors=True).eval().to(a.device)
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", a.clip + ".mp4"))
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(3)), int(cap.get(4)); S = max(W, H)      # OWLv2 pads to a square
    step = int(round(a.every * fps)); idx = list(range(0, total, step))[: a.limit]
    text = proc(text=[PROMPTS], return_tensors="pt").to(a.device)
    res, times = [], []
    for n, f in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, bgr = cap.read()
        if not ok:
            continue
        t0 = time.time()
        pix = proc(images=Image.fromarray(bgr[:, :, ::-1].copy()), return_tensors="pt").to(a.device)
        with torch.no_grad():
            o = model(pixel_values=pix["pixel_values"], input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        sc = torch.sigmoid(o.logits[0]).float().cpu().numpy()          # (queries, prompts)
        cx, cy, w, h = (o.pred_boxes[0].float().cpu().numpy() * S).T   # normalised to the padded square
        bx = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        times.append(time.time() - t0)
        dets = []
        for p in range(len(PROMPTS)):
            order = np.argsort(-sc[:, p])[:TOPK]
            for q in order:
                if sc[q, p] >= MIN_SCORE:
                    dets.append({"p": p, "s": round(float(sc[q, p]), 4), "b": [round(float(v), 1) for v in bx[q]]})
        res.append({"frame": f, "t": f / fps, "dets": dets})
        if n % 25 == 0:
            print(f"[{a.clip}] {n + 1}/{len(idx)}  {np.mean(times[1:] or times):.2f} s/frame", flush=True)
    json.dump({"clip": a.clip, "model": MODEL, "prompts": PROMPTS, "fps": fps, "width": W, "height": H,
               "every_s": a.every, "sec_per_frame": float(np.mean(times[1:] or times)), "device": a.device, "frames": res},
              open(out, "w"))
    print(f"[{a.clip}] DONE {len(res)} frames, {np.mean(times[1:] or times):.2f} s/frame -> {out}", flush=True)

if __name__ == "__main__":
    main()
