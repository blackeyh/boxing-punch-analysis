"""Second opinion from an open vision-language model (Qwen3-VL-8B-Instruct) on SELECTED windows.

    python videodet/vlm_check.py videodet/vlm_selection.json [--limit 5]

The selection (made locally by select_vlm.py) lists {video, start, crop, group}. For each window
the model sees 8 frames (every 2nd of the 16, the same person-centred crop as the main detector,
336x336) as an ordered image sequence and is asked whether a punch is thrown. Score =
P(Yes) from the next-token logits of "Yes" vs "No" (no sampling). Pixels only.
Output $RUN_DIR/vlm_answers.json.
"""
import json
import os
import sys
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
MODEL = ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
QUESTION = ("These 8 images are consecutive frames (in order, about 0.6 seconds in total) from a boxing "
            "workout video. Does the person throw a punch (a jab, cross, hook or uppercut, at a bag or in "
            "the air) during these frames? Only an actual punch counts: standing in guard, footwork, "
            "bouncing, slipping, blocking or resting do not. Answer with Yes or No.")
SIDE = 336


class Reader:
    """Sequential reader (selection is sorted by start): read forward, keeping the last 64 frames
    so overlapping / repeated windows work. OpenCV seeking on these H.264 files took ~10 s/window."""
    def __init__(self, path):
        self.cap, self.pos, self.buf = cv2.VideoCapture(str(path)), 0, {}

    def frames(self, start, n):
        assert start >= self.pos - 64, "selection must be sorted by start within a video"
        while self.pos < start - 64:
            self.cap.grab(); self.pos += 1
        while self.pos < start + n:
            ok, im = self.cap.read()
            if not ok:
                raise RuntimeError(f"decode failed at {self.pos}")
            self.buf[self.pos] = im
            self.buf.pop(self.pos - 64, None)
            self.pos += 1
        return [self.buf[f] for f in range(start, start + n)]


def crop_frames(reader, start, crop):
    x, y, side = crop
    x0, y0, s = int(round(x)), int(round(y)), int(round(side))
    out = []
    for k, im in enumerate(reader.frames(int(start), Wd.W)):
        if k % 2:
            continue
        H, Wi = im.shape[:2]
        canvas = np.zeros((s, s, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(Wi, x0 + s), min(H, y0 + s)
        canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = im[sy0:sy1, sx0:sx1]
        out.append(cv2.cvtColor(cv2.resize(canvas, (SIDE, SIDE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return out


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    sel = json.loads(Path(sys.argv[1]).read_text())
    if "--limit" in sys.argv:
        sel = sel[:int(sys.argv[sys.argv.index("--limit") + 1])]
    proc = AutoProcessor.from_pretrained(MODEL[0], revision=MODEL[1])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL[0], revision=MODEL[1], dtype=torch.bfloat16, use_safetensors=True).eval().to("cuda")
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")]
    msgs = [{"role": "user", "content": [{"type": "image"}] * 8 + [{"type": "text", "text": QUESTION}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    progress.start("videodet VLM check", total=len(sel))
    caps, res, t0 = {}, [], time.time()
    tsum = np.zeros(3)                         # seconds in decode+crop, processor, model forward
    for i, w in enumerate(sorted(sel, key=lambda w: (w["video"], w["start"]))):
        v = w["video"]
        if v not in caps:
            caps = {v: Reader(VIDEOS / f"{Wd.VIDEO_OF[v]}.mp4")}      # one open video at a time
        ta = time.time()
        ims = [Image.fromarray(f) for f in crop_frames(caps[v], w["start"], w["crop"])]
        tb = time.time()
        inp = proc(text=[prompt], images=ims, return_tensors="pt").to("cuda")
        tc = time.time()
        with torch.inference_mode():
            logits = model(**inp, logits_to_keep=1).logits[0, -1].float()
        torch.cuda.synchronize()
        tsum += [tb - ta, tc - tb, time.time() - tc]
        lp = torch.log_softmax(logits, -1)
        py, pn = lp[yes].exp().sum().item(), lp[no].exp().sum().item()
        top = tok.decode(int(logits.argmax()))
        res.append(dict(w, p_yes=py / max(py + pn, 1e-9), mass=py + pn, top=top))
        if i < 5 or i % 50 == 0:
            print(f"{i} V{v} {w['start']} {w['group']}: P(yes)={res[-1]['p_yes']:.3f} mass={py + pn:.3f} top={top!r} "
                  f"tokens={inp['input_ids'].shape[1]} {time.time() - t0:.0f}s  "
                  f"[decode {tsum[0]:.0f}s, processor {tsum[1]:.0f}s, forward {tsum[2]:.0f}s]", flush=True)
        progress.tick(1, note=f"V{v}")
        if i % 25 == 24:                      # partial results survive a time-out
            (OUT / "vlm_answers.json").write_text(json.dumps(res, indent=0))
    (OUT / "vlm_answers.json").write_text(json.dumps(res, indent=0))
    progress.finish()
    print(f"done {len(res)} windows in {time.time() - t0:.0f}s [decode {tsum[0]:.0f}s, processor {tsum[1]:.0f}s, forward {tsum[2]:.0f}s]")


if __name__ == "__main__":
    main()
