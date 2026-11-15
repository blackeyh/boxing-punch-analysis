"""VLM labeller (branch vlmlabel): ask an open vision-language model whether a boxer throws a punch in a
short clip, and if so with which hand and what kind. Runs on the HPC GPU (32 GB).

    python external/vlm_label.py SELECTION.json --model 32b|8b [--videos DIR] [--limit N]

Selection rows: file (video stem in --videos), start, n (frames), crop [x, y, side] px, optional
"who" (appearance description of the boxer to judge, for sparring), plus any extra fields (kept).
The model sees all n frames, cropped and resized to SIDE px, as an ordered image sequence.
Score: P(Yes) from the first answer token's Yes/No logits; the rest of the greedy answer gives
hand + kind ("left straight", ...). Output $RUN_DIR/vlm_<model>.json.
Models: Qwen3-VL-32B-Instruct in 4-bit NF4 (bitsandbytes; vision tower kept in bf16) or
Qwen3-VL-8B-Instruct in bf16. Weights from HF_HOME=/l/users/<user>/hf_cache (offline).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/l/users/<user>/pylibs_bnb")            # bitsandbytes, outside the conda env
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from progress import progress  # noqa: E402

# (name, HF_HOME, revision): the 8B is the exact snapshot videodet used; the 32B is in a fresh cache
# (the first 32B download, hf_cache/, stalled at 3 GB)
MODELS = {"32b": ("Qwen/Qwen3-VL-32B-Instruct", "/l/users/<user>/hf_cache3", None),
          "8b": ("Qwen/Qwen3-VL-8B-Instruct", os.path.expanduser("~/.cache/huggingface"), "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")}
SIDE = 448
OUT = Path(os.environ.get("RUN_DIR", ROOT / "runs" / "vlm"))


def question(n, fps, who):
    subject = f"the boxer {who}" if who else "the boxer"
    return (f"These {n} images are consecutive frames, in order, from a boxing video (about {n / fps:.1f} seconds). "
            f"Look at {subject}. Does this boxer throw a punch (a jab, cross, hook or uppercut, at a bag, an opponent or "
            f"in the air) during these frames? Only an actual punch counts: standing in guard, footwork, bouncing, "
            f"slipping, ducking, blocking, parrying, pushing, holding and resting do not. "
            f"Answer 'Yes' or 'No'. If Yes, continue with the punching hand from the boxer's OWN point of view "
            f"(left or right) and the kind (straight, hook or uppercut), e.g. 'Yes, left straight'.")


class Reader:
    def __init__(self, path):
        self.cap, self.pos, self.buf = cv2.VideoCapture(str(path)), 0, {}
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))

    def frames(self, start, n):
        assert start >= self.pos - 64, "selection must be sorted by start within a video"
        while self.pos < start - 64:
            self.cap.grab(); self.pos += 1
        while self.pos < start + n:
            ok, im = self.cap.read()
            if not ok:
                raise RuntimeError(f"decode failed at {self.pos}")
            self.buf[self.pos] = im; self.buf.pop(self.pos - 64, None); self.pos += 1
        return [self.buf[f] for f in range(start, start + n)]


def crop(ims, c):
    x0, y0, s = int(round(c[0])), int(round(c[1])), int(round(c[2]))
    out = []
    for im in ims:
        H, Wi = im.shape[:2]
        canvas = np.zeros((s, s, 3), np.uint8)
        sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(Wi, x0 + s), min(H, y0 + s)
        if sx1 > sx0 and sy1 > sy0:
            canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = im[sy0:sy1, sx0:sx1]
        out.append(cv2.cvtColor(cv2.resize(canvas, (SIDE, SIDE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return out


def fast_patch_embed(model):
    """The vision tower's first layer is a Conv3d whose kernel == stride (non-overlapping patches), i.e.
    exactly one matrix multiply per patch. On this HPC (torch 2.9 + cu128, bf16) that Conv3d took ~98%
    of the forward time (13.4 s of 13.6 s for 8 frames). Replace it by the identical linear map with
    the SAME weights (reshaped)."""
    pe = model.model.visual.patch_embed
    conv = pe.proj
    W = conv.weight.detach().reshape(conv.out_channels, -1).clone()
    b = conv.bias.detach().clone() if conv.bias is not None else None
    k = W.shape[1]

    def forward(hidden_states):
        x = hidden_states.reshape(-1, k).to(W.dtype)
        return torch.nn.functional.linear(x, W, b)
    pe.forward = forward
    return model


def parse(text):
    t = text.lower()
    hand = "left" if "left" in t else ("right" if "right" in t else None)
    kind = next((k for k in ("straight", "hook", "uppercut") if k in t), None)
    return hand, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("--model", choices=list(MODELS), default="32b")
    ap.add_argument("--videos", default="/l/users/<user>/punch_videodet/videodet/raw/videos")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--slow-conv", action="store_true", help="keep the original Conv3d patch embedding")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    args = ap.parse_args()
    name, hf_home, rev = MODELS[args.model]
    os.environ["HF_HOME"] = hf_home
    from PIL import Image
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    sel = json.loads(Path(args.selection).read_text())
    sel = sorted(sel, key=lambda w: (w["file"], w["start"]))[: args.limit]
    proc = AutoProcessor.from_pretrained(name, revision=rev)
    kw = dict(dtype=torch.bfloat16, use_safetensors=True, device_map="cuda:0", revision=rev)
    if args.model == "32b":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                                                       llm_int8_skip_modules=["visual", "lm_head"])
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(name, **kw).eval()
    if not args.slow_conv:
        fast_patch_embed(model)
    print(f"loaded {name} in {time.time() - t0:.0f}s | GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
    tok = proc.tokenizer
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No")]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"vlm_{args.model}{args.tag}.json"
    progress.start(f"VLM {args.model}", total=len(sel))
    readers, res, t0 = {}, [], time.time()
    for i, w in enumerate(sel):
        if w["file"] not in readers:
            readers = {w["file"]: Reader(Path(args.videos) / f"{w['file']}.mp4")}
        rd = readers[w["file"]]
        ims = [Image.fromarray(f) for f in crop(rd.frames(int(w["start"]), int(w["n"])), w["crop"])]
        msgs = [{"role": "user", "content": [{"type": "image"}] * len(ims) +
                 [{"type": "text", "text": question(len(ims), rd.fps, w.get("who"))}]}]
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[prompt], images=ims, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=10, do_sample=False, output_scores=True, return_dict_in_generate=True)
        lp = torch.log_softmax(g.scores[0][0].float(), -1)
        py, pn = lp[yes].exp().sum().item(), lp[no].exp().sum().item()
        text = tok.decode(g.sequences[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        hand, kind = parse(text)
        res.append(dict(w, p_yes=py / max(py + pn, 1e-9), mass=py + pn, answer=text, hand=hand, kind=kind))
        if i < 5 or i % 25 == 0:
            print(f"{i} {w['file']} {w['start']} [{w.get('label', w.get('who', ''))}] P(yes)={res[-1]['p_yes']:.2f} "
                  f"'{text}' tokens={inp['input_ids'].shape[1]} {time.time() - t0:.0f}s", flush=True)
        progress.tick(1, note=w["file"])
        if i % 20 == 19:
            out_path.write_text(json.dumps(res, indent=0))
    out_path.write_text(json.dumps(res, indent=0))
    progress.finish()
    print(f"done {len(res)} windows in {time.time() - t0:.0f}s ({(time.time() - t0) / max(len(res), 1):.1f} s/window)")


if __name__ == "__main__":
    main()
