"""Diagnose VLM slowness: time each stage for a few windows (decode+crop, processor, prefill forward,
generate) and report attention implementation, dtypes, device placement, GPU clocks.
    python external/vlm_timing.py external/vlm_bench.json"""
import json, os, sys, time, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
import torch
from external.vlm_label import Reader, crop, question
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
name, rev = "Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
print(subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,utilization.gpu,memory.used", "--format=csv"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "cuda", torch.version.cuda, "cpu threads", torch.get_num_threads())
proc = AutoProcessor.from_pretrained(name, revision=rev)
for attn in ("sdpa",):
    m = Qwen3VLForConditionalGeneration.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation=attn).eval()
    print("attn", m.config._attn_implementation, "| vision dtype", next(m.model.visual.parameters()).dtype, "| devices", {str(p.device) for p in m.parameters()})
    sel = sorted(json.loads(Path(sys.argv[1]).read_text()), key=lambda w: (w["file"], w["start"]))[:3]
    for side in (448, 336):
        import external.vlm_label as VL; VL.SIDE = side
        for n in (16, 8):
            w = sel[0]; rd = Reader(Path("/l/users/<user>/punch_videodet/videodet/raw/videos") / f"{w['file']}.mp4")
            t0 = time.time(); fr = rd.frames(int(w["start"]), 16)[:: 16 // n]; ims = [Image.fromarray(f) for f in crop(fr, w["crop"])]; t1 = time.time()
            msgs = [{"role": "user", "content": [{"type": "image"}] * len(ims) + [{"type": "text", "text": question(len(ims), 30, None)}]}]
            inp = proc(text=[proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], images=ims, return_tensors="pt").to("cuda:0"); torch.cuda.synchronize(); t2 = time.time()
            with torch.inference_mode():
                m(**inp, logits_to_keep=1); torch.cuda.synchronize(); t3 = time.time()
                m(**inp, logits_to_keep=1); torch.cuda.synchronize(); t4 = time.time()
                m.generate(**inp, max_new_tokens=10, do_sample=False); torch.cuda.synchronize(); t5 = time.time()
            print(f"side {side} frames {n}: tokens {inp['input_ids'].shape[1]} | decode+crop {t1-t0:.1f}s processor {t2-t1:.1f}s "
                  f"forward#1 {t3-t2:.1f}s forward#2 {t4-t3:.1f}s generate {t5-t4:.1f}s", flush=True)
print(subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,utilization.gpu", "--format=csv"], capture_output=True, text=True).stdout)
