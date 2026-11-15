"""Profile one Qwen3-VL-8B forward (336 px, 8 frames): vision tower vs language model time, top CUDA ops."""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
import torch
import external.vlm_label as VL
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
VL.SIDE = 336
name, rev = "Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
proc = AutoProcessor.from_pretrained(name, revision=rev)
m = Qwen3VLForConditionalGeneration.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map="cuda:0").eval()
w = sorted(json.loads(Path(sys.argv[1]).read_text()), key=lambda w: (w["file"], w["start"]))[0]
rd = VL.Reader(Path("/l/users/<user>/punch_videodet/videodet/raw/videos") / f"{w['file']}.mp4")
ims = [Image.fromarray(f) for f in VL.crop(rd.frames(int(w["start"]), 16)[::2], w["crop"])]
msgs = [{"role": "user", "content": [{"type": "image"}] * 8 + [{"type": "text", "text": "Describe."}]}]
inp = proc(text=[proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)], images=ims, return_tensors="pt").to("cuda:0")
print("pixel_values", tuple(inp["pixel_values"].shape), inp["pixel_values"].dtype, "grid", inp["image_grid_thw"].tolist()[:2], "tokens", inp["input_ids"].shape[1])
with torch.inference_mode():
    torch.cuda.synchronize(); t = time.time()
    m.model.visual(inp["pixel_values"].to(torch.bfloat16), grid_thw=inp["image_grid_thw"]); torch.cuda.synchronize()
    print(f"vision tower alone: {time.time() - t:.1f}s", flush=True)
    emb = m.model.get_input_embeddings()(inp["input_ids"])
    torch.cuda.synchronize(); t = time.time()
    m.model.language_model(inputs_embeds=emb); torch.cuda.synchronize()
    print(f"language model alone (text-only positions): {time.time() - t:.1f}s", flush=True)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        m(**inp, logits_to_keep=1); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
