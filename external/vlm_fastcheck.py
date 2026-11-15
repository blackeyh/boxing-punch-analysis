"""Check the fast patch embedding: same vision output as the Conv3d (max abs diff) and the speed-up."""
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
name, rev = "Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
proc = AutoProcessor.from_pretrained(name, revision=rev)
m = Qwen3VLForConditionalGeneration.from_pretrained(name, revision=rev, dtype=torch.bfloat16, device_map="cuda:0").eval()
w = sorted(json.loads(Path(sys.argv[1]).read_text()), key=lambda w: (w["file"], w["start"]))[0]
rd = VL.Reader(Path("/l/users/<user>/punch_videodet/videodet/raw/videos") / f"{w['file']}.mp4")
ims = [Image.fromarray(f) for f in VL.crop(rd.frames(int(w["start"]), 16), w["crop"])]
inp = proc(text=[proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}] * 16 + [{"type": "text", "text": VL.question(16, 30, None)}]}],
                                          tokenize=False, add_generation_prompt=True)], images=ims, return_tensors="pt").to("cuda:0")
pv = inp["pixel_values"].to(torch.bfloat16)
with torch.inference_mode():
    torch.cuda.synchronize(); t = time.time(); a = m.model.visual.patch_embed(pv); torch.cuda.synchronize(); t_conv = time.time() - t
    la = m(**inp, logits_to_keep=1).logits[0, -1].float()
    VL.fast_patch_embed(m)
    m.model.visual.patch_embed(pv); torch.cuda.synchronize()
    t = time.time(); b = m.model.visual.patch_embed(pv); torch.cuda.synchronize(); t_lin = time.time() - t
    t = time.time(); lb = m(**inp, logits_to_keep=1).logits[0, -1].float(); torch.cuda.synchronize(); t_full = time.time() - t
print(f"patch embed: conv3d {t_conv:.2f}s vs linear {t_lin:.4f}s | max abs diff {(a.float() - b.float()).abs().max().item():.4f} "
      f"(values ~{a.float().abs().mean().item():.3f}) | final-token logits max diff {(la - lb).abs().max().item():.3f} | "
      f"same top token {bool(la.argmax() == lb.argmax())} | full forward now {t_full:.2f}s ({inp['input_ids'].shape[1]} tokens)", flush=True)
