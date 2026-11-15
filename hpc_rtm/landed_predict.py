"""HPC GPU job: judge clips with the SAVED fine-tuned landed models (no training).
Each clip is judged by the model that never saw its gym (models/landed_{TAG}_{MODEL}_heldout_{gym}.pt).
    python landed_predict.py CLIPS.json CLIPS.npy OUT.npz MODELS_DIR TAG MODEL
"""
import json, os, shutil, sys
import numpy as np, torch
IDS = {"large": ("MCG-NJU/videomae-large-finetuned-kinetics", "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"),
       "base": ("MCG-NJU/videomae-base-finetuned-kinetics", "488eb9a0565f257b32866000305c8178965eb9f6")}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def main():
    clips = json.load(open(sys.argv[1])); out, mdir, tag, model = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]; dev = "cuda"
    loc = os.path.join(os.environ.get("TMPDIR", "/tmp"), "pred.npy"); shutil.copy(sys.argv[2], loc); A = np.load(loc, mmap_mode="r")
    from transformers import VideoMAEForVideoClassification
    rid, rev = IDS[model]; mean, std = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1)
    P = np.full((len(clips), 3), np.nan, np.float32)
    for gym in sorted({c["gym"] for c in clips}):
        net = VideoMAEForVideoClassification.from_pretrained(rid, revision=rev, use_safetensors=True, num_labels=3, ignore_mismatched_sizes=True, attn_implementation="sdpa")
        net.load_state_dict(torch.load(os.path.join(mdir, f"landed_{tag}_{model}_heldout_{gym}.pt"), map_location="cpu", weights_only=True)); net = net.to(dev).eval()
        idx = np.array([n for n, c in enumerate(clips) if c["gym"] == gym])
        with torch.no_grad():
            for k in range(0, len(idx), 32):
                ii = idx[k:k + 32]; x = torch.from_numpy(np.stack([A[i] for i in ii])).to(dev)[:, 2:18].permute(0, 1, 4, 2, 3).float() / 255
                with torch.autocast("cuda", dtype=torch.float16):
                    P[ii] = torch.softmax(net(pixel_values=(x - mean) / std).logits.float(), 1).cpu().numpy()
        print(f"gym {gym}: judged {len(idx)} clips", flush=True); del net
    np.savez_compressed(out + ".tmp.npz", extra_idx=np.arange(len(clips)), extra_P=P); os.replace(out + ".tmp.npz", out); print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
