"""HPC GPU job: scan clips in time with a SAVED pixel contact + landed model (no training).
Each stored clip is T frames with the nominal centre at index C0; for each scan offset the 16-frame window centred
there is judged: [landed, blocked, missed, off-centre].  Each clip is judged by the model that never saw its gym
(clip["gym"]); with --model-gym G every clip uses models/cl_large_heldout_G.pt (for data outside the Olympic set).
    python cl_scan.py CLIPS.json CLIPS.npy OUT.npz MODELS_DIR C0 OFFSETS(comma list) [--model-gym G]
"""
import json, os, shutil, sys
import numpy as np, torch
RID, REV = "MCG-NJU/videomae-large-finetuned-kinetics", "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def main():
    clips = json.load(open(sys.argv[1])); out, mdir, C0 = sys.argv[3], sys.argv[4], int(sys.argv[5]); offs = [int(v) for v in sys.argv[6].split(",")]
    fixed = sys.argv[sys.argv.index("--model-gym") + 1] if "--model-gym" in sys.argv else None; dev = "cuda"
    loc = os.path.join(os.environ.get("TMPDIR", "/tmp"), "scan.npy"); shutil.copy(sys.argv[2], loc); A = np.load(loc, mmap_mode="r"); print("clips", A.shape, flush=True)
    from transformers import VideoMAEForVideoClassification
    mean, std = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1)
    S = np.full((len(clips), len(offs), 4), np.nan, np.float32); gyms = [fixed] if fixed else sorted({c["gym"] for c in clips})
    for gym in gyms:
        net = VideoMAEForVideoClassification.from_pretrained(RID, revision=REV, use_safetensors=True, num_labels=4, ignore_mismatched_sizes=True, attn_implementation="sdpa")
        net.load_state_dict(torch.load(os.path.join(mdir, f"cl_large_heldout_{gym}.pt"), map_location="cpu", weights_only=True)); net = net.to(dev).eval()
        idx = np.arange(len(clips)) if fixed else np.array([n for n, c in enumerate(clips) if c["gym"] == gym])
        with torch.no_grad():
            for j, d in enumerate(offs):
                for k in range(0, len(idx), 24):
                    ii = idx[k:k + 24]; x = torch.from_numpy(np.stack([A[i, C0 - 8 + d:C0 + 8 + d] for i in ii])).to(dev).permute(0, 1, 4, 2, 3).float() / 255
                    with torch.autocast("cuda", dtype=torch.float16):
                        S[ii, j] = torch.softmax(net(pixel_values=(x - mean) / std).logits.float(), 1).cpu().numpy()
                print(f"model heldout_{gym}: offset {d} done ({len(idx)} clips)", flush=True)
        del net
    np.savez_compressed(out + ".tmp.npz", scan=S, offsets=np.array(offs)); os.replace(out + ".tmp.npz", out); print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
