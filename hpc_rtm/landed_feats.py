"""HPC GPU job: VideoMAE features of 16-frame contact clips (olympic/landed_clips.py).
For each clip: video frames [f-8, f+7] (50 fps = 0.32 s), one fixed square crop, resized to 224.
Two crops per clip: "glove" (centre = puncher's glove, run-time available) and "box" (centre = referee's box).
Features: VideoMAE-L and VideoMAE-B (Kinetics-400), mean-pooled tokens after fc_norm (as videodet/extract.py).
    python landed_feats.py CLIPS.json VIDEOS_DIR OUT.npz PART N_PARTS
"""
import json, os, sys, time
import cv2, numpy as np, torch
IDS = {"large": ("MCG-NJU/videomae-large-finetuned-kinetics", "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"),
       "base": ("MCG-NJU/videomae-base-finetuned-kinetics", "488eb9a0565f257b32866000305c8178965eb9f6")}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def crop(im, cx, cy, side):
    h, w = im.shape[:2]; s = int(max(side, 64)); x1, y1 = int(cx - s / 2), int(cy - s / 2); out = np.zeros((s, s, 3), np.uint8)
    xa, ya, xb, yb = max(x1, 0), max(y1, 0), min(x1 + s, w), min(y1 + s, h)
    if xb > xa and yb > ya:
        out[ya - y1:yb - y1, xa - x1:xb - x1] = im[ya:yb, xa:xb]
    return cv2.cvtColor(cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

def main():
    clips = json.load(open(sys.argv[1])); vids, out_path, part, nparts = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    ids = sorted({c["id"] for c in clips}); ids = ids[part::nparts]
    from transformers import VideoMAEForVideoClassification
    dev = "cuda"; nets = {m: VideoMAEForVideoClassification.from_pretrained(r, revision=rev, use_safetensors=True, attn_implementation="sdpa").half().eval().to(dev) for m, (r, rev) in IDS.items()}
    mean, std = torch.tensor(MEAN, device=dev), torch.tensor(STD, device=dev)
    res = {f"{m}_{k}": [] for m in nets for k in ("glove", "box")}; order = []; t0 = time.time()
    def flush(batch):
        if not batch:
            return
        for k, j in (("glove", 0), ("box", 1)):
            x = torch.from_numpy(np.stack([b[j] for b in batch])).to(dev).float().div_(255); x = ((x - mean) / std).permute(0, 1, 4, 2, 3).contiguous().half()
            with torch.no_grad():
                for m, net in nets.items():
                    seq = net.videomae(x)[0]; f = net.fc_norm(seq.mean(1)) if net.fc_norm is not None else seq[:, 0]
                    res[f"{m}_{k}"].append(f.float().cpu().numpy().astype(np.float16))
    for cid in ids:
        mine = sorted((c for c in clips if c["id"] == cid), key=lambda c: c["frame"]); idx = [clips.index(c) for c in mine]
        need = {}
        for n, c in enumerate(mine):
            for f in range(c["frame"] - 8, c["frame"] + 8):
                need.setdefault(f, []).append(n)
        bufs = [[None] * 16 for _ in mine]; cap = cv2.VideoCapture(os.path.join(vids, cid + ".mp4")); k = 0; last = max(need); batch = []; done = 0
        while k <= last:
            ok = cap.grab()
            if not ok:
                break
            if k in need:
                ok, im = cap.retrieve()
                for n in need[k]:
                    c = mine[n]; bufs[n][k - (c["frame"] - 8)] = (crop(im, c["cx"], c["cy"], c["side"]), crop(im, c["bx"], c["by"], c["side"]))
            k += 1
            while done < len(mine) and mine[done]["frame"] + 7 < k:
                b = bufs[done]; first = next((x for x in b if x is not None), None)
                if first is None:
                    first = (np.zeros((224, 224, 3), np.uint8),) * 2
                b = [x if x is not None else first for x in b]
                batch.append((np.stack([x[0] for x in b]), np.stack([x[1] for x in b]))); order.append(idx[done]); bufs[done] = None; done += 1
                if len(batch) == 24:
                    flush(batch); batch = []
        while done < len(mine):
            b = bufs[done]; first = next((x for x in b if x is not None), (np.zeros((224, 224, 3), np.uint8),) * 2); b = [x if x is not None else first for x in b]
            batch.append((np.stack([x[0] for x in b]), np.stack([x[1] for x in b]))); order.append(idx[done]); done += 1
        flush(batch)
        print(f"[{cid}] {len(mine)} clips, total {len(order)}, {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(out_path + ".tmp.npz", order=np.array(order), **{k: np.concatenate(v) for k, v in res.items() if v})
    os.replace(out_path + ".tmp.npz", out_path); print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
