"""HPC GPU job: punch TYPE (straight / hook / uppercut) from pixels. VideoMAE on the BoxingWeb punch clips
(bw_clips.json + bw_44x224.npy: 44 frames at 30 fps around the middle of each labelled punch, crop centred on the
punching glove, side 2.4 body scales). 16 frames centred on the middle (+-2 frames jitter in training).
Train = BoxingWeb train rounds minus the last 5 (validation picks the epoch); test = the authors' test split.
Baseline on the same test punches: skeleton detector 72.9% family right, macro recall 70.1% (boxingweb/type_baseline.py).
Saves per test clip the 3 probabilities for fusion with the skeleton on the laptop.
    python type_finetune.py BW.json BW.npy OUT.npz [--model large] [--save-model PATH] [--tiny]
"""
import argparse, json, os, shutil, time
import numpy as np, torch, torch.nn.functional as F
IDS = {"large": ("MCG-NJU/videomae-large-finetuned-kinetics", "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"),
       "base": ("MCG-NJU/videomae-base-finetuned-kinetics", "488eb9a0565f257b32866000305c8178965eb9f6")}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]; C0 = 22
FAM = {"lstraight": 0, "rstraight": 0, "lhook": 1, "rhook": 1, "luppercut": 2, "ruppercut": 2}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clips"); ap.add_argument("npy"); ap.add_argument("out"); ap.add_argument("--model", default="large"); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--bs", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save-model"); ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cpu" if a.tiny else "cuda"; amp = not a.tiny; t0 = time.time()
    clips = json.load(open(a.clips)); y = np.array([FAM[c["technique"]] for c in clips]); split = np.array([c["split"] for c in clips]); rid_ = np.array([c["id"] for c in clips])
    if a.tiny:
        A = np.load(a.npy, mmap_mode="r")
    else:
        loc = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bw.npy"); shutil.copy(a.npy, loc); A = np.load(loc, mmap_mode="r")
    rounds = sorted(set(rid_[split == "train"])); val_r = rounds[-5:]; va = np.flatnonzero(np.isin(rid_, val_r)); tr = np.flatnonzero((split == "train") & ~np.isin(rid_, val_r)); te = np.flatnonzero(split == "test")
    if a.tiny:
        tr, va, te = tr[:12], va[:6], te[:6]
    print(f"clips ready in {time.time() - t0:.0f}s, shape {A.shape}; train {len(tr)}, validation {len(va)}, test {len(te)}; train classes {np.bincount(y[tr], minlength=3).tolist()}", flush=True)
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    if a.tiny:
        net = VideoMAEForVideoClassification(VideoMAEConfig(num_labels=3, hidden_size=48, num_hidden_layers=1, num_attention_heads=2, intermediate_size=96)).to(dev)
    else:
        rid, rev = IDS[a.model]; net = VideoMAEForVideoClassification.from_pretrained(rid, revision=rev, use_safetensors=True, num_labels=3, ignore_mismatched_sizes=True, attn_implementation="sdpa").to(dev)
    head = [p for n, p in net.named_parameters() if n.startswith(("classifier", "fc_norm"))]; body = [p for n, p in net.named_parameters() if not n.startswith(("classifier", "fc_norm"))]
    opt = torch.optim.AdamW([dict(params=body, lr=a.lr), dict(params=head, lr=1e-3)], weight_decay=0.05); nb = max(len(tr) // a.bs, 1); steps = a.epochs * nb
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / nb) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    cnt = np.bincount(y[tr], minlength=3).astype(float); w = torch.tensor((len(tr) / (3 * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32, device=dev)
    mean, std = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    def prep(x, train):
        x = x.permute(0, 1, 4, 2, 3).float() / 255; B = x.shape[0]
        if train:
            if rng.random() < 0.5:
                x = x.flip(-1)                                # mirror: the family (straight / hook / uppercut) is unchanged
            s = rng.uniform(0.75, 1.0); side = int(224 * s); ox, oy = int(rng.integers(0, 224 - side + 1)), int(rng.integers(0, 224 - side + 1))
            x = F.interpolate(x[..., oy:oy + side, ox:ox + side].reshape(B * 16, 3, side, side), size=224, mode="bilinear", align_corners=False).reshape(B, 16, 3, 224, 224)
            x = ((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1)).clamp(0, 1)
        return (x - mean) / std
    def take(idx, offs):
        return torch.from_numpy(np.stack([A[i, C0 - 8 + d:C0 + 8 + d] for i, d in zip(idx, offs)])).to(dev)
    def fwd(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return net(pixel_values=x).logits.float()
    @torch.no_grad()
    def predict(idx):
        net.eval(); P = []
        for k in range(0, len(idx), 24):
            ii = idx[k:k + 24]; P.append(torch.softmax(fwd(prep(take(ii, [0] * len(ii)), False)), 1).cpu().numpy())
        return np.concatenate(P)
    def score(idx):
        P = predict(idx); yh = P.argmax(1); rec = [float(np.mean(yh[y[idx] == c] == c)) if (y[idx] == c).any() else 0.0 for c in range(3)]; return float(np.mean(yh == y[idx])), float(np.mean(rec)), rec, P
    best = (-1, None); best_state = None; hist = []
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr); tl = 0.0
        for k in range(nb):
            ii = np.sort(perm[k * a.bs:(k + 1) * a.bs]); loss = F.cross_entropy(fwd(prep(take(ii, rng.integers(-2, 3, len(ii))), True)), torch.from_numpy(y[ii]).to(dev), weight=w, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        acc, mac, rec, _ = score(va); hist.append(dict(epoch=ep + 1, loss=tl / nb, val_acc=acc, val_macro=mac))
        print(f"epoch {ep + 1}/{a.epochs} loss {tl / nb:.3f} | validation: family right {acc:.3f}, macro recall {mac:.3f} (straight {rec[0]:.2f}, hook {rec[1]:.2f}, uppercut {rec[2]:.2f}) | {time.time() - t0:.0f}s", flush=True)
        if acc + mac > best[0]:
            best = (acc + mac, ep + 1); best_state = {k2: v.detach().cpu().clone() for k2, v in net.state_dict().items()}
    net.load_state_dict(best_state); acc, mac, rec, P = score(te)
    print(f"best validation epoch {best[1]}\nTEST ({len(te)} punches): family right {acc:.3f}, macro recall {mac:.3f} (straight {rec[0]:.2f}, hook {rec[1]:.2f}, uppercut {rec[2]:.2f})   skeleton baseline: 0.729 / 0.701", flush=True)
    if a.save_model and not a.tiny:
        os.makedirs(os.path.dirname(a.save_model), exist_ok=True); torch.save(best_state, a.save_model + ".part"); os.replace(a.save_model + ".part", a.save_model)
    np.savez_compressed(a.out + ".tmp.npz", idx=te, P=P.astype(np.float32), y=y[te], hist=json.dumps(hist), best_epoch=best[1]); os.replace(a.out + ".tmp.npz", a.out); print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
