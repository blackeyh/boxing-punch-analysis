"""HPC GPU job: fine-tune a video model end to end on landed / blocked / missed (Olympic set, referee labels).

Split by gym: train on the bouts of one gym (2 of them held back as validation, to pick the epoch), test on the other.
Model: VideoMAE (Kinetics-400 checkpoint), new 3-way head. 16 of the 20 stored frames (random 0-4 frame offset in
training, centred at test), 224x224. Augmentation: horizontal flip, random zoom/shift (scale 0.75-1), brightness /
contrast. AdamW, backbone lr LR, head lr 1e-3, cosine with 1 warm-up epoch, label smoothing 0.1, square-root
class weights, mixed precision. Saves test probabilities at the best-validation epoch and at the last epoch.
    python landed_finetune.py CLIPS.json CLIPS.npy OUT.npz HELD_GYM [--model base|large] [--epochs 12] [--lr 3e-5] [--bs 12]
"""
import argparse, json, os, shutil, sys, time
import numpy as np, torch, torch.nn.functional as F
IDS = {"large": ("MCG-NJU/videomae-large-finetuned-kinetics", "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"),
       "base": ("MCG-NJU/videomae-base-finetuned-kinetics", "488eb9a0565f257b32866000305c8178965eb9f6")}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clips"); ap.add_argument("npy"); ap.add_argument("out"); ap.add_argument("held")
    ap.add_argument("--model", default="base"); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=12); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra-json"); ap.add_argument("--extra-npy"); ap.add_argument("--save-model")
    ap.add_argument("--train-extra", action="store_true", help="also TRAIN on the extra clips of the training gym that carry a transferred referee label")      # also judge these clips (held-out gym only) with the best-validation model
    a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda"
    clips = json.load(open(a.clips)); y = np.array([c["y"] for c in clips]); gym = np.array([c["gym"] for c in clips]); bout = np.array([c["id"] for c in clips])
    local = os.path.join(os.environ.get("TMPDIR", "/tmp"), "clips.npy"); t0 = time.time(); shutil.copy(a.npy, local); A = np.load(local, mmap_mode="r")
    print(f"clips copied to node-local disk in {time.time() - t0:.0f}s, shape {A.shape}", flush=True)
    tr_bouts = sorted(set(bout[gym != a.held])); val_bouts = set(tr_bouts[::7][:2])
    te = np.flatnonzero(gym == a.held); va = np.flatnonzero(np.isin(bout, list(val_bouts))); tr = np.flatnonzero((gym != a.held) & ~np.isin(bout, list(val_bouts)))
    EX = B = None; etr = np.array([], int); ey = None
    if a.extra_json:
        EX = json.load(open(a.extra_json)); loc2 = os.path.join(os.environ.get("TMPDIR", "/tmp"), "extra.npy"); shutil.copy(a.extra_npy, loc2); B = np.load(loc2, mmap_mode="r")
        MAP = {"head": 0, "body": 0, "block": 1, "miss": 2}; ey = np.array([MAP.get(c["label"], -1) for c in EX])
        if a.train_extra:
            etr = np.array([n for n, c in enumerate(EX) if c["gym"] != a.held and c["label"] is not None and c["id"] not in val_bouts])
    print(f"extra training clips (detector's own punches with a transferred label): {len(etr)}", flush=True)
    print(f"held-out gym {a.held}: train {len(tr)}, validation {len(va)} ({sorted(val_bouts)}), test {len(te)}; train classes {np.bincount(y[tr], minlength=3).tolist()}", flush=True)
    from transformers import VideoMAEForVideoClassification
    rid, rev = IDS[a.model]
    net = VideoMAEForVideoClassification.from_pretrained(rid, revision=rev, use_safetensors=True, num_labels=3, ignore_mismatched_sizes=True, attn_implementation="sdpa").to(dev)
    head = [p for n, p in net.named_parameters() if n.startswith("classifier") or n.startswith("fc_norm")]; body = [p for n, p in net.named_parameters() if not (n.startswith("classifier") or n.startswith("fc_norm"))]
    opt = torch.optim.AdamW([dict(params=body, lr=a.lr), dict(params=head, lr=1e-3)], weight_decay=0.05)
    steps = a.epochs * (len(tr) // a.bs); warm = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    w = torch.tensor((len(tr) / (3 * np.bincount(y[tr], minlength=3))) ** 0.5, dtype=torch.float32, device=dev)
    mean, std = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1); scaler = torch.amp.GradScaler("cuda")
    def batch(idx, train, src=None):
        idx = np.sort(idx); x = torch.from_numpy(np.stack([(A if src is None else src)[i] for i in idx])).to(dev)   # B,20,H,W,3 uint8
        o = int(rng.integers(0, 5)) if train else 2; x = x[:, o:o + 16].permute(0, 1, 4, 2, 3).float() / 255   # B,16,3,H,W
        if train:
            B = x.shape[0]
            if rng.random() < 0.5:
                x = x.flip(-1)
            s = rng.uniform(0.75, 1.0); side = int(224 * s); ox, oy = int(rng.integers(0, 224 - side + 1)), int(rng.integers(0, 224 - side + 1))
            x = F.interpolate(x[..., oy:oy + side, ox:ox + side].reshape(B * 16, 3, side, side), size=224, mode="bilinear", align_corners=False).reshape(B, 16, 3, 224, 224)
            x = ((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1)).clamp(0, 1)
        return (x - mean) / std, torch.from_numpy((y if src is None else ey)[idx]).to(dev)
    @torch.no_grad()
    def predict(idx):
        net.eval(); P = []
        for k in range(0, len(idx), 32):
            x, _ = batch(idx[k:k + 32], False)
            with torch.autocast("cuda", dtype=torch.float16):
                P.append(torch.softmax(net(pixel_values=x).logits.float(), 1).cpu().numpy())
        order = np.argsort(np.concatenate([np.sort(idx[k:k + 32]) for k in range(0, len(idx), 32)])); P = np.concatenate(P)
        srt = np.concatenate([np.sort(idx[k:k + 32]) for k in range(0, len(idx), 32)]); out = np.zeros_like(P); out[np.searchsorted(np.sort(idx), srt)] = P
        return out                                                                                             # rows follow np.sort(idx)
    def summ(P, idx):
        yy = y[np.sort(idx)]; return float((P.argmax(1) == yy).mean()), auc((yy == 0).astype(int), P[:, 0])
    best = (-1, None, None); hist = []; step = 0
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr); tl = 0.0; nb = len(tr) // a.bs; eperm = rng.permutation(etr) if len(etr) else etr; ek = 0
        for k in range(nb):
            x, yy = batch(perm[k * a.bs:(k + 1) * a.bs], True)
            if len(etr):                                        # half as many of the detector's own clips in every step
                h = max(a.bs // 2, 1); ii = eperm[ek:ek + h]; ek = (ek + h) % max(len(eperm) - h, 1)
                x2, y2 = batch(ii, True, B); x = torch.cat([x, x2]); yy = torch.cat([yy, y2])
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(net(pixel_values=x).logits.float(), yy, weight=w, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item(); step += 1
        Pv = predict(va); av, uv = summ(Pv, va); Pt = predict(te); at, ut = summ(Pt, te); hist.append(dict(epoch=ep + 1, loss=tl / nb, val_acc=av, val_auc=uv, test_acc=at, test_auc=ut))
        print(f"epoch {ep + 1}/{a.epochs} loss {tl / nb:.3f} | validation acc {av:.3f} landed-AUC {uv:.3f} | test acc {at:.3f} landed-AUC {ut:.3f} | {time.time() - t0:.0f}s", flush=True)
        if av + uv > best[0]:
            best = (av + uv, ep + 1, Pt.copy()); best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    extra = {}
    if a.extra_json:
        net.load_state_dict(best_state); net.eval(); eidx = np.array([n for n, c in enumerate(EX) if c["gym"] == a.held]); PE = []
        with torch.no_grad():
            for k in range(0, len(eidx), 32):
                x, _ = batch(eidx[k:k + 32], False, B)
                with torch.autocast("cuda", dtype=torch.float16):
                    PE.append(torch.softmax(net(pixel_values=x).logits.float(), 1).cpu().numpy())
        extra = dict(extra_idx=eidx, extra_P=np.concatenate(PE)); print(f"judged {len(eidx)} extra clips of gym {a.held}", flush=True)
    if a.save_model:
        os.makedirs(os.path.dirname(a.save_model), exist_ok=True); torch.save(best_state, a.save_model + ".part"); os.replace(a.save_model + ".part", a.save_model)
    np.savez_compressed(a.out + ".tmp.npz", idx=np.sort(te), P_best=best[2], P_last=Pt, best_epoch=best[1], hist=json.dumps(hist), held=a.held, model=a.model, **extra)
    os.replace(a.out + ".tmp.npz", a.out); print(f"best validation epoch {best[1]}; ALL DONE", flush=True)

if __name__ == "__main__":
    main()
