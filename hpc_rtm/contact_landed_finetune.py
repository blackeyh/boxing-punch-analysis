"""HPC GPU job: ONE video model that finds the contact moment from pixels AND judges the punch.
4 answers per 16-frame clip:  0 landed, 1 blocked, 2 missed  (clip centred on the contact, +-1 frame)
                              3 off-centre                   (clip centred 4-14 frames before or after the contact)
Training clips: the referee's punches, stored as 44 frames (contact at index 22), crop centred on the puncher's glove.
Half of every batch is centred, half is shifted. Split by gym as before (2 training bouts held back for validation).
After training (best validation epoch):
  A. referee clips of the held-out gym, centred: 3-way accuracy (comparable with the earlier 74.2%)
  B. the same clips, SCANNED: 13 positions from -12 to +12 frames; the position with the lowest "off-centre" score
     is taken as the contact -> timing error when the pixels alone must find the moment
  C. the chain's own punches of the held-out gym (44 frames around the skeleton model's rough contact, crop on the
     detector's glove), scanned the same way: saved per position for scoring on the laptop.
    python contact_landed_finetune.py REF.json REF.npy OUT.npz HELD --chain-json C.json --chain-npy C.npy [--model large] [--tiny]
"""
import argparse, json, os, shutil, sys, time
import numpy as np, torch, torch.nn.functional as F
IDS = {"large": ("MCG-NJU/videomae-large-finetuned-kinetics", "0f6adcd5f6902900aa0281f9daacfe52bb3c4ad4"),
       "base": ("MCG-NJU/videomae-base-finetuned-kinetics", "488eb9a0565f257b32866000305c8178965eb9f6")}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
C0, SCAN = 22, list(range(-12, 13, 2))                     # contact index in the stored clip; scan offsets (frames)

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("clips"); ap.add_argument("npy"); ap.add_argument("out"); ap.add_argument("held")
    ap.add_argument("--chain-json"); ap.add_argument("--chain-npy"); ap.add_argument("--save-model")
    ap.add_argument("--bw-json"); ap.add_argument("--bw-npy"); ap.add_argument("--bw-scan")   # extra TRAINING data: BoxingWeb punches, contact found by an earlier scan
    ap.add_argument("--bw-min", type=float, default=0.5, help="use a BoxingWeb punch only if the scan's best centred score is at least this")
    ap.add_argument("--model", default="large"); ap.add_argument("--epochs", type=int, default=14); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--frac", type=float, default=1.0, help="learning curve: train on this share of the training punches")
    ap.add_argument("--mix", type=int, default=-1, help="same-gym test: hold out every 3rd bout (starting at MIX) of BOTH gyms instead of a whole gym")
    ap.add_argument("--all", action="store_true", help="deployment model: train on BOTH gyms (2 bouts held back for validation); the 'test' numbers printed are the validation bouts")
    ap.add_argument("--stride", type=int, default=1, help="2 = every 2nd frame: 16 frames cover 0.64 s instead of 0.32 s")
    ap.add_argument("--neg", action="store_true", help="extra head: REAL punch vs NOT a punch, trained on the chain's own counted punches of the training bouts (chain json needs the field 'real')")
    ap.add_argument("--bs", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tiny", action="store_true", help="laptop test of the code path: toy model, CPU")
    a = ap.parse_args(); ST = a.stride; OFFMAX = 14 if ST == 1 else 6; SHIFT = 8 if ST == 1 else 6; SCAN = list(range(-12, 13, 2)) if ST == 1 else list(range(-6, 7, 2))
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cpu" if a.tiny else "cuda"; amp = not a.tiny
    clips = json.load(open(a.clips)); y = np.array([c["y"] for c in clips]); gym = np.array([c["gym"] for c in clips]); bout = np.array([c["id"] for c in clips])
    t0 = time.time()
    def local(path, name):
        if a.tiny:
            return np.load(path, mmap_mode="r")
        loc = os.path.join(os.environ.get("TMPDIR", "/tmp"), name); shutil.copy(path, loc); return np.load(loc, mmap_mode="r")
    A = local(a.npy, "ref.npy"); print(f"referee clips ready in {time.time() - t0:.0f}s, shape {A.shape}", flush=True); assert A.shape[1] >= C0 + 22
    tr_bouts = sorted(set(bout[gym != a.held])); val_bouts = set(tr_bouts[::7][:2])
    te = np.flatnonzero(gym == a.held); va = np.flatnonzero(np.isin(bout, list(val_bouts))); tr = np.flatnonzero((gym != a.held) & ~np.isin(bout, list(val_bouts)))
    if a.mix >= 0:
        allb = sorted(set(bout)); te_b = set(allb[a.mix::3]); rest = [b for b in allb if b not in te_b]; val_bouts = set(rest[::7][:2])
        te = np.flatnonzero(np.isin(bout, list(te_b))); va = np.flatnonzero(np.isin(bout, list(val_bouts))); tr = np.flatnonzero(~np.isin(bout, list(te_b | val_bouts)))
    if a.all:
        allb = sorted(set(bout)); val_bouts = set(allb[3::14][:2]); va = np.flatnonzero(np.isin(bout, list(val_bouts))); tr = np.flatnonzero(~np.isin(bout, list(val_bouts))); te = va
    if a.frac < 1:
        tr = np.sort(np.random.default_rng(123).choice(tr, int(round(len(tr) * a.frac)), replace=False))
    print(f"held-out gym {a.held}: train {len(tr)}, validation {len(va)}, test {len(te)}; train classes {np.bincount(y[tr], minlength=3).tolist()}", flush=True)
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    if a.tiny:
        net = VideoMAEForVideoClassification(VideoMAEConfig(num_labels=4, hidden_size=48, num_hidden_layers=1, num_attention_heads=2, intermediate_size=96)).to(dev)
    else:
        rid, rev = IDS[a.model]
        net = VideoMAEForVideoClassification.from_pretrained(rid, revision=rev, use_safetensors=True, num_labels=4, ignore_mismatched_sizes=True, attn_implementation="sdpa").to(dev)
    BW = None
    if a.bw_json:                                             # second task on the same features: BoxingWeb "effective" (clean scoring blow) vs not
        bw = json.load(open(a.bw_json)); Bw = local(a.bw_npy, "bw.npy"); zs = np.load(a.bw_scan); cen_b = 1 - zs["scan"][:, :, 3]; kb = np.nan_to_num(cen_b, nan=-1).argmax(1)
        bw_off = zs["offsets"][kb]; bw_ok = np.nan_to_num(cen_b, nan=-1).max(1) >= a.bw_min; bw_y = np.array([int(c["landed"]) for c in bw])
        rounds = sorted({c["id"] for c in bw if c["split"] == "train"}); bw_val_r = set(rounds[-5:])
        bw_tr = np.array([n for n, c in enumerate(bw) if c["split"] == "train" and c["id"] not in bw_val_r and bw_ok[n]])
        bw_va = np.array([n for n, c in enumerate(bw) if c["id"] in bw_val_r and bw_ok[n]]); bw_te = np.array([n for n, c in enumerate(bw) if c["split"] == "test" and bw_ok[n]])
        net.head_b = torch.nn.Linear(net.config.hidden_size, 2).to(dev); BW = True
        wb = torch.tensor([1.0, float(((bw_y[bw_tr] == 0).sum() / max((bw_y[bw_tr] == 1).sum(), 1)) ** 0.5)], device=dev)
        print(f"BoxingWeb clips: {len(bw)} punches, contact found for {bw_ok.mean():.1%} (score >= {a.bw_min}); train {len(bw_tr)}, validation {len(bw_va)}, test {len(bw_te)}; effective {bw_y[bw_tr].mean():.1%}; "
              f"chosen offsets (frames) 10/50/90 pct {np.percentile(bw_off[bw_ok], [10, 50, 90]).tolist()}", flush=True)
    NEG = None
    if a.neg:                                                 # third task on the same features: is this counted punch a real punch at all
        EXn = json.load(open(a.chain_json)); Bn = local(a.chain_npy, "chain.npy"); ng_real = np.array([bool(c["real"]) for c in EXn]); ng_bout = np.array([c["id"] for c in EXn]); ng_gym = np.array([c["gym"] for c in EXn])
        ng_tr = np.flatnonzero(~np.isin(ng_bout, list(val_bouts))) if a.all else np.flatnonzero((ng_gym != a.held) & ~np.isin(ng_bout, list(val_bouts)))
        ng_pos, ng_neg = ng_tr[ng_real[ng_tr]], ng_tr[~ng_real[ng_tr]]; net.head_n = torch.nn.Linear(net.config.hidden_size, 2).to(dev); NEG = True
        print(f"chain clips for the real-punch head: train {len(ng_tr)} ({len(ng_pos)} real, {len(ng_neg)} not)", flush=True)
    HN = ("classifier", "fc_norm", "head_b", "head_n")
    head = [p for n, p in net.named_parameters() if n.startswith(HN)]; body = [p for n, p in net.named_parameters() if not n.startswith(HN)]
    opt = torch.optim.AdamW([dict(params=body, lr=a.lr), dict(params=head, lr=1e-3)], weight_decay=0.05)
    nb = max(len(tr) // a.bs, 1); steps = a.epochs * nb; warm = nb
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    cnt = np.bincount(y[tr], minlength=3).astype(float); w3 = (len(tr) / (3 * np.maximum(cnt, 1))) ** 0.5
    w = torch.tensor(np.r_[w3, 1.0], dtype=torch.float32, device=dev)
    mean, std = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    def prep(x, train):                                       # x uint8 (B,16,H,W,3) on dev -> normalised (B,16,3,224,224)
        x = x.permute(0, 1, 4, 2, 3).float() / 255; B = x.shape[0]
        if train:
            if rng.random() < 0.5:
                x = x.flip(-1)
            s = rng.uniform(0.75, 1.0); side = int(224 * s); ox, oy = int(rng.integers(0, 224 - side + 1)), int(rng.integers(0, 224 - side + 1))
            x = F.interpolate(x[..., oy:oy + side, ox:ox + side].reshape(B * 16, 3, side, side), size=224, mode="bilinear", align_corners=False).reshape(B, 16, 3, 224, 224)
            x = ((x - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.1, 0.1)).clamp(0, 1)
        return (x - mean) / std
    def take(src, idx, offs):                                 # clips idx (sorted), each at its own offset
        return torch.from_numpy(np.stack([src[i, C0 - 8 * ST + d:C0 + 8 * ST + d:ST] for i, d in zip(idx, offs)])).to(dev)
    def feats(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            seq = net.videomae(x)[0]; return net.fc_norm(seq.mean(1)) if net.fc_norm is not None else seq[:, 0]
    def fwd(x):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return net.classifier(feats(x)).float()
    @torch.no_grad()
    def predict_b(idx):                                       # BoxingWeb clips at their found contact -> P(effective)
        net.eval(); P = []
        for k in range(0, len(idx), 24):
            ii = idx[k:k + 24]; x = prep(take(Bw, ii, bw_off[ii]), False)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                P.append(torch.softmax(net.head_b(feats(x)).float(), 1)[:, 1].cpu().numpy())
        return np.concatenate(P)
    @torch.no_grad()
    def predict(src, idx, off):                               # -> probabilities (len(idx), 4), rows follow idx (must be sorted)
        net.eval(); P = []
        for k in range(0, len(idx), 24):
            ii = idx[k:k + 24]; P.append(torch.softmax(fwd(prep(take(src, ii, [off] * len(ii)), False)), 1).cpu().numpy())
        return np.concatenate(P)
    def evaluate(idx):
        P0 = predict(A, idx, 0); yy = y[idx]; p3 = P0[:, :3] / P0[:, :3].sum(1, keepdims=True)
        acc = float((p3.argmax(1) == yy).mean()); au = auc((yy == 0).astype(int), p3[:, 0]); cen = float((P0[:, 3] < 0.5).mean())
        off = float(np.mean([(predict(A, idx, d)[:, 3] >= 0.5).mean() for d in (-SHIFT, SHIFT)]))
        return acc, au, cen, off, p3
    best = (-1, None); hist = []; best_state = None; sub_va = va[:: max(len(va) // 240, 1)]
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr); tl = 0.0
        for k in range(nb):
            ii = np.sort(perm[k * a.bs:(k + 1) * a.bs]); cen = rng.random(len(ii)) < 0.5
            offs = np.where(cen, rng.integers(-1, 2, len(ii)), rng.choice([-1, 1], len(ii)) * rng.integers(4, OFFMAX + 1, len(ii)))
            yy = torch.from_numpy(np.where(cen, y[ii], 3)).to(dev)
            loss = F.cross_entropy(fwd(prep(take(A, ii, offs), True)), yy, weight=w, label_smoothing=0.1)
            if BW:                                                # half as many BoxingWeb punches per step, centred on their found contact (+-1 frame)
                jj = np.sort(rng.choice(bw_tr, max(a.bs // 2, 1), replace=False)); ob = bw_off[jj] + rng.integers(-1, 2, len(jj))
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    lb = net.head_b(feats(prep(take(Bw, jj, ob), True))).float()
                loss = loss + 0.5 * F.cross_entropy(lb, torch.from_numpy(bw_y[jj]).to(dev), weight=wb, label_smoothing=0.1)
            if NEG:                                               # half a batch of the chain's own counted punches, real and not in equal numbers, anywhere within +-6 frames of the rough contact
                h = max(a.bs // 4, 1); jj = np.concatenate([rng.choice(ng_pos, h, replace=False), rng.choice(ng_neg, h, replace=False)]); o_ = np.argsort(jj); jj = jj[o_]; on = rng.integers(-6, 7, len(jj))
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    ln = net.head_n(feats(prep(take(Bn, jj, on), True))).float()
                loss = loss + 0.5 * F.cross_entropy(ln, torch.from_numpy(ng_real[jj].astype(np.int64)).to(dev), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        acc, au, cen, off, _ = evaluate(sub_va); score = acc + (cen + off) / 2
        hist.append(dict(epoch=ep + 1, loss=tl / nb, val_acc=acc, val_auc=au, val_centred_ok=cen, val_off_ok=off))
        print(f"epoch {ep + 1}/{a.epochs} loss {tl / nb:.3f} | validation: 3-way {acc:.3f}, landed-AUC {au:.3f}, centred recognised {cen:.3f}, shifted (8 frames) recognised {off:.3f} | {time.time() - t0:.0f}s", flush=True)
        if score > best[0]:
            best = (score, ep + 1); best_state = {k2: v.detach().cpu().clone() for k2, v in net.state_dict().items()}
    net.load_state_dict(best_state); print(f"best validation epoch {best[1]}", flush=True)
    acc, au, cen, off, P3 = evaluate(te); print(f"A. TEST centred: 3-way {acc:.3f}, landed-AUC {au:.3f}, centred recognised {cen:.3f}, shifted recognised {off:.3f}", flush=True)
    sub = te[:: max(len(te) // 800, 1)]; S = np.stack([predict(A, sub, d) for d in SCAN], 1)                 # (n, 13, 4)
    dh = np.array(SCAN)[S[:, :, 3].argmin(1)]
    print(f"B. TEST scan ({len(sub)} clips): pixel contact within +-2 frames {np.mean(np.abs(dh) <= 2):.1%}, +-4 {np.mean(np.abs(dh) <= 4):.1%}, median error {np.median(np.abs(dh)):.1f}", flush=True)
    extra = {}
    if BW:
        def f1(yv, pv, t):
            yh = (pv >= t).astype(int); o = []
            for c in (0, 1):
                tp = ((yv == c) & (yh == c)).sum(); pr = tp / max((yh == c).sum(), 1); rc = tp / max((yv == c).sum(), 1); o.append(2 * pr * rc / max(pr + rc, 1e-9))
            return float(np.mean(o))
        pv = predict_b(bw_va); thr = max(np.arange(0.1, 0.91, 0.05), key=lambda t: f1(bw_y[bw_va], pv, t)); pt_ = predict_b(bw_te)
        print(f"BoxingWeb TEST 'effective' ({len(bw_te)} punches, {bw_y[bw_te].mean():.0%} effective): macro-F1 {f1(bw_y[bw_te], pt_, thr):.3f} at threshold {thr:.2f}, AUC {auc(bw_y[bw_te], pt_):.3f}"
              f"   (BoxMind: 0.621 pose-only, 0.645 best)", flush=True)
        extra.update(bw_te=bw_te, bw_p=pt_)
    if a.chain_json:
        EX = json.load(open(a.chain_json)); Bc = local(a.chain_npy, "chain.npy"); eidx = np.array([n for n, c in enumerate(EX) if c["gym"] == a.held])
        if a.tiny:
            eidx = eidx[:6]
        if a.tiny and BW:
            pass
        CS = np.stack([predict(Bc, eidx, d) for d in SCAN], 1); extra.update(chain_idx=eidx, chain_scan=CS.astype(np.float32)); print(f"C. scanned {len(eidx)} chain punches", flush=True)
        if NEG:
            @torch.no_grad()
            def predict_n(idx, off):
                net.eval(); P = []
                for k in range(0, len(idx), 24):
                    ii = idx[k:k + 24]
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        P.append(torch.softmax(net.head_n(feats(prep(take(Bc, ii, [off] * len(ii)), False))).float(), 1)[:, 1].cpu().numpy())
                return np.concatenate(P)
            CR = np.stack([predict_n(eidx, d) for d in SCAN], 1); rr = np.array([bool(EX[n]["real"]) for n in eidx]).astype(int); extra.update(chain_real=CR.astype(np.float32))
            print(f"D. real punch vs not on the held-out chain punches ({rr.mean():.0%} real): AUC mean over the scan {auc(rr, CR.mean(1)):.3f}, at the rough contact {auc(rr, CR[:, len(SCAN) // 2]):.3f}, max {auc(rr, CR.max(1)):.3f}", flush=True)
    if a.save_model and not a.tiny:
        os.makedirs(os.path.dirname(a.save_model), exist_ok=True); torch.save(best_state, a.save_model + ".part"); os.replace(a.save_model + ".part", a.save_model)
    np.savez_compressed(a.out + ".tmp.npz", idx=te, P3=P3, scan_idx=sub, scan=S.astype(np.float32), scan_offsets=np.array(SCAN), best_epoch=best[1], hist=json.dumps(hist), held=a.held, **extra)
    os.replace(a.out + ".tmp.npz", a.out); print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
