"""Punch TYPE behind our own detector on BoxingWeb data_test: does the pixel type model (hpc_rtm/type_finetune.py,
runs/type_model/type_large.pt) still help when the span and the hand come from the detector, not from the labels?
Per test round (truth-matched streams, as punch_eval.py): detector events, matched one-to-one to labels by overlap.
Per matched punch: skeleton family = detector class probabilities summed over the DETECTED span, restricted to the
DETECTED hand; pixel family = VideoMAE-L on 16 video frames (as in training: most rounds 30 fps, a few 24) around the middle of the detected span, crop centred
on the detected hand's glove there (side 2.4 body scales): the same recipe the pipeline can run on any video.
Fusion = mean of the two probabilities.   -> runs/type_chain.txt, runs/type_chain.npz
    python boxingweb/type_chain.py
"""
import glob, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mem_guard  # noqa: F401
import cv2, numpy as np, torch
from progress import progress
import punch_data as pd, punch_tcn as pt
THR = 0.40; MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def crop(im, cx, cy, side):
    h, w = im.shape[:2]; s = int(max(side, 64)); x1, y1 = int(cx - s / 2), int(cy - s / 2); out = np.zeros((s, s, 3), np.uint8)
    xa, ya, xb, yb = max(x1, 0), max(y1, 0), min(x1 + s, w), min(y1 + s, h)
    if xb > xa and yb > ya:
        out[ya - y1:yb - y1, xa - x1:xb - x1] = im[ya:yb, xa:xb]
    return cv2.cvtColor(cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

def type_model(dev):
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16, tubelet_size=2, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
                         use_mean_pooling=True, num_labels=3, attn_implementation="sdpa")
    net = VideoMAEForVideoClassification(cfg); sd = torch.load(os.path.join(ROOT, "runs", "type_model", "type_large.pt"), map_location="cpu", weights_only=True)
    sd = {k.replace(".q_bias", ".query.bias").replace(".v_bias", ".value.bias"): v for k, v in sd.items()}
    net.load_state_dict(sd, strict=True); return net.to(dev).eval()

def type_probs(net, A, dev, note="type"):
    """A: (N, 16, 224, 224, 3) uint8 RGB -> (N, 3) family probabilities (straight, hook, uppercut)."""
    mean, std = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1); out = []
    progress.start(note, total=len(A))
    with torch.no_grad():
        for k in range(0, len(A), 8):
            x = torch.from_numpy(np.ascontiguousarray(A[k:k + 8])).to(dev).permute(0, 1, 4, 2, 3).float() / 255
            out.append(torch.softmax(net(pixel_values=(x - mean) / std).logits.float(), 1).cpu().numpy()); progress.tick(len(x))
    progress.finish(); return np.concatenate(out) if out else np.zeros((0, 3), np.float32)

def main():
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "punch_ds", "W_test*.npz"))); models = []; t0 = time.time()
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn", "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); models.append(m)
    rows = []; need_all = {}                                         # rows: (round, boxer, y6, c, mid, cx, cy, side, sk3...)
    for cid in ids:
        z = np.load(os.path.join(ROOT, "external", "punch_ds", cid + ".npz")); P = np.mean([pt.predict(m, z["X"][:2]) for m in models], 0)
        R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); role = np.load(os.path.join(ROOT, "external", "boxers_gt", cid + ".npz"))["role"]
        kp, sc, fr = R["kp"].astype(np.float64), R["score"], R["frame"]
        for b in (0, 1):
            K = pd.stream(kp, sc, role, b + 1)[0]; ev = pt.events(P[b], z["M"][b], THR); lab = pt.labels_of(z["Y"][b]); used = set()
            for a, e, y6 in lab:
                best, bo = None, 0
                for k, (pa, pe, pc) in enumerate(ev):
                    if k in used:
                        continue
                    o = min(e + pt.TOL, pe) - max(a - pt.TOL, pa) + 1
                    if o > bo:
                        best, bo = k, o
                if best is None:
                    continue
                used.add(best); pa, pe, c = ev[best]; h = (c - 1) % 2; w = 9 + h; mid = (pa + pe) // 2; k = K[mid]; g = k[w] + 0.4 * (k[w] - k[w - 2])
                p = P[b, pa:pe + 1, 1:].sum(0); pf = p[[h, h + 2, h + 4]]; pf = pf / max(pf.sum(), 1e-9)
                ok = bool(np.isfinite(g).all() and np.isfinite(k[[5, 6, 11, 12]]).all()); S = max(np.linalg.norm(k[[5, 6]].mean(0) - k[[11, 12]].mean(0)), 20.0) if ok else 0.0
                rows.append(dict(id=cid, y6=int(y6), c=int(c), frame=int(fr[mid]), cx=float(g[0]) if ok else None, cy=float(g[1]) if ok else None, side=float(2.4 * S), sk=pf.tolist()))
    idx = [n for n, r in enumerate(rows) if r["cx"] is not None]; N = len(idx); print(f"{len(rows)} labelled test punches matched to a counted punch; {N} with a skeleton at the middle; detector {time.time() - t0:.0f}s", flush=True)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "type_chain_clips.npy"); A = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(N, 16, 224, 224, 3)); filled = np.zeros((N, 16), bool)
    progress.start("cut type clips", total=len(ids))
    for cid in ids:
        need = {}
        for k, n in enumerate(idx):
            if rows[n]["id"] == cid:
                for j in range(16):
                    need.setdefault(rows[n]["frame"] - 8 + j, []).append((k, j))
        cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); f = 0; last = max(need) if need else -1
        while f <= last:
            if not cap.grab():
                break
            if f in need:
                ok, im = cap.retrieve()
                if ok:
                    for k, j in need[f]:
                        r = rows[idx[k]]; A[k, j] = crop(im, r["cx"], r["cy"], r["side"]); filled[k, j] = True
            f += 1
        cap.release(); progress.tick(1, note=cid[-30:])
    progress.finish()
    for k in range(N):
        if not filled[k].all() and filled[k].any():
            good = np.flatnonzero(filled[k])
            for j in np.flatnonzero(~filled[k]):
                A[k, j] = A[k, good[np.abs(good - j).argmin()]]
    net = type_model(pt.DEV); t1 = time.time(); PX = type_probs(net, A, pt.DEV); ms = (time.time() - t1) / max(N, 1) * 1000; del A; os.remove(tmp)
    y6 = np.array([rows[n]["y6"] for n in idx]); c = np.array([rows[n]["c"] for n in idx]); SK = np.array([rows[n]["sk"] for n in idx]); h = (c - 1) % 2; fam = (y6 - 1) // 2; hok = h == (y6 - 1) % 2
    L = [f"BoxingWeb data_test, punch type behind our detector: {N} labelled punches matched to a counted punch (hand right {hok.mean():.1%}); pixel type model {ms:.0f} ms per punch on {pt.DEV}", ""]
    def rep(name, P3):
        f = P3.argmax(1); six = 1 + h + 2 * f; rec = [np.mean(f[hok & (fam == k)] == k) for k in range(3)]
        L.append(f"  {name:<28s} all 6 classes right {np.mean(six == y6):.1%} | family right (hand right) {np.mean(f[hok] == fam[hok]):.1%}, macro {np.mean(rec):.1%} "
                 f"(straight {rec[0]:.0%}, hook {rec[1]:.0%}, uppercut {rec[2]:.0%})")
    L.append(f"  {'detector class as counted':<28s} all 6 classes right {np.mean(c == y6):.1%}")
    rep("skeleton (span, given hand)", SK); rep("pixels", PX); rep("mean of both", 0.5 * SK + 0.5 * PX); rep("geometric mean", np.sqrt((SK + 1e-6) * (PX + 1e-6)))
    s = "\n".join(L); print(s); open(os.path.join(ROOT, "runs", "type_chain.txt"), "w").write(s + "\n")
    np.savez_compressed(os.path.join(ROOT, "runs", "type_chain.npz"), y6=y6, c=c, SK=SK, PX=PX)

if __name__ == "__main__":
    main()
