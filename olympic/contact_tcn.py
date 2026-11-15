"""CONTACT-MOMENT model: for each boxer stream, per frame, "a punch of this boxer connects NOW, with the left / right
hand". Trained on the referee's contact moments (Olympic set, 4,605 punches), which say when and which hand but not
which boxer: the referee's box sits on the target, so the puncher is the stream whose head is FARTHER from the box
(until 2026-11-15 the nearest labelled-hand glove was used, which picked the defender for about a third of labels).
Input: the chain's own skeleton streams (external/chain_streams, 25 fps). Target: a narrow bump (sigma 1 frame) around
the contact frame in the channel of the punching hand. Same network as the punch detector (dilated 1D convolutions).
Train on one gym, test on the other.
Test = the chain's own counted punches (hpc_rtm/lists/chain_clips.json): inside each counted punch (grown by 4 frames)
take the frame and hand with the highest contact score.
  timing : our contact frame vs the referee's (50 fps video frames)        before: +-2 frames 31%, +-5 frames 52%
  hand   : agrees with the referee                                          before: 79.7%
Writes hpc_rtm/lists/chain_clips_v2.json (same punches, new contact frame, hand and glove crop).
    python olympic/contact_tcn.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "boxingweb"))
import mem_guard  # noqa: F401
import numpy as np, torch, torch.nn as nn
from progress import progress
import punch_tcn as pt, punch_tcn_oly as po
L, SIGMA, GROW = 512, 1.0, 4

def load(cid, ev):
    z = np.load(os.path.join(ROOT, "external", "chain_streams", cid + ".npz")); X, M, K, fr = z["X"], z["M"], z["K"].astype(np.float64), z["frame"]; F = len(fr); step = int(fr[1] - fr[0])
    Y = np.zeros((4, F, 2), np.float32); t = np.arange(F); labs = []
    for e in [e for e in ev if "O_" + e["video"] == cid and e["end"] <= fr[-1]]:
        mid = (e["start"] + e["end"]) / 2; i = mid / step; ii = int(np.clip(round(i), 0, F - 1)); w = 9 if e["hand"] == "left" else 10
        b = np.array(e["boxes"]); c = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
        s = po.puncher_stream(K, ii, c, e["outcome"])                # the box sits on the target: puncher = the boxer whose head is farther from it
        if s < 0 or not np.isfinite(K[s, ii, w, 0]):
            continue
        h = 0 if e["hand"] == "left" else 1; g = np.exp(-(t - i) ** 2 / (2 * SIGMA ** 2)).astype(np.float32)
        Y[s, :, h] = np.maximum(Y[s, :, h], g); Y[s + 2, :, 1 - h] = np.maximum(Y[s + 2, :, 1 - h], g)       # mirrored stream: the other hand
        labs.append(dict(mid=mid, hand=e["hand"], outcome=e["outcome"], stream=s))
    return dict(id=cid, X=X, M=M, K=K, frame=fr, step=step, Y=Y, labs=labs)

def train(data, seed=0, epochs=40):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = pt.TCN(157, ncls=2).to(pt.DEV)
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2); steps = epochs * sum(d["X"].shape[1] * 4 for d in data) // (L * 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps); progress.start("contact-tcn", total=steps)
    Ms = [np.concatenate([d["M"], d["M"]]) for d in data]
    for st in range(steps):
        model.train(); xb, yb, mb = [], [], []
        for _ in range(32):
            k = rng.integers(len(data)); s = rng.integers(4); F = data[k]["X"].shape[1]; i = rng.integers(0, F - L)
            xb.append(data[k]["X"][s, i:i + L]); yb.append(data[k]["Y"][s, i:i + L]); mb.append(Ms[k][s, i:i + L])
        x, y, m = (torch.from_numpy(np.stack(v)).to(pt.DEV) for v in (xb, yb, mb))
        lg = model(x); p = torch.sigmoid(lg); bce = nn.functional.binary_cross_entropy_with_logits(lg, y, reduction="none")
        wgt = torch.where(y > 0.05, 8.0 * torch.ones_like(y), (p.detach() ** 2) * 2 + 0.25)                    # up-weight contact frames, focus on confident false alarms
        loss = (bce * wgt * m.float().unsqueeze(-1)).sum() / m.float().sum().clamp(min=1) / 2
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if st % 20 == 0:
            progress.tick(20, note=f"step {st}/{steps} loss {loss.item():.4f}")
    progress.finish(); return model

@torch.no_grad()
def heat(model, d):
    model.eval(); return torch.sigmoid(model(torch.from_numpy(d["X"][:2]).to(pt.DEV))).cpu().numpy() * d["M"][:, :, None]       # (2, F, 2)

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips.json")))
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "chain_streams", "O_*.npz")))
    D = {c: load(c, ev) for c in ids}; print(f"{len(ids)} bouts, {sum(len(d['labs']) for d in D.values())} contact labels placed on a boxer stream", flush=True)
    new = [dict(c) for c in clips]; rows = []
    for held in ("kam2", "kam4"):
        models = [train([D[c] for c in ids if held not in c], seed=s) for s in (0, 1)]
        for cid in [c for c in ids if held in c]:
            d = D[cid]; H = np.mean([heat(m, d) for m in models], 0); F = len(d["frame"]); step = d["step"]
            for n, c in enumerate(clips):
                if c["id"] != cid:
                    continue
                b = "AB".index(c["boxer"]); a, e = max(c["f0"] // step - GROW, 0), min(c["f1"] // step + GROW, F - 1); win = H[b, a:e + 1]            # (T, 2)
                t, h = np.unravel_index(win.argmax(), win.shape); t += a
                off = 0.0
                if 0 < t < F - 1:                                                                             # sub-frame peak (parabola through 3 points)
                    y0, y1, y2 = H[b, t - 1, h], H[b, t, h], H[b, t + 1, h]; den = y0 - 2 * y1 + y2; off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if abs(den) > 1e-6 else 0.0
                cf = int(round((t + off) * step)); w = 9 if h == 0 else 10; K = d["K"][b, t]
                g = K[w] + 0.4 * (K[w] - K[w - 2]); S = max(np.linalg.norm(K[[5, 6]].mean(0) - K[[11, 12]].mean(0)), 20.0); ok = np.isfinite(g).all()
                new[n].update(frame=cf, hand="left" if h == 0 else "right", cx=float(g[0]) if ok else c["cx"], cy=float(g[1]) if ok else c["cy"], side=float(2.4 * S) if ok else c["side"], contact_score=float(win.max()))
                if c["label"] is not None:
                    cand = [l for l in d["labs"] if c["f0"] - 10 <= l["mid"] <= c["f1"] + 10 and l["outcome"] == c["label"] and l["hand"] == c["label_hand"]]
                    if cand:
                        l = min(cand, key=lambda l: abs(l["mid"] - (c["f0"] + c["f1"]) / 2))
                        rows.append((c["frame"] - l["mid"], cf - l["mid"], c["hand"] == l["hand"], new[n]["hand"] == l["hand"]))
    r = np.array(rows, float); print(f"\n{len(r)} counted punches matched to a referee label (held-out gym for each):")
    for name, dt, hd in (("before (greatest arm extension, detector's hand)", r[:, 0], r[:, 2]), ("contact model", r[:, 1], r[:, 3])):
        print(f"  {name:50s} timing within +-2 frames {np.mean(np.abs(dt) <= 2):.1%}, +-5 frames {np.mean(np.abs(dt) <= 5):.1%}, median error {np.median(np.abs(dt)):.1f} frames | hand right {hd.mean():.1%} | "
              f"timing +-2 AND hand right {np.mean((np.abs(dt) <= 2) & (hd > 0)):.1%}, +-5 AND hand {np.mean((np.abs(dt) <= 5) & (hd > 0)):.1%}")
    json.dump(new, open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v2.json"), "w"))
    json.dump(dict(n=int(len(r)), before=dict(pm2=float(np.mean(np.abs(r[:, 0]) <= 2)), pm5=float(np.mean(np.abs(r[:, 0]) <= 5)), hand=float(r[:, 2].mean())),
                   after=dict(pm2=float(np.mean(np.abs(r[:, 1]) <= 2)), pm5=float(np.mean(np.abs(r[:, 1]) <= 5)), hand=float(r[:, 3].mean()))), open(os.path.join(ROOT, "runs", "olympic_contact_tcn.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
