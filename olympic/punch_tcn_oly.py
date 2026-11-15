"""Punch detector trained on BoxingWeb PLUS the Olympic referee labels (one gym held out), tested on the other gym.

The Olympic labels say when a punch connects and with which hand, but not its type and not its full span. So:
  - puncher = the chain stream whose head (torso for body shots) is FARTHER from the referee's box, which sits on the target;
  - span = the BoxingWeb detector's own event in that stream around the label, else a fixed span around the contact
    (median offsets of the matched events);
  - Olympic frames are trained with a 3-way target (no punch / left / right): the loss sums the type probabilities of
    each hand. Frames inside detector events that match no referee label count as "no punch" (the set is fully labelled).
  - BoxingWeb frames keep the 7-way target.
Scoring on the held-out gym exactly as olympic/fullchain.py (label middle inside a counted span grown by 0.2 s,
one-to-one, either boxer). Threshold chosen on 3 bouts of the TRAINING gym; the baseline gets the same choice.
    python olympic/punch_tcn_oly.py [--mode joint|oly|none] [--seeds 0 1 2] [--save]
"""
import argparse, glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "boxingweb"))
import mem_guard  # noqa: F401
import numpy as np, torch, torch.nn as nn
from progress import progress
import punch_tcn as pt
L, TOL, VALN = 512, 10, 3

def base_models():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn", "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); out.append(m)
    return out

def puncher_stream(K, i, c, outcome):
    """The referee's box sits on the TARGET: the puncher is the boxer whose head (torso for body shots) is FARTHER from it.
    (The earlier rule, nearest labelled-hand glove, picked the defender for about a third of the labels.) -1 if unknown."""
    part = [5, 6, 11, 12] if outcome == "body" else [0, 1, 2, 3, 4]; d = [np.linalg.norm(np.nanmean(K[s, i, part], 0) - c) for s in (0, 1)]
    return int(np.argmax(d)) if np.isfinite(d).all() else -1

def load(cid, ev, models):
    z = np.load(os.path.join(ROOT, "external", "chain_streams", cid + ".npz")); X, M, K, fr = z["X"], z["M"], z["K"].astype(np.float64), z["frame"]; F = len(fr); step = int(fr[1] - fr[0])
    P = np.mean([pt.predict(m, X[:2]) for m in models], 0); evs = [pt.events(P[b], M[b], 0.40) for b in (0, 1)]
    labs = []
    for e in [e for e in ev if "O_" + e["video"] == cid and e["end"] <= fr[-1]]:
        mid = (e["start"] + e["end"]) / 2; ii = int(np.clip(round(mid / step), 0, F - 1)); w = 9 if e["hand"] == "left" else 10
        b = np.array(e["boxes"]); c = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
        labs.append(dict(mid=mid, i=ii, hand=0 if e["hand"] == "left" else 1, stream=puncher_stream(K, ii, c, e["outcome"]), outcome=e["outcome"], raw=e))
    return dict(id=cid, X=X, M=M, K=K, frame=fr, step=step, labs=labs, evs=evs, P=P)

def targets(D):
    """3-way frame targets per stream (0 none, 1 left, 2 right; -100 ignored). Returns the median span offsets used."""
    offs = []
    for d in D:
        for l in d["labs"]:
            if l["stream"] < 0:
                continue
            c = [(a, e) for a, e, _ in d["evs"][l["stream"]] if a - 5 <= l["i"] <= e + 5]; l["span"] = min(c, key=lambda s: abs((s[0] + s[1]) / 2 - l["i"])) if c else None
            if l["span"]:
                offs.append((l["span"][0] - l["i"], l["span"][1] - l["i"]))
    o = np.median(np.array(offs), 0).round().astype(int)
    for d in D:
        F = len(d["frame"]); Y = np.zeros((4, F), np.int64)
        for l in d["labs"]:
            if l["stream"] < 0:
                continue
            a, e = l["span"] if l["span"] else (l["i"] + o[0], l["i"] + o[1]); a, e = max(a, 0), min(e, F - 1); s = l["stream"]
            Y[s, a:e + 1] = 1 + l["hand"]; Y[s + 2, a:e + 1] = 2 - l["hand"]                       # mirrored stream: the other hand
        Mm = np.concatenate([d["M"], d["M"]]); d["Y3"] = np.where(Mm, Y, -100); d["M4"] = Mm
    return o

def match(labels, counts, shift=0, span=None):
    used = set(); pairs = []
    for li, e in enumerate(labels):
        m = (e["start"] + e["end"]) // 2 + shift
        if span:
            m = m % span
        for ci, c in enumerate(counts):
            if ci not in used and c["f0"] - TOL <= m <= c["f1"] + TOL:
                used.add(ci); pairs.append((li, ci)); break
    return pairs

def score(D, Ps, thr):
    A = dict(lab=0, cnt=0, found=0, hand=0, boxer=0, known=0, chance=0)
    for d, P in zip(D, Ps):
        fr = d["frame"]; counts = []
        for b in (0, 1):
            for a, e, c in pt.events(P[b], d["M"][b], thr):
                counts.append(dict(f0=int(fr[a]), f1=int(fr[e]), boxer=b, hand=0 if c % 2 == 1 else 1))
        counts.sort(key=lambda c: c["f0"]); labels = [l["raw"] for l in d["labs"]]; pairs = match(labels, counts)
        A["lab"] += len(labels); A["cnt"] += len(counts); A["found"] += len(pairs); A["chance"] += len(match(labels, counts, 350, int(fr[-1])))
        for li, ci in pairs:
            l = d["labs"][li]; A["hand"] += l["hand"] == counts[ci]["hand"]
            if l["stream"] >= 0:
                A["known"] += 1; A["boxer"] += l["stream"] == counts[ci]["boxer"]
    r, p = A["found"] / max(A["lab"], 1), A["found"] / max(A["cnt"], 1)
    return dict(found=r, real=p, f1=2 * r * p / max(r + p, 1e-9), hand=A["hand"] / max(A["found"], 1), boxer=A["boxer"] / max(A["known"], 1), chance=A["chance"] / max(A["lab"], 1), counted=A["cnt"], labels=A["lab"])

def train(bw, oly, seed, mode, epochs=40):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); model = pt.TCN(157).to(pt.DEV); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-2)
    nb = sum(d["X"].shape[1] * 4 for d in bw) if mode != "oly" else 0; no = sum(d["X"].shape[1] * 4 for d in oly); steps = epochs * (nb + no) // (L * 32)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps); w7 = torch.tensor([1.0] + [2.0] * 6, device=pt.DEV); ce = nn.CrossEntropyLoss(weight=w7, ignore_index=-100)
    w3 = torch.tensor([1.0, 2.0, 2.0], device=pt.DEV); nll = nn.NLLLoss(weight=w3, ignore_index=-100); po = no / (nb + no); progress.start(f"punch-tcn-oly s{seed}", total=steps)
    for st in range(steps):
        model.train(); xb, yb, ob = [], [], []
        for _ in range(32):
            o = rng.random() < po; src = oly if o else bw; k = rng.integers(len(src)); s = rng.integers(4); F = src[k]["X"].shape[1]; i = rng.integers(0, F - L)
            xb.append(src[k]["X"][s, i:i + L]); yb.append(src[k]["Y3"][s, i:i + L] if o else src[k]["Yb"][s, i:i + L]); ob.append(o)
        x = torch.from_numpy(np.stack(xb)).to(pt.DEV); y = torch.from_numpy(np.stack(yb)).to(pt.DEV); o = torch.tensor(ob, device=pt.DEV); lg = model(x); loss = 0.0
        if (~o).any():
            loss = loss + ce(lg[~o].reshape(-1, 7), y[~o].reshape(-1)) * (~o).float().mean()
        if o.any():
            p = torch.softmax(lg[o], -1); p3 = torch.stack([p[..., 0], p[..., [1, 3, 5]].sum(-1), p[..., [2, 4, 6]].sum(-1)], -1).clamp_min(1e-7).log()
            loss = loss + nll(p3.reshape(-1, 3), y[o].reshape(-1)) * o.float().mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if st % 20 == 0:
            progress.tick(20, note=f"step {st}/{steps} loss {loss.item():.3f}")
    progress.finish(); return model

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="joint"); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--save", action="store_true"); a = ap.parse_args()
    ev = json.load(open(os.path.join(HERE, "events.json"))); base = base_models()
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "chain_streams", "O_*.npz"))); D = {c: load(c, ev, base) for c in ids}
    bwi = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "punch_ds", pre + "*.npz"))); tr_ids = bwi("W_train")[:-pt.VAL_N]
    bw = [dict(X=z["X"], Yb=np.where(z["M"], z["Y"], -100)) for z in pt.load(tr_ids)]; bwte = pt.load(bwi("W_test")); res = {}; tot = {}
    for held in ("kam2", "kam4"):
        trn = [D[c] for c in ids if held not in c]; val, trn = trn[-VALN:], trn[:-VALN]; te = [D[c] for c in ids if held in c]; o = targets(trn)
        print(f"\nheld-out {held}: train {len(trn)} bouts ({sum(len(d['labs']) for d in trn)} labels, {np.mean([l['span'] is not None for d in trn for l in d['labs'] if l['stream'] >= 0]):.0%} take the detector's span; "
              f"others {o[0]:+d}..{o[1]:+d} frames around the contact), threshold on {len(val)} bouts, test {len(te)} bouts", flush=True)
        thrs = np.arange(0.2, 0.81, 0.05); pick = lambda Pv: max(thrs, key=lambda t: score(val, Pv, t)["f1"])
        tb = pick([d["P"] for d in val]); rows = [("BoxingWeb-only detector, threshold 0.40", score(te, [d["P"] for d in te], 0.40), None), (f"BoxingWeb-only, threshold {tb:.2f} from the training gym", score(te, [d["P"] for d in te], tb), None)]
        if a.mode != "none":
            models = [train(bw, trn, s, a.mode) for s in a.seeds]; pred = lambda dd: [np.mean([pt.predict(m, d["X"][:2]) for m in models], 0) for d in dd]
            t = pick(pred(val)); Pte = pred(te); bwP = [np.mean([pt.predict(m, z["X"][:2]) for m in models], 0) for z in bwte]; bwB = [np.mean([pt.predict(m, z["X"][:2]) for m in base], 0) for z in bwte]
            rows.append((f"{a.mode}: + Olympic labels of the other gym, threshold {t:.2f}", score(te, Pte, t), pt.score(bwP, bwte, 0.35)))
            print(f"  BoxingWeb test (threshold 0.35): BoxingWeb-only F1 {pt.score(bwB, bwte, 0.35)['f1']:.3f}, type right {pt.score(bwB, bwte, 0.35)['type']:.1%} | new F1 {rows[-1][2]['f1']:.3f}, type right {rows[-1][2]['type']:.1%}")
            if a.save:
                out = os.path.join(ROOT, "runs", "punch-tcn-oly"); os.makedirs(out, exist_ok=True)
                for s, m in zip(a.seeds, models):
                    torch.save(m.state_dict(), os.path.join(out, f"model_heldout_{held}_seed{s}.pt"))
                np.savez_compressed(os.path.join(out, f"pred_{held}.npz"), thr=t, **{d["id"]: p for d, p in zip(te, Pte)})
        for name, s, _ in rows:
            print(f"  {name:62s} counted {s['counted']:5d} | found {s['found']:.1%} (chance {s['chance']:.1%}) real {s['real']:.1%} F1 {s['f1']:.3f} | hand right {s['hand']:.1%} | right boxer {s['boxer']:.1%}", flush=True)
            k = name.split(",")[0].split(":")[0]; T = tot.setdefault(name.split(" threshold")[0].rstrip(","), []); T.append(s)
        res[held] = [(n, s) for n, s, _ in rows]
    print("\nBOTH folds pooled:")
    for name, ss in tot.items():
        lab = sum(s["labels"] for s in ss); cnt = sum(s["counted"] for s in ss); f = sum(s["found"] * s["labels"] for s in ss); h = sum(s["hand"] * s["found"] * s["labels"] for s in ss); r, p = f / lab, f / cnt
        print(f"  {name:55s} found {r:.1%} real {p:.1%} F1 {2 * r * p / (r + p):.3f} | hand right {h / f:.1%}")
    json.dump(res, open(os.path.join(ROOT, "runs", f"olympic_punch_tcn_oly_{a.mode}.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    main()
