"""The NEW chain on the Olympic bouts, end to end, scored exactly like this morning's baseline (score_counter.py):
  RTMPose -> learned selector (pair rule) -> identity (OSNet-AIN + self-updating classifier, carried through overlaps)
  -> per-boxer streams -> punch TCN trained on BoxingWeb (3-seed ensemble, threshold fixed from BoxingWeb validation).
Nothing here was trained on the Olympic bouts except nothing: selector, identity and detector all come from elsewhere.
Scoring: a referee label is FOUND by a counted punch whose span (grown by 0.2 s) contains the label's middle frame,
one-to-one, either boxer; chance = labels shifted by 7 s. Baseline (old pipeline): found 24.8% (chance 15.8%),
real 47.7%, hand right 60.6%.
Also writes hpc_rtm/lists/chain_clips.json: one contact clip per counted punch (contact = the frame of greatest
extension of the punching arm; crop centred on that glove) for the landed model.
    python olympic/fullchain.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "boxingweb"))
import mem_guard  # noqa: F401
import numpy as np, torch
import identity_episodes as ie, identity_reid2 as ir2, punch_data as pd, punch_fullchain as pf, punch_tcn as pt
TOL, SHIFT, THR = 10, 350, 0.40

def prepare(cid):
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); pick = np.load(os.path.join(ROOT, "external", "boxers_rtm", cid + ".npz"))["pick"]
    pseudo = np.zeros(pick.shape[:1] + (R["kp"].shape[1],), np.int8)
    for i, pk in enumerate(pick):
        for p in pk:
            if p >= 0:
                pseudo[i, p] = 1
    rd = dict(id=cid, segs=ie.segments(R, pseudo, None))
    return dict(id=cid, segs=rd["segs"], scrops=ir2.crops_for_round(rd), pick=pick, cut=np.zeros(len(pick), bool))

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

def main():
    from multiprocessing import Pool
    ev = json.load(open(os.path.join(HERE, "events.json")))
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "O_*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(6) as pool:
        prs = pool.map(prepare, ids)
    emb = ir2.make_models()["OSNet-AIN (re-identification)"]; models = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn", "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); models.append(m)
    A = dict(lab=0, cnt=0, found=0, chance=0, hand=0, sec=0.0); by = {}; clips = []; rows = []
    for pr in prs:
        ident, R = pf.identities(pr, emb); kp, sc, fr = R["kp"].astype(np.float64), R["score"], R["frame"]; F = len(fr)
        role = np.where(ident == 1, 1, np.where(ident == 0, 2, 0)).astype(np.int8); st = {r: pd.stream(kp, sc, role, r) for r in (1, 2)}
        X = np.stack([pd.feats(*st[r], st[3 - r][0], st[3 - r][2]) for r in (1, 2)]); M = np.stack([st[r][2] for r in (1, 2)])
        P = np.mean([pt.predict(m, X) for m in models], 0); counts = []
        for b, r in ((0, 1), (1, 2)):
            K = st[r][0]
            for a, e, c in pt.events(P[b], M[b], THR):
                w = 9 if c % 2 == 1 else 10; seg = K[a:e + 1]; ext = np.linalg.norm(seg[:, w] - seg[:, w - 4], axis=1)
                t = a + (int(np.nanargmax(ext)) if np.isfinite(ext).any() else (e - a) // 2); g = K[t, w] + 0.4 * (K[t, w] - K[t, w - 2])
                S = max(np.linalg.norm(K[t, [5, 6]].mean(0) - K[t, [11, 12]].mean(0)), 20.0)
                counts.append(dict(f0=int(fr[a]), f1=int(fr[e]), boxer="AB"[b], cls=int(c), hand="left" if c % 2 == 1 else "right", contact=int(fr[t]),
                                   cx=float(g[0]), cy=float(g[1]), side=float(2.4 * S)))
        counts.sort(key=lambda c: c["f0"]); labels = [e for e in ev if "O_" + e["video"] == pr["id"] and e["end"] <= fr[-1]]
        pairs = match(labels, counts); ch = len(match(labels, counts, SHIFT, int(fr[-1])))
        hand = sum(labels[l]["hand"] == counts[c]["hand"] for l, c in pairs); fl = {l for l, _ in pairs}; cm = {c: l for l, c in pairs}
        for i, e in enumerate(labels):
            o = by.setdefault(e["outcome"], [0, 0]); o[0] += 1; o[1] += i in fl
        for ci, c in enumerate(counts):
            ok = np.isfinite(c["cx"]) and np.isfinite(c["cy"])
            clips.append(dict(id=pr["id"], frame=c["contact"], cx=c["cx"] if ok else 0.0, cy=c["cy"] if ok else 0.0, side=c["side"], bx=0.0, by=0.0, gym=pr["id"][2:6],
                              y=-1, f0=c["f0"], f1=c["f1"], hand=c["hand"], cls=c["cls"], boxer=c["boxer"], label=(labels[cm[ci]]["outcome"] if ci in cm else None),
                              label_hand=(labels[cm[ci]]["hand"] if ci in cm else None), n_labels=len(labels)))
        sec = F / float(R["fps"])
        rows.append(f"{pr['id']:20s} labels {len(labels):3d} counted {len(counts):4d} | found {len(pairs) / max(len(labels), 1):4.0%} (chance {ch / max(len(labels), 1):4.0%}) real {len(pairs) / max(len(counts), 1):4.0%} hand {hand / max(len(pairs), 1):4.0%} | boxer skeleton present {M.mean():.0%}")
        for k, v in dict(lab=len(labels), cnt=len(counts), found=len(pairs), chance=ch, hand=hand, sec=sec).items():
            A[k] += v
    print("\n".join(rows)); r, p = A["found"] / A["lab"], A["found"] / A["cnt"]
    print(f"\nALL {len(prs)} bouts, {A['sec'] / 60:.0f} min: {A['lab']} labelled punches ({A['lab'] / A['sec'] * 60:.0f}/min), {A['cnt']} counted ({A['cnt'] / A['sec'] * 60:.0f}/min)")
    print(f"  found {r:.1%} (chance {A['chance'] / A['lab']:.1%}) | real {p:.1%} | F1 {2 * r * p / (r + p):.3f} | hand right {A['hand'] / max(A['found'], 1):.1%}")
    print(f"  OLD pipeline this morning: found 24.8% (chance 15.8%) | real 47.7% | F1 0.326 | hand right 60.6%")
    print("  found by outcome: " + ", ".join(f"{o} {b / a:.0%} of {a}" for o, (a, b) in sorted(by.items())))
    json.dump(dict(A, by_outcome=by, threshold=THR), open(os.path.join(ROOT, "runs", "olympic_fullchain.json"), "w"), indent=1, default=float)
    json.dump(clips, open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips.json"), "w"))
    print(f"  {len(clips)} contact clips written for the landed model")

if __name__ == "__main__":
    main()
