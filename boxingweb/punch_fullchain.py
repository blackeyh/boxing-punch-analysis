"""The FULL CHAIN on BoxingWeb data_test, no truth handed over:
  RTMPose people -> learned selector (2 boxers) -> identity (OSNet-AIN, stretch 0 = A and B, self-updating A-vs-B
  classifier, carried by motion through overlaps) -> one skeleton stream per boxer -> punch TCN (3-seed ensemble,
  trained on truth-matched streams) -> counted punches per boxer, with type.
Truth is used only to score: A/B = red/blue or blue/red (one choice per round), then per boxer, one-to-one, a counted
punch matches a labelled punch of THAT boxer if they overlap in time (label grown by 3 frames).
Also reported: counted punches that match only a label of the OTHER boxer (credited to the wrong fighter).
Reference: the same detector on truth-matched streams (punch_eval.py).
    python boxingweb/punch_fullchain.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "olympic"))
import mem_guard  # noqa: F401
import numpy as np, torch
import identity_pipeline as ip, identity_reid2 as ir2, identity_swaps as isw, punch_data as pd, punch_tcn as pt

def identities(pr, emb):
    """-> (F, K) 1 = boxer A, 0 = boxer B, -1 = nobody; stretch labels by recognition, overlaps carried forward."""
    from sklearn.linear_model import LogisticRegression
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", pr["id"] + ".npz")); kp, sc = R["kp"].astype(np.float64), R["score"]; F = len(R["frame"])
    segs, pick = pr["segs"], pr["pick"]; E = emb(pr["scrops"], False); per = {}
    for (si, t, *_), e in zip(pr["scrops"], E):
        per.setdefault((si, t), []).append(e)
    per = {k: np.array(v) for k, v in per.items()}
    lab = np.full((F, kp.shape[1]), -1, np.int8); in_seg = np.zeros(F, bool)
    if len(segs) < 3 or (0, 0) not in per or (0, 1) not in per:
        return lab, R
    isA0 = {0: True}; Xs, ys = [per[(0, 0)], per[(0, 1)]], [np.ones(len(per[(0, 0)])), np.zeros(len(per[(0, 1)]))]; clf = None
    gA, gB = per[(0, 0)].mean(0), per[(0, 1)].mean(0)
    for si in range(1, len(segs)):
        if (si, 0) not in per or (si, 1) not in per:
            isA0[si] = isA0[si - 1]; continue
        a, b = per[(si, 0)], per[(si, 1)]
        d = ((a.mean(0) @ gA + b.mean(0) @ gB) - (a.mean(0) @ gB + b.mean(0) @ gA)) if clf is None else (clf.decision_function(a).mean() - clf.decision_function(b).mean())
        isA0[si] = bool(d > 0); Xs += [a, b]; ys += [np.full(len(a), float(isA0[si])), np.full(len(b), float(not isA0[si]))]
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack(Xs), np.concatenate(ys))
    for si, s in enumerate(segs):
        for i, p0, p1 in zip(s["frames"], s["tr"][0], s["tr"][1]):
            lab[i, p0] = int(isA0[si]); lab[i, p1] = int(not isA0[si]); in_seg[i] = True
    return isw.carry(kp, sc, pick, lab, in_seg, pr["cut"]), R

def match(ev, lab, used_ev=None):
    used = set() if used_ev is None else used_ev; hits = []
    for li, (a, e, c) in enumerate(lab):
        best, bo = None, 0
        for k, (pa, pe, pc) in enumerate(ev):
            if k in used:
                continue
            o = min(e + pt.TOL, pe) - max(a - pt.TOL, pa) + 1
            if o > bo:
                best, bo = k, o
        if best is not None:
            used.add(best); hits.append((li, best))
    return hits, used

def main():
    from multiprocessing import Pool
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_test*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(5) as pool:
        prs = pool.map(ip.prepare, ids)
    emb = ir2.make_models()["OSNet-AIN (re-identification)"]
    ds_ids = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "punch_ds", pre + "*.npz")))
    va = pt.load(ds_ids("W_train")[-pt.VAL_N:]); models = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn", "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); models.append(m)
    ens = lambda X: np.mean([pt.predict(m, X) for m in models], 0)
    Pva = [ens(d["X"][:2]) for d in va]; thr = max(np.arange(0.2, 0.81, 0.05), key=lambda t: pt.score(Pva, va, t)["f1"])
    T = {k: dict(lab=0, cnt=0, hit=0, typ=0, hand=0, wrong_fighter=0) for k in ("truth boxers", "full chain")}
    for pr in prs:
        d = np.load(os.path.join(ROOT, "external", "punch_ds", pr["id"] + ".npz")); Y = d["Y"][:2]; labs = [pt.labels_of(Y[0]), pt.labels_of(Y[1])]
        ident, R = identities(pr, emb); kp, sc = R["kp"].astype(np.float64), R["score"]
        role_pipe = np.where(ident == 1, 1, np.where(ident == 0, 2, 0)).astype(np.int8)
        st = {r: pd.stream(kp, sc, role_pipe, r) for r in (1, 2)}
        Xp = np.stack([pd.feats(*st[r], st[3 - r][0], st[3 - r][2]) for r in (1, 2)]); Mp = np.stack([st[r][2] for r in (1, 2)])
        for name, P, M in (("truth boxers", ens(d["X"][:2]), d["M"][:2]), ("full chain", ens(Xp), Mp)):
            evs = [pt.events(P[b], M[b], thr) for b in (0, 1)]; best = None
            for mapping in ((0, 1), (1, 0)) if name == "full chain" else ((0, 1),):          # stream b -> colour mapping[b]
                r = dict(lab=len(labs[0]) + len(labs[1]), cnt=len(evs[0]) + len(evs[1]), hit=0, typ=0, hand=0, wrong_fighter=0)
                for b in (0, 1):
                    hits, used = match(evs[b], labs[mapping[b]])
                    for li, k in hits:
                        c, pc = labs[mapping[b]][li][2], evs[b][k][2]; r["hit"] += 1; r["typ"] += c == pc; r["hand"] += (c % 2) == (pc % 2)
                    other, _ = match(evs[b], labs[1 - mapping[b]], set(used)); r["wrong_fighter"] += len(other)
                if best is None or r["hit"] > best["hit"]:
                    best = r
            for k, v in best.items():
                T[name][k] += v
        print(f"{pr['id'][7:42]:36s} boxer skeleton present: truth {d['M'][:2].mean():.0%}, chain {Mp.mean():.0%}", flush=True)
    print(f"\nBoxingWeb test rounds, threshold {thr:.2f} (validation), 3-seed ensemble")
    for name, t in T.items():
        r, p = t["hit"] / t["lab"], t["hit"] / max(t["cnt"], 1)
        print(f"  {name:13s} {t['lab']} labelled, {t['cnt']} counted: found+credited {r:.1%}, real {p:.1%}, F1 {2 * r * p / (r + p):.3f} | type right {t['typ'] / max(t['hit'], 1):.1%}, hand right {t['hand'] / max(t['hit'], 1):.1%} | "
              f"counts that belong to the OTHER boxer {t['wrong_fighter'] / max(t['cnt'], 1):.1%}")
    json.dump(dict(threshold=float(thr), **{k: {kk: float(vv) for kk, vv in t.items()} for k, t in T.items()}), open(os.path.join(ROOT, "runs", "punch_fullchain.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
