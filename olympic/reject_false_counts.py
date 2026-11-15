"""Reject counted punches that are not real punches (feints, guard movements, the other boxer's reaction).
Olympic bouts, the chain's own 5,198 counted punches. Truth: a counted punch is REAL if a referee label is matched to it
one-to-one, time-compatible (label middle inside the span grown by 0.2 s) AND thrown by the same boxer (corrected puncher
rule, punch_tcn_oly.puncher_stream); among candidates the nearest glove to the referee's box wins.
A small classifier (gradient boosting, trained on one gym, tested on the other) scores each counted punch from what the
chain already computes: the pixel scan (how much any position looks like a contact, and the landed/blocked/missed
answer), the skeleton contact score, the detector's own confidence, whether the other boxer is also counted at that
moment, and the glove's reach towards the opponent. The keep/drop threshold is chosen on the training gym.
Reports thrown found / real / F1 and landed end to end, before and after.
    python olympic/reject_false_counts.py [cl_large]
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "boxingweb"))
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import punch_tcn_oly as po
TOL = 10; MAP = {"head": 0, "body": 0, "block": 1, "miss": 2}

def build(pre="cl_large"):
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); ev = json.load(open(os.path.join(HERE, "events.json"))); N = len(clips)
    S = np.full((N, 13, 4), np.nan)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", pre + "_kam*.npz"))):
        z = np.load(f); S[z["chain_idx"]] = z["chain_scan"]
    S = np.nan_to_num(S, nan=0.25); cen = 1 - S[:, :, 3]; k = cen.argmax(1); p3 = S[np.arange(N), k, :3]; p3 = p3 / np.maximum(p3.sum(1, keepdims=True), 1e-9)
    base = po.base_models(); idx = {}; X = np.zeros((N, 40), np.float32); real = np.zeros(N, bool); lab = np.full(N, -1); nlab = {}; nland = 0; names = None
    for n, c in enumerate(clips):
        idx.setdefault(c["id"], []).append(n)
    for cid, ns in idx.items():
        d = po.load(cid, ev, base); K, fr, P, M = d["K"], d["frame"], d["P"], d["M"]; step = d["step"]; F = len(fr); L = d["labs"]; nlab[cid] = len(L); nland += sum(l["outcome"] in ("head", "body") for l in L)
        C = np.full((len(L), len(ns)), 1e6)
        for i, l in enumerate(L):
            b = np.array(l["raw"]["boxes"]); bc = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
            for j, n in enumerate(ns):
                c = clips[n]
                if c["f0"] - TOL <= l["mid"] <= c["f1"] + TOL and (l["stream"] < 0 or l["stream"] == "AB".index(c["boxer"])):
                    C[i, j] = np.hypot(c["cx"] - bc[0], c["cy"] - bc[1]) / max(c["side"], 1.0)
        ri, ci = linear_sum_assignment(C)
        for i, j in zip(ri, ci):
            if C[i, j] < 1e5:
                real[ns[j]] = True; lab[ns[j]] = MAP[L[i]["outcome"]]
        pp = 1 - P[:, :, 0]
        for n in ns:
            c = clips[n]; s = "AB".index(c["boxer"]); a, e = int(np.searchsorted(fr, c["f0"])), min(int(np.searchsorted(fr, c["f1"])), F - 1); t = int(np.clip(round(c["coarse"] / step), 0, F - 1)); w = 9 if c["hand"] == "left" else 10
            own, oth = pp[s, a:e + 1], pp[1 - s, max(a - 5, 0):e + 6]; Ph = P[s, a:e + 1, 1:].sum(0); hand_m = Ph[[0, 2, 4]].sum() if c["hand"] == "left" else Ph[[1, 3, 5]].sum()
            Ks, Ko = K[s, t], K[1 - s, t]; T = max(np.linalg.norm(np.nanmean(Ks[[5, 6]], 0) - np.nanmean(Ks[[11, 12]], 0)), 20.0); g = Ks[w] + 0.4 * (Ks[w] - Ks[w - 2])
            dh = np.linalg.norm(g - np.nanmean(Ko[:5], 0)) / T; dt = np.linalg.norm(g - np.nanmean(Ko[[5, 6, 11, 12]], 0)) / T; ext = np.linalg.norm(Ks[w] - Ks[w - 4]) / T
            gap = np.linalg.norm(np.nanmean(Ks[[5, 6, 11, 12]], 0) - np.nanmean(Ko[[5, 6, 11, 12]], 0)) / T
            sp = np.linalg.norm(K[s, min(t + 1, F - 1), w] - K[s, max(t - 3, 0), w]) / T
            f = dict(cen_max=cen[n].max(), cen_mean=cen[n].mean(), cen_std=cen[n].std(), p_land=p3[n, 0], p_block=p3[n, 1], p_miss=p3[n, 2], contact_score=c.get("contact_score", 0.0), dur=e - a + 1,
                     det_max=own.max(), det_mean=own.mean(), other_max=oth.max() if len(oth) else 0.0, hand_share=hand_m / max(Ph.sum(), 1e-6), type_top=Ph.max() / max(Ph.sum(), 1e-6), cls=c["cls"], hook=float(c["cls"] in (3, 4)), upper=float(c["cls"] in (5, 6)),
                     glove_to_head=dh, glove_to_torso=dt, extension=ext, boxers_apart=gap, wrist_speed=sp, present=M[s, a:e + 1].mean(), other_present=M[1 - s, a:e + 1].mean())
            for q in range(13):
                f[f"cen{q}"] = cen[n, q]
            names = list(f); X[n, :len(names)] = np.nan_to_num(np.array([f[k] for k in names], np.float32), nan=-1.0, posinf=99.0)
    return clips, X[:, :len(names)], names, real, lab, p3, nlab, nland

def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "cl_large"; clips, X, names, real, lab, p3, nlab, nland = build(pre); N = len(clips); gym = np.array([c["gym"] for c in clips]); bout = np.array([c["id"] for c in clips])
    ntot = sum(nlab.values()); called = p3.argmax(1) == 0; score = np.zeros(N); keep = np.zeros(N, bool); thr = {}
    print(f"{N} counted punches, {ntot} referee labels: REAL (a label of the same boxer matched) {real.mean():.1%}")
    for g in ("kam2", "kam4"):
        tr, te = gym != g, gym == g; m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X[tr], real[tr])
        # threshold: out-of-fold scores inside the training gym (bouts split in 3), maximise the thrown F1 there
        oof = np.zeros(tr.sum()); bt = bout[tr]; ub = sorted(set(bt))
        for q in range(3):
            hold = np.isin(bt, ub[q::3]); mm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(X[tr][~hold], real[tr][~hold]); oof[hold] = mm.predict_proba(X[tr][hold])[:, 1]
        nl = sum(v for k, v in nlab.items() if g not in k)
        def f1(t):
            kk = oof >= t; tp = (kk & real[tr]).sum(); r, p = tp / nl, tp / max(kk.sum(), 1); return 2 * r * p / max(r + p, 1e-9)
        thr[g] = max(np.arange(0.1, 0.8, 0.02), key=f1); score[te] = m.predict_proba(X[te])[:, 1]; keep[te] = score[te] >= thr[g]
    print(f"real-vs-not AUC on the held-out gym: {np.mean([roc_auc_score(real[gym == g], score[gym == g]) for g in ('kam2', 'kam4')]):.3f}; thresholds from the training gym: " + ", ".join(f"{g} {t:.2f}" for g, t in thr.items()))
    for f in ("cen_max", "contact_score", "det_max", "glove_to_head", "other_max"):
        v = X[:, names.index(f)]; print(f"   single cue {f:14s} AUC {max(roc_auc_score(real, v), 1 - roc_auc_score(real, v)):.3f}")
    res = {}
    for name, kk in (("every counted punch (now)", np.ones(N, bool)), ("after rejecting", keep)):
        tp = (kk & real).sum(); r, p = tp / ntot, tp / kk.sum(); ltp = (kk & called & (lab == 0)).sum(); lr, lp = ltp / nland, ltp / max((kk & called).sum(), 1); m3 = kk & (lab >= 0)
        print(f"  {name:28s} counted {kk.sum():5d} | thrown: found {r:.1%}, real {p:.1%}, F1 {2 * r * p / (r + p):.3f} | landed end to end: found {lr:.1%}, real {lp:.1%}, F1 {2 * lr * lp / (lr + lp):.3f} | 3-way right on real ones {np.mean(p3[m3].argmax(1) == lab[m3]):.1%}")
        res[name] = dict(counted=int(kk.sum()), found=float(r), real=float(p), landed_found=float(lr), landed_real=float(lp))
    print(f"  dropped {np.mean(~keep):.1%} of counts: {np.mean(~real[~keep]):.0%} of the dropped were not real; real punches lost {np.mean(~keep[real]):.1%} (landed {np.mean(~keep[lab == 0]):.1%}, blocked {np.mean(~keep[lab == 1]):.1%}, missed {np.mean(~keep[lab == 2]):.1%})")
    np.savez_compressed(os.path.join(ROOT, "runs", "olympic_reject_scores.npz"), score=score, keep=keep, real=real, lab=lab)
    json.dump(dict(res, thresholds=thr, features=names), open(os.path.join(ROOT, "runs", "olympic_reject_false_counts.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    main()
