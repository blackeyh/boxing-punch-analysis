"""Learned boxer selector, skeleton-only, trained with BoxingWeb's per-frame red/blue boxer truth.

Train: BoxingWeb data_train rounds (person-frames from frames where both boxers were matched; role > 0 = boxer).
Test 1 (exact truth): BoxingWeb data_test rounds, per frame: are the picks exactly the two boxers?
Test 2 (other camera set-up, never trained on): the Olympic gym bouts, via the referee's punch boxes.
Picks: smoothed P(boxer) along the tracklet, then the same-apparent-size pair rule (or plain top 2 for comparison).
    python boxingweb/train_selector.py
"""
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "olympic"))
import feats, select_boxers as sb, selector_ring as sr
R, DMAX, THR, THR_LOW, WIN = 1.5, 4.0, 0.5, 0.25, 25

def smooth(D, P):
    S = np.full(P.shape, np.nan); fs, ks = np.nonzero(D["tid"] >= 0); t = D["tid"][fs, ks]
    order = np.lexsort((fs, t)); fs, ks, t = fs[order], ks[order], t[order]; bounds = np.flatnonzero(np.diff(t)) + 1
    for f, k in zip(np.split(fs, bounds), np.split(ks, bounds)):
        a = np.searchsorted(f, f - WIN); b = np.searchsorted(f, f + WIN, "right"); cs = np.r_[0, np.cumsum(P[f, k])]
        S[f, k] = (cs[b] - cs[a]) / (b - a)
    return S

def pick_pair(D, P, pair=True):
    hip = (D["kp"][:, :, 11] + D["kp"][:, :, 12]) / 2; S = D["S"]; F = len(D["frame"]); pick = np.full((F, 2), -1, np.int32)
    for f in range(F):
        c = [k for k in range(int(D["n"][f])) if P[f, k] >= THR_LOW]
        if not pair:
            o = sorted((k for k in c if P[f, k] >= THR), key=lambda k: -P[f, k])[:2]; pick[f, :len(o)] = o; continue
        best, bs = None, -1
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                i, j = c[a], c[b]
                if max(P[f, i], P[f, j]) < THR or not (1 / R <= S[f, i] / S[f, j] <= R):
                    continue
                if np.linalg.norm(hip[f, i] - hip[f, j]) > DMAX * max(S[f, i], S[f, j]):
                    continue
                if P[f, i] + P[f, j] > bs:
                    best, bs = (i, j), P[f, i] + P[f, j]
        if best:
            pick[f] = best
        else:
            c = [k for k in c if P[f, k] >= THR]
            if c:
                pick[f, 0] = max(c, key=lambda k: P[f, k])
    return pick

def exact_eval(role, pick):
    """On frames where both boxers have an RTMPose skeleton."""
    both = (role == 1).any(1) & (role == 2).any(1); out = np.zeros(5)
    for f in np.flatnonzero(both):
        ps = [p for p in pick[f] if p >= 0]; r = [role[f, p] for p in ps]
        wrong = any(x == 0 for x in r)
        out += [1, (not wrong) and len(ps) == 2, wrong, (not wrong) and len(ps) == 1, len(ps) == 0]
    return out                                             # frames, exact, a non-boxer picked, one boxer only, nobody

def rows(D, role, step=3):
    both = (role == 1).any(1) & (role == 2).any(1); fr = np.flatnonzero(both)[::step]
    m = role[fr] >= 0
    return D["X2"][fr][m], (role[fr][m] > 0).astype(int)

def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    P = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", pre + "*.npz")) if os.path.basename(p).count(".") == 1)
    tr, te, oly = P("W_train"), P("W_test"), P("O_")
    role = {c: np.load(os.path.join(ROOT, "external", "boxers_gt", c + ".npz"))["role"] for c in tr + te}
    Xs, ys = zip(*(rows(feats.load(c), role[c]) for c in tr)); X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"training rows {len(y)} ({y.mean():.1%} boxers) from {len(tr)} rounds, {X.shape[1]} skeleton features", flush=True)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, random_state=0).fit(X, y)
    ev = json.load(open(os.path.join(ROOT, "olympic", "events.json")))
    # baseline: the 9-feature logistic selector fitted on ALL Olympic weak labels (it has never seen BoxingWeb)
    ox, oy = [], []
    for c in oly:
        D = sb.load(c); rws = sb.weak_labels(D, [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]])
        ox += [D["X"][i, p] for i, p, _ in rws]; oy += [l for *_, l in rws]
    lr = sb.fit_lr(np.array(ox), np.array(oy))
    res = {}
    tot = {k: np.zeros(5) for k in ("old 9-feature selector (Olympic-trained) + pair rule", "learned, top 2", "learned + pair rule")}
    acc = []
    out_dir = os.path.join(ROOT, "external", "boxers_rtm"); os.makedirs(out_dir, exist_ok=True)
    for c in te:
        D = feats.load(c); m = D["tid"] >= 0
        Pg = np.full(m.shape, np.nan); Pg[m] = gbm.predict_proba(D["X2"][m])[:, 1]
        r = role[c]; v = r >= 0; acc.append((((Pg[v] >= 0.5) == (r[v] > 0)).sum(), v.sum(), ((Pg[v] >= .5) & (r[v] > 0)).sum(), (r[v] > 0).sum(), ((Pg[v] < .5) & (r[v] == 0)).sum(), (r[v] == 0).sum()))
        Ps = smooth(D, np.nan_to_num(Pg)); Pl = smooth(D, np.nan_to_num(np.where(m, sb.prob(lr, np.nan_to_num(D["X"])), np.nan)))
        pk = pick_pair(D, Ps); np.savez_compressed(os.path.join(out_dir, c + ".npz"), frame=D["frame"], pick=pk, prob=Ps)
        tot["old 9-feature selector (Olympic-trained) + pair rule"] += exact_eval(r, pick_pair(D, Pl))
        tot["learned, top 2"] += exact_eval(r, pick_pair(D, Ps, pair=False)); tot["learned + pair rule"] += exact_eval(r, pk)
    a = np.array(acc).sum(0)
    print(f"\nBoxingWeb TEST rounds ({len(te)}): person-frame accuracy {a[0] / a[1]:.1%} | boxers recognised {a[2] / a[3]:.1%} | others rejected {a[4] / a[5]:.1%}")
    for k, t in tot.items():
        print(f"  {k:52s} exactly the 2 boxers {t[1] / t[0]:.1%} | a non-boxer picked {t[2] / t[0]:.1%} | one boxer only {t[3] / t[0]:.1%} | nobody {t[4] / t[0]:.1%}  ({int(t[0])} frames)")
        res[k] = t.tolist()
    # Olympic gyms: never seen in training
    for held in ("kam2", "kam4"):
        t = np.zeros(6)
        for c in [c for c in oly if held in c]:
            D = feats.load(c); m = D["tid"] >= 0; Pg = np.full(m.shape, np.nan); Pg[m] = gbm.predict_proba(D["X2"][m])[:, 1]
            Ps = smooth(D, np.nan_to_num(Pg)); pk = pick_pair(D, Ps)
            np.savez_compressed(os.path.join(out_dir, c + ".npz"), frame=D["frame"], pick=pk, prob=Ps)
            es = [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]]
            k, n, w, s = sr.evaluate(D, es, pk); two = (pk >= 0).sum(1); t += [k, n, w, s, (two == 2).sum(), len(two)]
        print(f"Olympic {held} (never trained on): puncher kept {t[0] / t[1]:.1%} of {int(t[1])} | picks nowhere near the punch {t[2] / t[3]:.1%} | frames with 2 selected {t[4] / t[5]:.1%}")
        res["olympic_" + held] = t.tolist()
    imp = None
    json.dump(res, open(os.path.join(ROOT, "runs", "selector_learned.json"), "w"), indent=1)
    import joblib; os.makedirs(os.path.join(ROOT, "runs", "selector_learned"), exist_ok=True)
    joblib.dump(gbm, os.path.join(ROOT, "runs", "selector_learned", "gbm_boxingweb_train.joblib"))

if __name__ == "__main__":
    main()
