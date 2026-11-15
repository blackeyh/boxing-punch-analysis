"""Learned selector, step 2: add gym footage to the training set.

Training = BoxingWeb data_train (exact per-frame truth) + the Olympic bouts of ONE gym (free labels from the
referee's punch boxes: the two people at a contact punch are boxers, people nowhere near it are not).
The other gym is held out and tested; BoxingWeb data_test checks that nothing is lost on broadcast footage.
Olympic rows are few (a few thousand vs 425k), so they are up-weighted to OLY_SHARE of the total weight.
    python boxingweb/train_selector_mix.py
"""
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "olympic"))
import feats, select_boxers as sb, selector_ring as sr, train_selector as ts
OLY_SHARE = 0.3

def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    P = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", pre + "*.npz")) if os.path.basename(p).count(".") == 1)
    tr, te, oly = P("W_train"), P("W_test"), P("O_")
    role = {c: np.load(os.path.join(ROOT, "external", "boxers_gt", c + ".npz"))["role"] for c in tr + te}
    Xs, ys = zip(*(ts.rows(feats.load(c), role[c]) for c in tr)); Xw, yw = np.vstack(Xs), np.concatenate(ys)
    ev = json.load(open(os.path.join(ROOT, "olympic", "events.json")))
    O = {}
    for c in oly:
        D = feats.load(c); es = [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]]
        rw = sb.weak_labels(D, es); O[c] = (D, es, np.array([D["X2"][i, p] for i, p, _ in rw]), np.array([l for *_, l in rw]))
    out_dir = os.path.join(ROOT, "external", "boxers_rtm_mix"); os.makedirs(out_dir, exist_ok=True)
    res = {}
    for held in ("kam2", "kam4"):
        src = [c for c in oly if held not in c]
        Xo, yo = np.vstack([O[c][2] for c in src]), np.concatenate([O[c][3] for c in src])
        w = np.r_[np.ones(len(yw)), np.full(len(yo), OLY_SHARE / (1 - OLY_SHARE) * len(yw) / len(yo))]
        gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        gbm.fit(np.vstack([Xw, Xo]), np.r_[yw, yo], sample_weight=w)
        t = np.zeros(6)
        for c in [c for c in oly if held in c]:
            D, es = O[c][:2]; m = D["tid"] >= 0; Pg = np.full(m.shape, np.nan); Pg[m] = gbm.predict_proba(D["X2"][m])[:, 1]
            Ps = ts.smooth(D, np.nan_to_num(Pg)); pk = ts.pick_pair(D, Ps)
            np.savez_compressed(os.path.join(out_dir, c + ".npz"), frame=D["frame"], pick=pk, prob=Ps)
            k, n, wr, s = sr.evaluate(D, es, pk); two = (pk >= 0).sum(1); t += [k, n, wr, s, (two == 2).sum(), len(two)]
        print(f"held-out gym {held} (+{len(yo)} gym rows from the other gym): puncher kept {t[0] / t[1]:.1%} | picks nowhere near the punch {t[2] / t[3]:.1%} | frames with 2 selected {t[4] / t[5]:.1%}", flush=True)
        tot = np.zeros(5)
        for c in te:
            D = feats.load(c); m = D["tid"] >= 0; Pg = np.full(m.shape, np.nan); Pg[m] = gbm.predict_proba(D["X2"][m])[:, 1]
            tot += ts.exact_eval(role[c], ts.pick_pair(D, ts.smooth(D, np.nan_to_num(Pg)), pair=False))
        print(f"    same model on BoxingWeb test (top 2): exactly the 2 boxers {tot[1] / tot[0]:.1%} | a non-boxer picked {tot[2] / tot[0]:.1%} | one boxer only {tot[3] / tot[0]:.1%}", flush=True)
        res[held] = dict(olympic=t.tolist(), boxingweb=tot.tolist())
    json.dump(res, open(os.path.join(ROOT, "runs", "selector_learned_mix.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
