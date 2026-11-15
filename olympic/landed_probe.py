"""Step 3: landed / blocked / missed from PIXELS at the contact point (Olympic set, referee labels).
Features: VideoMAE-L / -B on 16-frame contact clips (hpc_rtm/landed_feats.py); crops centred on the puncher's glove
("glove", available at run time) or on the referee's box ("box", upper bound). Probe: logistic regression with strong
L2, class-balanced, trained on one gym and tested on the other. Also: skeleton cues alone (olympic/landed_cues.py)
and pixels + skeleton cues (average of the two models' log-probabilities).
    python olympic/landed_probe.py
"""
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

def report(name, y, P):
    yh = P.argmax(1); cm = np.zeros((3, 3), int); np.add.at(cm, (y, yh), 1); f1 = []
    for k in range(3):
        pr = cm[k, k] / max(cm[:, k].sum(), 1); rc = cm[k, k] / max(cm[k].sum(), 1); f1.append(2 * pr * rc / max(pr + rc, 1e-9))
    # binary landed: threshold-free AUC and the precision/recall at the argmax decision
    lp = cm[0, 0] / max(cm[:, 0].sum(), 1); lr_ = cm[0, 0] / max(cm[0].sum(), 1)
    print(f"  {name:38s} 3-way accuracy {np.trace(cm) / cm.sum():.1%}, macro-F1 {np.mean(f1):.3f} (landed {f1[0]:.2f}, blocked {f1[1]:.2f}, missed {f1[2]:.2f}) | "
          f"landed-vs-not AUC {auc((y == 0).astype(int), P[:, 0]):.3f}; called landed are landed {lp:.1%}, landed caught {lr_:.1%}")
    return dict(acc=float(np.trace(cm) / cm.sum()), macro_f1=float(np.mean(f1)), f1=[float(v) for v in f1], auc_landed=float(auc((y == 0).astype(int), P[:, 0])), cm=cm.tolist())

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips.json"))); N = len(clips)
    F = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_feats", "landed_feats_*.npz"))):
        z = np.load(f)
        for k in z.files:
            if k != "order":
                F.setdefault(k, np.full((N, z[k].shape[1]), np.nan, np.float32))[z["order"]] = z[k].astype(np.float32)
    have = ~np.isnan(F["large_glove"][:, 0]); y = np.array([c["y"] for c in clips]); gym = np.array([c["gym"] for c in clips])
    cz = np.load(os.path.join(ROOT, "external", "olympic_landed_cues.npz")); assert len(cz["y"]) == N and (cz["y"] == y).all()
    print(f"{N} punches, features for {have.sum()}: landed {np.mean(y[have] == 0):.1%}, blocked {np.mean(y[have] == 1):.1%}, missed {np.mean(y[have] == 2):.1%} (always 'landed' = {np.mean(y[have] == 0):.1%} accuracy)")
    out = {}; P = {}
    def cv(fit, X):
        Pr = np.zeros((N, 3))
        for held in ("kam2", "kam4"):
            tr, te = have & (gym != held), have & (gym == held); Pr[te] = fit(X[tr], y[tr]).predict_proba(X[te])
        return Pr
    lr = lambda C: (lambda X, Y: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced")).fit(X, Y))
    P["skeleton cues"] = cv(lambda X, Y: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=0).fit(X, Y), cz["X"])
    for k in ("large_glove", "base_glove", "large_box", "base_box"):
        if k in F:
            best = None
            for C in (1e-4, 1e-3, 1e-2):
                Pr = cv(lr(C), F[k]); a = (Pr[have].argmax(1) == y[have]).mean()
                if best is None or a > best[0]:
                    best = (a, C, Pr)
            P[f"pixels VideoMAE-{k.replace('_', ', crop on ')} (C={best[1]:g})"] = best[2]
    names = list(P); g = [n for n in names if "large, crop on glove" in n]
    if g:
        P["pixels (L, glove) + skeleton cues"] = np.exp((np.log(P[g[0]] + 1e-9) + np.log(P["skeleton cues"] + 1e-9)) / 2); P["pixels (L, glove) + skeleton cues"] /= P["pixels (L, glove) + skeleton cues"].sum(1, keepdims=True)
        both = [n for n in names if "large, crop on glove" in n or "base, crop on glove" in n]
        if len(both) == 2:
            P["pixels L + B (glove)"] = (P[both[0]] + P[both[1]]) / 2
    for n, Pr in P.items():
        out[n] = report(n, y[have], Pr[have])
    for held in ("kam2", "kam4"):
        m = have & (gym == held)
        if g:
            print(f"  per gym, pixels (L, glove), tested on {held}: accuracy {(P[g[0]][m].argmax(1) == y[m]).mean():.1%} (always 'landed' {np.mean(y[m] == 0):.1%}), landed AUC {auc((y[m] == 0).astype(int), P[g[0]][m, 0]):.3f}")
    json.dump(out, open(os.path.join(ROOT, "runs", "olympic_landed_probe.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
