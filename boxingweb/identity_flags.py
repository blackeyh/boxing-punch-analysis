"""Can uncertain separations be FLAGGED, so identity errors are not silent? (skeleton-only, BoxingWeb test rounds)
A flagged separation would be shown as "uncertain" or sent to one user tap, instead of guessed.
    python boxingweb/identity_flags.py
"""
import glob, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import identity_episodes as ie

def main():
    from multiprocessing import Pool
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ids = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", pre + "*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as pool:
        tr = pool.map(ie.load_round, ids("W_train")); te = pool.map(ie.load_round, ids("W_test"))
    rows = lambda rds: [r for rd in rds for r in rd["rows"] if not r["cut"]]
    Xtr = np.clip(np.array([r["x"] for r in rows(tr)]), -10, 10); ytr = np.array([r["y"] for r in rows(tr)])
    R = rows(te); X = np.clip(np.array([r["x"] for r in R]), -10, 10); y = np.array([r["y"] for r in R]); gap = np.array([r["gap"] for r in R]) / 30
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xtr, ytr)
    conf = np.abs(lr.predict_proba(X)[:, 1] - 0.5) * 2; wrong = (X[:, 0] >= 0).astype(int) != y
    mins = sum(rd["sec"] for rd in te) / 60
    print(f"{len(y)} separations in {mins:.0f} min; continuity wrong in {wrong.sum()} ({wrong.sum() / mins:.2f}/min)")
    print("overlap length (s), percentiles 50/75/90:", np.percentile(gap, [50, 75, 90]).round(2))
    for name, flag in (("overlap lasted >= 1 s", gap >= 1.0), ("overlap lasted >= 2 s", gap >= 2.0), ("model confidence < 0.6", conf < 0.6),
                       ("model confidence < 0.8", conf < 0.8), (">= 1 s OR confidence < 0.6", (gap >= 1) | (conf < 0.6))):
        print(f"  flag '{name}': {flag.mean():.0%} of separations ({flag.sum() / mins:.1f}/min) flagged, catching {wrong[flag].sum()}/{wrong.sum()} errors; "
              f"unflagged right {1 - wrong[~flag].mean():.1%} ({wrong[~flag].sum() / mins:.2f} silent errors/min)")

if __name__ == "__main__":
    main()
