"""Small LOVO grid for the probe: feature set x per-video standardisation x C x head (7-way / binary).
Every number is still a leave-one-video-out score; choosing the best of this small grid on those
numbers is a mild selection bias (reported in REPORT.md). Output videodet/probe_grid.json."""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score as S  # noqa: E402
from progress import progress  # noqa: E402


def run(D, kind, pervideo, C, head):
    X = {v: S.feature(D[v], kind) for v in S.VIDS}
    if pervideo:
        X = {v: (x - x.mean(0)) / (x.std(0) + 1e-6) for v, x in X.items()}
    y = {}
    for v in S.VIDS:
        t = np.full(len(X[v]), -1); t[D[v]["is_gap"]] = 0
        for c in range(1, 7):
            t[D[v]["is_punch"] & (D[v]["label"] == S.CLASSES[c])] = c if head == "7way" else 1
        y[v] = t
    ncls = 7 if head == "7way" else 2
    sc, P = {}, {}
    for k in S.VIDS:
        tr = [v for v in S.VIDS if v != k]
        Xtr = np.concatenate([X[v][y[v] >= 0] for v in tr]); ytr = np.concatenate([y[v][y[v] >= 0] for v in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        W, b = S.fit_softmax((Xtr - mu) / sd, ytr, C, n_cls=ncls)
        P[k] = S.predict_softmax((X[k] - mu) / sd, W, b)
        sc[k] = 1 - P[k][:, 0]
    rows, pooled = S.validate(D, sc)
    return sc, P, rows, pooled


def one(D, cfg):
    import torch
    torch.set_num_threads(4)
    return cfg, run(D, *cfg)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    D = S.load()
    grid = list(itertools.product(["base", "large", "base+large"], [False, True], [0.01, 0.001], ["7way", "binary"]))
    progress.start("videodet probe grid", total=len(grid))
    res = []
    for (kind, pv, C, head), (_, _, rows, pooled) in Parallel(n_jobs=4, return_as="generator_unordered")(
            delayed(one)(D, cfg) for cfg in grid):
        r = dict(kind=kind, pervideo=pv, C=C, head=head, mean_auc=pooled["mean_auc"], pooled_auc=pooled["pooled_auc"],
                 t_glob=pooled["t_glob"], gap_frac_tglob=float(np.mean([x["gap_frac_tglob"] for x in rows])),
                 auc=[round(x["auc"], 3) for x in rows])
        res.append(r)
        print(f"{kind:11s} pervideo={pv!s:5s} C={C:<6} {head:6s} meanAUC {r['mean_auc']:.3f} pooledAUC {r['pooled_auc']:.3f} "
              f"gap>tglob {r['gap_frac_tglob']:.2f}  {r['auc']}", flush=True)
        progress.tick(1, note=f"{kind} pv={pv} C={C} {head}")
        (Path(os.environ.get("RUN_DIR", HERE)) / "probe_grid.json").write_text(json.dumps(res, indent=1))
    progress.finish()
