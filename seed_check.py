"""Is B3b's 63.1% real or a lucky seed?

Reruns B2 and B3b with several random-forest seeds. The LOVO folds are fixed
(one per video), so only the forest's randomness (bootstrap samples, feature
choices) and the random 80/20 split change between seeds.
"""
import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from baselines import evaluate, flatten, normalise, to_orthodox
from punch_data import load_dataset

SEEDS = [0, 1, 2, 3, 4]

X, y, video_id, _ = load_dataset()
norm = lambda X: flatten(normalise(X, "per_frame"))
configs = {"B2 norm": X, "B3b orthodox": to_orthodox(X, video_id)}

rows, per_video = [], {}
for name, data in configs.items():
    for seed in SEEDS:
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        with contextlib.redirect_stdout(io.StringIO()):   # silence per-run report
            r = evaluate(name, rf, norm, data, y, video_id, seed=seed)
        rows.append({"config": name, "seed": seed, "random": r["random"],
                     "lovo_mean": r["lovo_mean"], "lovo_worst": r["lovo"].min()})
        per_video[(name, seed)] = r["lovo"]
        print(f"{name:13s} seed {seed}: LOVO {r['lovo_mean']:.1%}", flush=True)

df = pd.DataFrame(rows)
print("\n=== Across seeds: mean ± std (min–max) ===")
for name, g in df.groupby("config", sort=False):
    for col in ["random", "lovo_mean", "lovo_worst"]:
        s = g[col]
        print(f"{name:13s} {col:10s}: {s.mean():.1%} ± {s.std():.1%}  "
              f"({s.min():.1%}–{s.max():.1%})")

print("\n=== B3b per-video LOVO, mean ± std across seeds ===")
pv = pd.DataFrame({s: per_video[("B3b orthodox", s)] for s in SEEDS})
print(pv.apply(lambda r: f"{r.mean():.0%} ± {r.std():.0%}", axis=1)
        .rename(lambda k: f"V{k}").to_string())
