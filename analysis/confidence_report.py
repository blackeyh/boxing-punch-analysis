"""Step 9 analysis: (a) performance of single models and 2D/3D ensembles, (b) MediaPipe joint
confidence of the clips each model gets WRONG vs RIGHT.

Confidence per clip = mean MediaPipe visibility over the clip's real frames (V4 mirrored like
everything else, so 'lead wrist' is always joint 9). Groups: all joints, arms (5-10), wrists (9, 10).

Ensembles: per seed, average the per-clip LOVO probabilities of the member models (seed i with
seed i), take the argmax; every probability already comes from a model that never saw that clip's
video, so the ensemble is a fair leave-one-video-out result. Reported as mean +- std over seeds.

    /opt/anaconda3/bin/python analysis/confidence_report.py
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baselines import to_orthodox  # noqa: E402
from deep import frontalness  # noqa: E402
from lift3d.features import load_mediapipe3d  # noqa: E402
from punch_data import CLASSES, load_dataset  # noqa: E402

pd.set_option("display.width", 250)
X, y, v, L = load_dataset()
_, conf = load_mediapipe3d(y, v, with_conf=True)
real = np.arange(conf.shape[1])[None] < L[:, None]


def clip_conf(joints):
    c = conf[:, :, joints].mean(axis=2)                     # (N, 25) mean over the joint group
    return np.nanmean(np.where(real, c, np.nan), axis=1)     # mean over real frames


CONF = {"all joints": clip_conf(list(range(17))), "arms": clip_conf([5, 6, 7, 8, 9, 10]), "wrists": clip_conf([9, 10])}
front = np.abs(frontalness(to_orthodox(X, v)))
jab = y == "Jab"
side_q = np.full(len(y), -1)
side_q[jab] = pd.qcut(front[jab], 5, labels=False)


def load(pattern):
    fs = sorted(glob.glob(str(ROOT / "runs" / pattern / "results.json")))
    if not fs:
        return None
    r = json.load(open(fs[-1]))
    return r["runs"]


def lovo(pred):
    return np.mean([balanced_accuracy_score(y[v == k], pred[v == k]) for k in np.unique(v)])


def summarise(name, preds):
    """preds: list (one per seed) of per-clip predicted labels."""
    s = np.array([lovo(p) for p in preds])
    side = np.mean([np.mean(p[jab & (side_q == 0)] == "Lead Hook") for p in preds])
    rh = np.mean([np.mean(p[y == "Rear Hook"] == "Rear Hook") for p in preds])
    row = {"model": name, "LOVO": s.mean(), "std": s.std(ddof=1), "side-on jab->LHook": side, "Rear Hook": rh}
    wrong = np.mean([p != y for p in preds], axis=0) > 0.5          # wrong in most seeds
    for g, c in CONF.items():
        row[f"{g} conf: wrong"] = np.nanmean(c[wrong])
        row[f"{g} conf: right"] = np.nanmean(c[~wrong])
    return row, wrong


MODELS = {   # name: run pattern
    "2D": "s9-2d-local",
    "lifted 3D rot": "s9-lift-rot-local",
    "MediaPipe 3D rot": "s8-mp3d-rot-2*",
    "2D + MediaPipe 3D rot": "s8b-2d-plus-mp3d-rot-2*",
    "MP 3D rot + fill": "s9-mp3d-rot-fill-local",
    "MP 3D rot + fill + conf": "s9-mp3d-rot-fill-conf-local",
    "2D + MP 3D rot + fill + conf": "s9-2d-mp3d-rot-fill-conf-local",
}
ENSEMBLES = {
    "ENS 2D + lifted 3D": ["2D", "lifted 3D rot"],
    "ENS 2D + MP 3D (fill+conf)": ["2D", "MP 3D rot + fill + conf"],
    "ENS 2D + lifted + MP 3D": ["2D", "lifted 3D rot", "MP 3D rot + fill + conf"],
}

runs, rows, wrong_sets = {}, [], {}
for name, pat in MODELS.items():
    r = load(pat)
    if r is None:
        print(f"(missing: {name})")
        continue
    runs[name] = r
    row, wrong = summarise(name, [np.array(x["lovo_pred"]) for x in r])
    rows.append(row); wrong_sets[name] = wrong
for name, members in ENSEMBLES.items():
    if not all(m in runs and "lovo_proba" in runs[m][0] for m in members):
        print(f"(skipped: {name}, members missing or without probabilities)")
        continue
    n_seeds = min(len(runs[m]) for m in members)
    preds = []
    for s in range(n_seeds):
        P = np.mean([np.array(runs[m][s]["lovo_proba"]) for m in members], axis=0)
        preds.append(np.array(CLASSES)[P.argmax(1)])
    row, wrong = summarise(name, preds)
    rows.append(row); wrong_sets[name] = wrong

df = pd.DataFrame(rows).set_index("model")
print("\n=== Performance (CNN, 5 seeds; ensembles average probabilities per seed) ===")
print(df[["LOVO", "std", "side-on jab->LHook", "Rear Hook"]].map(lambda x: f"{x:.1%}").to_string())
print("\n=== MediaPipe confidence (0-1) of clips each model gets WRONG vs RIGHT (wrong = in most seeds) ===")
print(df[[c for c in df.columns if "conf" in c]].round(3).to_string())

print("\n=== Accuracy by WRIST-confidence quintile (lowest -> highest) ===")
q = pd.qcut(CONF["wrists"], 5, labels=False)
edges = pd.qcut(CONF["wrists"], 5).categories
print("quintile wrist-conf ranges:", ", ".join(f"{e.left:.2f}-{e.right:.2f}" for e in edges))
acc = {n: [np.mean(~w[q == k]) for k in range(5)] for n, w in wrong_sets.items()}
print(pd.DataFrame(acc, index=[f"Q{k + 1}" for k in range(5)]).T.map(lambda x: f"{x:.0%}").to_string())
print(f"\nshare of ALL clips with wrist conf < 0.5: {np.mean(CONF['wrists'] < 0.5):.1%}")
for n, w in wrong_sets.items():
    print(f"  {n:32s} wrong clips with wrist conf < 0.5: {np.mean(CONF['wrists'][w] < 0.5):.1%} of its {w.sum()} wrong clips")
