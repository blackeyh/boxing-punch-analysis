"""Per-clip leave-one-video-out predictions, for error analysis.

For each video, train the CNN baseline on the other 9 videos and predict every clip
of the held-out video. Probabilities are averaged over seeds for stability.

Output: runs/lovo_predictions/preds.npz
    y_true, y_pred (str), proba (N, 6, classes order = CLASSES), video_id, lengths
Run:    /opt/anaconda3/bin/python lovo_predictions.py [--seeds 0 1 2]
"""
import argparse
from pathlib import Path

import mem_guard  # noqa: F401  before torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from baselines import to_orthodox
from deep import TorchSeqClassifier, seq_features
from progress import progress
from punch_data import CLASSES, load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
ap.add_argument("--epochs", type=int, default=60)
args = ap.parse_args()

X, y, video_id, lengths = load_dataset()
F = seq_features(to_orthodox(X, video_id))
videos = np.unique(video_id)
proba = np.zeros((len(y), len(CLASSES)))

progress.start("lovo_predictions.py", total=len(args.seeds) * len(videos))
for seed in args.seeds:
    for vid in videos:
        te = video_id == vid
        m = TorchSeqClassifier("cnn", epochs=args.epochs, seed=seed).fit(F[~te], y[~te])
        order = [list(m.classes_).index(c) for c in CLASSES]
        proba[te] += m.predict_proba(F[te])[:, order] / len(args.seeds)
        progress.tick(1, note=f"seed {seed} V{vid}")
progress.finish()

y_pred = np.array(CLASSES)[proba.argmax(1)]
per_video = {int(v): balanced_accuracy_score(y[video_id == v], y_pred[video_id == v]) for v in videos}
print("LOVO mean (seed-averaged probabilities):", f"{np.mean(list(per_video.values())):.1%}")
print("per video:", "  ".join(f"V{k} {v:.0%}" for k, v in per_video.items()))
print("wrong clips:", int((y_pred != y).sum()), "of", len(y))

out = Path("runs/lovo_predictions"); out.mkdir(parents=True, exist_ok=True)
np.savez(out / "preds.npz", y_true=y.astype(str), y_pred=y_pred.astype(str), proba=proba, video_id=video_id, lengths=lengths)
print("saved", out / "preds.npz")
