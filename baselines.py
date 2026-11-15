"""Baselines for punch classification.

Every experiment is scored the same way:
  * random 80/20 stratified split   -> balanced accuracy (optimistic reference)
  * leave-one-video-out (10 folds)  -> balanced accuracy per video (the real test)
Goal: LOVO mean balanced accuracy >= 60%.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, train_test_split

from progress import progress
from punch_data import CLASSES, load_dataset

FITS_PER_EVAL = 11  # 1 random split + 10 leave-one-video-out folds

# Classes predicted but absent from a test video (e.g. Jab in V6) trigger a warning;
# balanced accuracy already averages recall over the classes that are present.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")


def flatten(X):
    """B1 features: raw coordinates, one row per clip (25*17*2 = 850 values)."""
    return X.reshape(len(X), -1)


L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def normalise(X, mode="per_frame"):
    """Body-relative coordinates: hip midpoint at (0, 0), torso length = 1.

    mode="per_frame": every frame is centred on its own hip midpoint.
    mode="first"    : every frame is centred on the hip midpoint of frame 0,
                      so movement of the whole body during the punch is kept.
    Scale is one number per clip: median torso length over the real frames.
    Padding frames stay exactly zero.
    """
    real = np.abs(X).sum(axis=(2, 3)) > 0                       # (N, T)
    assert real[:, 0].all(), "expected padding only at the end of each clip"

    hip = X[:, :, [L_HIP, R_HIP]].mean(axis=2)                  # (N, T, 2)
    shoulder = X[:, :, [L_SHOULDER, R_SHOULDER]].mean(axis=2)   # (N, T, 2)
    torso = np.linalg.norm(shoulder - hip, axis=-1)             # (N, T)
    scale = np.nanmedian(np.where(real, torso, np.nan), axis=1) # (N,)
    scale = np.maximum(scale, 1e-3)                             # guard against /0

    if mode == "per_frame":
        centre = hip[:, :, None, :]                             # (N, T, 1, 2)
    elif mode == "first":
        centre = hip[:, :1, None, :]                            # (N, 1, 1, 2)
    else:
        raise ValueError(mode)

    Z = (X - centre) / scale[:, None, None, None]
    Z[~real] = 0.0
    return Z


# COCO-17 index of each keypoint's mirror partner (left <-> right).
MIRROR_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def mirror(X):
    """Flip clips horizontally: x -> 1 - x and swap left/right keypoints.
    A mirrored orthodox jab looks like a southpaw jab, so the label is unchanged.
    Padding frames stay zero."""
    real = np.abs(X).sum(axis=(2, 3)) > 0
    M = X[:, :, MIRROR_IDX].copy()
    M[..., 0] = np.where(real[:, :, None], 1.0 - M[..., 0], 0.0)
    return M


# Stance canonicalisation: V4 is southpaw (its jabs come from the right wrist),
# so mirror it to look orthodox, in training AND testing. In the app, the user
# states their stance once, so this information is available live.
SOUTHPAW_VIDEOS = [4]


def to_orthodox(X, video_id):
    X_orth = X.copy()
    for vid in SOUTHPAW_VIDEOS:
        X_orth[video_id == vid] = mirror(X[video_id == vid])
    return X_orth


def evaluate(name, model, features, X, y, video_id, augment=False, seed=0):
    """augment=True adds a mirrored copy of every TRAINING clip (never test clips)."""
    F = features(X)

    def fit(tr):
        Xtr, ytr = X[tr], y[tr]
        if augment:
            Xtr, ytr = np.concatenate([Xtr, mirror(Xtr)]), np.concatenate([ytr, ytr])
        m = clone(model).fit(features(Xtr), ytr)
        if not getattr(model, "reports_progress", False):  # torch models tick per epoch themselves
            progress.tick(1, note=name)
        return m

    # 1. Random 80/20 split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2,
                              stratify=y, random_state=seed)
    m = fit(tr)
    random_score = balanced_accuracy_score(y[te], m.predict(F[te]))

    # 2. Leave-one-video-out
    per_video, y_true, y_pred = {}, [], []
    lovo_pred = np.empty(len(y), dtype=object)   # per-clip LOVO prediction, in input order
    lovo_proba = np.full((len(y), len(CLASSES)), np.nan)   # per-clip probabilities (CLASSES order), for ensembles
    for tr, te in LeaveOneGroupOut().split(F, y, groups=video_id):
        m = fit(tr)
        pred = m.predict(F[te])
        if hasattr(m, "predict_proba"):
            pp = m.predict_proba(F[te])
            lovo_proba[te] = pp[:, [list(m.classes_).index(c) for c in CLASSES]]
        per_video[int(video_id[te][0])] = balanced_accuracy_score(y[te], pred)
        y_true.append(y[te]); y_pred.append(pred); lovo_pred[te] = pred

    lovo = pd.Series(per_video).sort_index()
    print(f"\n=== {name} ===")
    print(f"random 80/20 : {random_score:6.1%}")
    print(f"LOVO mean    : {lovo.mean():6.1%}   worst: V{lovo.idxmin()} {lovo.min():.1%}"
          f"   best: V{lovo.idxmax()} {lovo.max():.1%}")
    print("LOVO per video:", "  ".join(f"V{k} {v:.0%}" for k, v in lovo.items()))

    cm = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred),
                          labels=CLASSES, normalize="true")
    print("LOVO confusion (rows = true, row-normalised):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).round(2).to_string())
    return {"random": random_score, "lovo_mean": lovo.mean(), "lovo": lovo, "confusion": cm,
            "lovo_pred": lovo_pred.astype(str), "lovo_proba": lovo_proba}


def resample(Z, K=10):
    """Stretch/squeeze each clip's real frames to exactly K frames by linear
    interpolation, so frame k always means 'k/(K-1) of the way through the punch'.
    Assumes real frames come first (checked in normalise)."""
    lengths = (np.abs(Z).sum(axis=(2, 3)) > 0).sum(axis=1)           # (N,)
    pos = np.linspace(0, 1, K)[None, :] * (lengths[:, None] - 1)     # (N, K) fractional frame index
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, lengths[:, None] - 1)
    w = (pos - lo)[:, :, None, None]
    n = np.arange(len(Z))[:, None]
    return Z[n, lo] * (1 - w) + Z[n, hi] * w                         # (N, K, 17, 2)


def time_features(X, K=10, velocity=True, duration=True):
    """B4 features: normalise -> resample to K frames [-> + velocities] [-> + duration]."""
    lengths = (np.abs(X).sum(axis=(2, 3)) > 0).sum(axis=1)
    R = resample(normalise(X, "per_frame"), K)
    parts = [R.reshape(len(R), -1)]                                  # K*17*2 positions
    if velocity:
        parts.append(np.diff(R, axis=1).reshape(len(R), -1))         # (K-1)*17*2 velocities
    if duration:
        parts.append(lengths[:, None].astype(float))                 # original real-frame count
    return np.hstack(parts)


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]  # e.g. `python baselines.py B4` runs only experiments whose name contains "B4"

    X, y, video_id, _ = load_dataset()
    X_orth = to_orthodox(X, video_id)
    print(f"{len(y)} clips, {len(np.unique(video_id))} videos")

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    norm = lambda X: flatten(normalise(X, "per_frame"))
    experiments = {  # name: (model, features, data, mirror-augment training data?)
        "B0 always-majority": (DummyClassifier(strategy="most_frequent"), flatten, X, False),
        "B1 raw + RF": (rf, flatten, X, False),
        "B2 norm per_frame + RF": (rf, norm, X, False),
        "B2 norm first + RF": (rf, lambda X: flatten(normalise(X, "first")), X, False),
        "B3 norm + mirror + RF": (rf, norm, X, True),
        "B3b orthodox-canon + RF": (rf, norm, X_orth, False),
        # B4: time alignment, all on orthodox data (B3b is the reference)
        "B4a B3b + resample": (
            rf, lambda X: time_features(X, velocity=False, duration=False), X_orth, False),
        "B4b B3b + resample + duration": (
            rf, lambda X: time_features(X, velocity=False), X_orth, False),
        "B4c B3b + resample + duration + velocity": (
            rf, time_features, X_orth, False),
    }
    selected = [n for n in experiments if not only or any(o in n for o in only)]
    progress.start("baselines.py", total=len(selected) * FITS_PER_EVAL)
    results = {}
    for name in selected:
        model, feats, data, aug = experiments[name]
        results[name] = evaluate(name, model, feats, data, y, video_id, augment=aug)
    progress.finish()

    print("\n=== Summary ===")
    print(pd.DataFrame({k: {"random 80/20": v["random"], "LOVO mean": v["lovo_mean"],
                            "LOVO worst": v["lovo"].min()}
                        for k, v in results.items()}).T.map("{:.1%}".format).to_string())
