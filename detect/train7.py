"""Train/evaluate the punch classifier WITH a no-punch class, leave-one-video-out.

Input per clip (all from the same MediaPipe frame data, detect/dataset7/clips.npz):
  MediaPipe 2D (normalised, resampled, + velocities: 68 ch) + MediaPipe 3D (rotated to frontal,
  conf-filled, + velocities + 17 visibility channels: 119 ch) = 187 channels x 10 frames.
Model: the 1D-CNN from deep.py (TorchSeqClassifier), 60 epochs.

Modes:
  --classes 7   6 punches + No Punch   (the detection-capable model)
  --classes 6   punches only            (reference: same MediaPipe-only input, to compare with the
                                          73.7% model that used the dataset's original 2D)
Metrics per seed (LOVO; per-video then averaged):
  bal_acc         balanced accuracy over the trained classes
  type_acc6       balanced accuracy over the 6 punch types on punch clips, argmax over punch classes
                  only (comparable with earlier 6-class numbers)
  miss_rate       share of punch clips predicted "No Punch"          (7-class only)
  false_alarm     share of no-punch clips predicted as any punch     (7-class only)
    python detect/train7.py --classes 7 --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mem_guard  # noqa: F401,E402
import numpy as np  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from deep import TorchSeqClassifier, seq_features  # noqa: E402
from lift3d.features import seq_features_3d  # noqa: E402
from progress import progress  # noqa: E402
from punch_data import CLASSES  # noqa: E402

NP = "No Punch"


def features(d):
    xy = d["xy"].astype(np.float32); world = d["world"].astype(np.float32); vis = d["vis"].astype(np.float32)
    return np.concatenate([seq_features(xy), seq_features_3d(world, rotate=True, layout="coco", conf=vis)], axis=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, choices=[6, 7], default=7)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=os.environ.get("RUN_DIR", str(ROOT / "runs" / "train7")))
    args = ap.parse_args()
    d = np.load(ROOT / "detect" / "dataset7" / "clips.npz")
    keep = np.ones(len(d["label"]), bool) if args.classes == 7 else d["label"] != NP
    F = features(d)[keep]; y = d["label"][keep]; vid = d["video"][keep]
    classes = CLASSES + ([NP] if args.classes == 7 else [])
    print(f"{args.classes}-class: {len(y)} clips, features {F.shape}, classes {dict(zip(*np.unique(y, return_counts=True)))}")
    videos = np.unique(vid)
    progress.start(f"train{args.classes}", total=len(args.seeds) * len(videos))
    runs = []
    for seed in args.seeds:
        t0 = time.time(); proba = np.zeros((len(y), len(classes)), np.float32)
        for v in videos:
            te = vid == v
            m = TorchSeqClassifier("cnn", epochs=args.epochs, seed=seed).fit(F[~te], y[~te])
            p = m.predict_proba(F[te]); order = [list(m.classes_).index(c) for c in classes]
            proba[te] = p[:, order]
            progress.tick(1, note=f"seed {seed} V{v}")
        pred = np.array(classes)[proba.argmax(1)]
        is_p = y != NP
        type_pred = np.array(CLASSES)[proba[:, :6].argmax(1)]          # best punch type, ignoring No Punch
        per = {}
        for v in videos:
            mv = vid == v; mp = mv & is_p
            r = dict(bal_acc=balanced_accuracy_score(y[mv], pred[mv]),
                     type_acc6=balanced_accuracy_score(y[mp], type_pred[mp]))
            if args.classes == 7:
                r["miss_rate"] = float(np.mean(pred[mp] == NP))
                mn = mv & ~is_p
                r["false_alarm"] = float(np.mean(pred[mn] != NP)) if mn.any() else float("nan")
            per[f"V{v}"] = r
        agg = {k: float(np.nanmean([r[k] for r in per.values()])) for k in next(iter(per.values()))}
        runs.append(dict(seed=seed, **agg, per_video=per, seconds=time.time() - t0,
                         lovo_proba=np.round(proba, 4).tolist()))
        print(f"[seed {seed}] " + "  ".join(f"{k} {val:.1%}" for k, val in agg.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    progress.finish()
    keys = [k for k in runs[0] if k not in ("seed", "per_video", "seconds", "lovo_proba")]
    print("==> " + "  ".join(f"{k} {np.mean([r[k] for r in runs]):.1%} ± {np.std([r[k] for r in runs], ddof=1):.1%}" for k in keys))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "results.json").write_text(json.dumps(dict(classes=classes, n=len(y), runs=runs,
                                                                 video=vid.tolist(), label=y.tolist())))


if __name__ == "__main__":
    main()
