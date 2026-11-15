"""Build review/punch_review.html: the error-analysis page, with all data embedded.

Inputs : runs/lovo_predictions/preds.npz (from lovo_predictions.py) + the dataset
Output : review/punch_review.html (template.html with the data filled in)

Skeletons are shown as the model saw them, stance-wise (V4 mirrored to orthodox), and
normalised with mode="first": hip midpoint of the FIRST frame at (0,0), torso length = 1,
so movement during the punch stays visible. Only real frames, no padding.
Coordinates are packed as int8 (value * 40), proba as uint8 (value * 255).
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from baselines import normalise, to_orthodox  # noqa: E402
from punch_data import CLASSES, load_dataset  # noqa: E402

SCALE = 40

X, y, video_id, lengths = load_dataset()
P = np.load(ROOT / "runs/lovo_predictions/preds.npz", allow_pickle=False)
assert (P["y_true"] == y).all() and (P["video_id"] == video_id).all(), "predictions don't match dataset"
Z = normalise(to_orthodox(X, video_id), "first")

ci = {c: i for i, c in enumerate(CLASSES)}
y_idx = np.array([ci[c] for c in y]); p_idx = np.array([ci[c] for c in P["y_pred"]])
proba = P["proba"]

frames = [Z[i, :lengths[i]].reshape(lengths[i], 34) for i in range(len(y))]
offsets = np.concatenate([[0], np.cumsum(lengths)])[:-1]
coords = np.clip(np.round(np.concatenate(frames) * SCALE), -127, 127).astype(np.int8)
pq = np.clip(np.round(proba * 255), 0, 255).astype(np.uint8)

videos = []
for v in np.unique(video_id):
    m = video_id == v
    videos.append({"id": int(v), "n": int(m.sum()), "wrong": int((y_idx[m] != p_idx[m]).sum()),
                   "acc": round(float(balanced_accuracy_score(y_idx[m], p_idx[m])), 4)})

data = {
    "classes": CLASSES, "scale": SCALE, "videos": videos,
    "lovo_mean": round(float(np.mean([v["acc"] for v in videos])), 4),
    "video": video_id.astype(int).tolist(), "y": y_idx.tolist(), "p": p_idx.tolist(),
    "len": lengths.astype(int).tolist(), "off": offsets.astype(int).tolist(),
    "coords": base64.b64encode(coords.tobytes()).decode(),
    "proba": base64.b64encode(pq.tobytes()).decode(),
}
tpl = (ROOT / "review/template.html").read_text()
out = ROOT / "review/punch_review.html"
out.write_text(tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))
print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(y)} clips, {int((y_idx != p_idx).sum())} wrong)")
