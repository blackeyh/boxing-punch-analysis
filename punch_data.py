"""Load the boxing punch skeleton dataset into one clean (X, y, video_id) set.

X        : (N, 25, 17, 2) float32 — 25 frames, 17 COCO keypoints, (x, y) in [0, 1].
           Frames after the punch ends are zero padding.
lengths  : (N,) number of real (non-padding) frames in each clip.
y        : (N,) punch label strings, one of CLASSES.
video_id : (N,) int 1..10, the source video of each clip.

Usage:
    from punch_data import load_dataset
    X, y, video_id, lengths = load_dataset()
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
T = 25  # frames per clip
CLASSES = ["Jab", "Cross", "Lead Hook", "Rear Hook", "Lead Uppercut", "Rear Uppercut"]

# COCO-17 keypoint order, in case you need names.
KEYPOINTS = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]


def _read_annotations(vid: int) -> pd.DataFrame:
    """Return a DataFrame with start, end, label, whatever the sheet's layout."""
    df = pd.read_excel(ROOT / "Annotation_files" / f"V{vid}.xlsx")
    numeric = [c for c in df.columns
               if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any()]
    text = [c for c in df.columns if df[c].dtype == object and df[c].notna().any()]
    out = pd.DataFrame({
        "start": df[numeric[0]],   # first two filled numeric columns are start/end
        "end": df[numeric[1]],
        "label": df[text[0]].astype(str).str.strip().str.title()
                 .where(df[text[0]].notna()),
    })
    return out


def _clips_from_stream(stream: np.ndarray, ann: pd.DataFrame) -> np.ndarray:
    """V6 is a continuous stream (frames, 1, 17, 3) with frame number in [..., 0].
    Cut each annotated [start, end] (inclusive) into a padded 25-frame clip."""
    frames = stream[:, 0, 0, 0].astype(int)
    row_of = {f: i for i, f in enumerate(frames)}
    xy = stream[:, 0, :, 1:]  # (frames, 17, 2)
    clips = np.zeros((len(ann), T, 17, 2))
    for k, (s, e) in enumerate(zip(ann["start"], ann["end"])):
        rows = [row_of[f] for f in range(int(s), int(e) + 1) if f in row_of][:T]
        clips[k, :len(rows)] = xy[rows]
    return clips


def load_dataset(min_frames: int = 3, verbose: bool = False):
    Xs, ys, vids = [], [], []
    for vid in range(1, 11):
        ann = _read_annotations(vid)
        arr = np.load(ROOT / "Skeleton_data" / f"V{vid}.npy")
        if vid == 6:
            ann = ann.dropna(subset=["start", "end", "label"]).reset_index(drop=True)
            arr = _clips_from_stream(arr, ann)
        assert len(arr) == len(ann), f"V{vid}: {len(arr)} clips vs {len(ann)} rows"

        dur = ann["end"] - ann["start"]
        ok = ann["label"].isin(CLASSES) & (dur > 0) & (dur < T)
        if verbose:
            print(f"V{vid}: {len(ann)} rows, dropped {(~ok).sum()} bad annotations")
        Xs.append(arr[ok.values])
        ys.append(ann.loc[ok, "label"].values)
        vids.append(np.full(ok.sum(), vid))

    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys)
    video_id = np.concatenate(vids)
    lengths = (np.abs(X).sum(axis=(2, 3)) > 0).sum(axis=1)

    keep = lengths >= min_frames
    return X[keep], y[keep], video_id[keep], lengths[keep]


if __name__ == "__main__":
    X, y, video_id, lengths = load_dataset(verbose=True)
    print("X", X.shape, "| clips per video:", np.bincount(video_id)[1:])
    print(pd.Series(y).value_counts().to_string())
    print("clip length (frames): median", np.median(lengths), "max", lengths.max())
