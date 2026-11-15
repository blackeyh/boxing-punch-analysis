"""Shared window definition (detect/WINDOWS.md) for the pixel-based punch detector.

Windows: W=16 frames, stride S=4, starts 0,4,8,... while start+16 <= n_frames, frames of the
ORIGINAL downloaded video (VIDEO_OF mapping). Labels are for EVALUATION / probe training only;
the detector's input is pixels only.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import punch_data as P  # noqa: E402

W, S = 16, 4
VIDEO_OF = {v: f"V{v}" for v in range(1, 11)}
VIDEO_OF.update({1: "V1_true", 2: "V3", 3: "V2"})
MANIFEST = json.loads((ROOT / "video3d" / "manifest.json").read_text())["videos"]


def n_frames(v):
    return int(MANIFEST[VIDEO_OF[v]]["frame_count"])


def fps(v):
    return float(MANIFEST[VIDEO_OF[v]]["fps"])


def starts(v):
    return np.arange(0, n_frames(v) - W + 1, S, dtype=np.int32)


def punches(v):
    """Labelled punches kept by the WINDOWS.md rule: valid label, 0 < end-start < 25.
    Returns DataFrame start, end (inclusive), label. Same rows as video3d/common.annotations(v)
    filtered like punch_data.load_dataset (its extra >=3-real-skeleton-frames filter is skeleton
    specific and NOT applied: a punch is a punch in the pixels whether or not pose was found)."""
    ann = P._read_annotations(v)
    if v == 6:
        ann = ann.dropna(subset=["start", "end", "label"]).reset_index(drop=True)
    dur = ann["end"] - ann["start"]
    ok = ann["label"].isin(P.CLASSES) & (dur > 0) & (dur < P.T)
    a = ann[ok].copy()
    a["start"] = a["start"].astype(int)
    a["end"] = a["end"].astype(int)
    return a.reset_index(drop=True)


def window_labels(v):
    """Per window: overlap (fraction of its 16 frames inside any punch), is_punch (contains
    >= 50% of some punch's frames), is_gap (overlap 0 and >= 8 frames from any punch),
    and the label of the punch it covers most (or '')."""
    st = starts(v)
    n = n_frames(v)
    a = punches(v)
    inside = np.zeros(n + 1, bool)
    near = np.zeros(n + 1, bool)          # punch frames dilated by 8 on each side
    for s, e in zip(a["start"], a["end"]):
        inside[s:e + 1] = True
        near[max(0, s - 8):e + 9] = True
    cs_in = np.concatenate([[0], np.cumsum(inside)])
    cs_near = np.concatenate([[0], np.cumsum(near)])
    overlap = (cs_in[st + W] - cs_in[st]) / W
    near_cnt = cs_near[st + W] - cs_near[st]
    is_gap = (overlap == 0) & (near_cnt == 0)
    best = np.zeros(len(st))
    lab = np.array([""] * len(st), dtype=object)
    for s, e, l in zip(a["start"], a["end"], a["label"]):
        L = e - s + 1
        i0 = max(0, (s - W + 1 + S - 1) // S)          # first window that can touch the punch
        i1 = min(len(st) - 1, e // S)
        for i in range(i0, i1 + 1):
            ws = st[i]
            frac = max(0, min(e, ws + W - 1) - max(s, ws) + 1) / L
            if frac > best[i]:
                best[i], lab[i] = frac, l
    is_punch = best >= 0.5
    return dict(start=st, overlap=overlap, is_punch=is_punch, is_gap=is_gap, label=lab, cover=best)


if __name__ == "__main__":
    tot = np.zeros(4, int)
    for v in range(1, 11):
        d = window_labels(v)
        r = np.array([len(d["start"]), d["is_punch"].sum(), d["is_gap"].sum(), len(punches(v))])
        tot += r
        print(f"V{v}: windows {r[0]}, punch {r[1]}, gap {r[2]}, punches {r[3]}")
    print("total", tot)
