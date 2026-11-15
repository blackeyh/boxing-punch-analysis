"""Build the 7-class dataset (6 punch types + no_punch), all from the SAME MediaPipe frame data.

Every clip, punch or not, is cut from detect/frames/V{v}.npz (tracked fighter, V4 mirrored, 3D
conf-filled), so the model cannot tell classes apart by data source.

Punch clips: every annotation row kept by punch_data.load_dataset() (start..end inclusive).
No-punch clips (the agreed strict rule, 2026-11-13):
  a 16-frame window is a no-punch SOURCE window only if
    * it is a gap window: >= 8 frames from any labelled punch,
    * the person is present in >= 90% of its frames,
    * detector A (video probe) score < 0.05 AND detector B (skeleton PU) score < 0.2,
    * it is >= 8 frames from any window flagged by A (>= 0.5) or B (>= 0.5).
  From each source window one clip is cut, with a length drawn from that video's labelled-punch
  length distribution (so length can't give the class away) and a random position inside it.
  Per video at most as many no-punch clips as labelled punches (keeps classes comparable).
Hidden punches (A >= 0.9 in a gap window) are listed separately for later use, not trained on.

Outputs detect/dataset7/:
  clips.npz  xy (N,25,17,2) f16, world (N,25,17,3) f16, vis (N,25,17) f16, label (N,) str,
             video (N,), start (N,), length (N,), source ("annotation" | "nopunch_rule")
  hidden_punches.json, summary.json
    /opt/anaconda3/bin/python detect/build_7class.py [--a-scores DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from punch_data import CLASSES, T  # noqa: E402
from video3d.common import annotations  # noqa: E402

FR = ROOT / "detect" / "frames"
B_DIR = ROOT / "detect" / "scores" / "pu_skeleton"
W, S, MARGIN = 16, 4, 8
A_NEG, B_NEG = 0.02, 1.01          # final detector A (2026-11-13): A < 0.02 kept 24 user-reviewed moments, 0 punches
A_FLAG, B_FLAG = 0.1, 0.5           # A >= 0.1: 33/34 agreed with the user (97%); B >= 0.5 still excluded nearby


def window_scores(path, n_starts):
    d = np.load(path)
    sc = np.full(n_starts, np.nan, np.float32)
    idx = d["start"] // S
    sc[idx[idx < n_starts]] = d["score"][: (idx < n_starts).sum()]
    return sc


def cut(arr, s, L):
    out = np.zeros((T,) + arr.shape[1:], arr.dtype)
    out[:L] = arr[s:s + L]
    return out


def fill_missing(xy, world, vis, length):
    """Frames inside a clip where MediaPipe found nobody (all-zero) get the values of the nearest
    frame of the same clip where the fighter WAS found (0.04% of frames, 30 of 8,896 clips)."""
    n_fixed = 0
    for i, L in enumerate(length):
        empty = np.abs(xy[i, :L]).sum(axis=(1, 2)) == 0
        if not empty.any() or empty.all():
            continue
        good = np.where(~empty)[0]
        for t in np.where(empty)[0]:
            g = good[np.argmin(np.abs(good - t))]
            xy[i, t], world[i, t], vis[i, t] = xy[i, g], world[i, g], vis[i, g]
        n_fixed += 1
    return n_fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-scores", default=str(ROOT / "detect" / "scores" / "videomae_probe"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = ROOT / "detect" / "dataset7"; out.mkdir(parents=True, exist_ok=True)
    clips = {k: [] for k in ["xy", "world", "vis", "label", "video", "start", "length", "source"]}
    hidden, summary = [], {}

    def add(fr, v, s, L, lab, src):
        clips["xy"].append(cut(fr["xy"], s, L)); clips["world"].append(cut(fr["world"], s, L))
        clips["vis"].append(cut(fr["vis"], s, L)); clips["label"].append(lab); clips["video"].append(v)
        clips["start"].append(s); clips["length"].append(L); clips["source"].append(src)

    for v in range(1, 11):
        fr = dict(np.load(FR / f"V{v}.npz"))
        F = len(fr["label"]); lab = fr["label"] >= 0; pres = fr["present"]
        # punch clips from the annotations (same rows as punch_data.load_dataset)
        ann = annotations(v); dur = ann["end"] - ann["start"]
        keep = ann["label"].isin(CLASSES) & (dur > 0) & (dur < T)
        lens = []
        for _, r in ann[keep].iterrows():
            s, e = int(r["start"]), int(r["end"]); L = min(e - s + 1, T, F - s)
            if L >= 3:
                add(fr, v, s, L, r["label"], "annotation"); lens.append(L)
        n_punch = len(lens)

        starts = np.arange(0, F - W + 1, S)
        a = window_scores(Path(args.a_scores) / f"V{v}.npz", len(starts))
        b = window_scores(B_DIR / f"V{v}.npz", len(starts))
        near_lab = np.convolve(lab.astype(float), np.ones(2 * MARGIN + 1), "same") > 0
        gap = np.array([not near_lab[s:s + W].any() for s in starts])
        pres_w = np.array([pres[s:s + W].mean() for s in starts])
        flagged = (np.nan_to_num(a) >= A_FLAG) | (np.nan_to_num(b) >= B_FLAG)
        flag_frames = np.zeros(F, bool)
        for s in starts[flagged]:
            flag_frames[max(0, s - MARGIN): s + W + MARGIN] = True
        clean = gap & (pres_w >= 0.9) & (a < A_NEG) & (b < B_NEG) & \
            np.array([not flag_frames[s:s + W].any() for s in starts])
        for s in starts[gap & (np.nan_to_num(a) >= A_FLAG)]:
            hidden.append(dict(video=v, start=int(s), score_A=round(float(a[s // S]), 3), score_B=round(float(b[s // S]), 3)))

        # one no-punch clip per selected source window, non-overlapping, length ~ punch lengths
        cand = starts[clean]; rng.shuffle(cand)
        taken, n_np = [], 0
        for s in cand:
            if n_np >= n_punch:
                break
            if any(abs(s - t) < W for t in taken):
                continue
            L = int(rng.choice(lens)); L = min(L, W)
            off = int(rng.integers(0, W - L + 1))
            add(fr, v, int(s + off), L, "No Punch", "nopunch_rule"); taken.append(s); n_np += 1
        summary[f"V{v}"] = dict(frames=F, punch_clips=n_punch, gap_windows=int(gap.sum()),
                                clean_windows=int(clean.sum()), nopunch_clips=n_np,
                                flagged_gap_windows=int((gap & flagged).sum()),
                                hidden_punch_windows=int((gap & (np.nan_to_num(a) >= A_FLAG)).sum()))
        print(f"V{v}: " + ", ".join(f"{k} {x}" for k, x in summary[f'V{v}'].items()), flush=True)

    XY, WD, VS = np.stack(clips["xy"]), np.stack(clips["world"]), np.stack(clips["vis"])
    print("clips with missing-person frames filled from the nearest frame:", fill_missing(XY, WD, VS, np.array(clips["length"])))
    np.savez_compressed(out / "clips.npz", xy=XY.astype(np.float16),
                        world=WD.astype(np.float16), vis=VS.astype(np.float16),
                        label=np.array(clips["label"]), video=np.array(clips["video"]), start=np.array(clips["start"]),
                        length=np.array(clips["length"]), source=np.array(clips["source"]))
    (out / "hidden_punches.json").write_text(json.dumps(hidden, indent=0))
    summary["rule"] = dict(A_neg=A_NEG, B_neg=B_NEG, A_flag=A_FLAG, B_flag=B_FLAG, margin=MARGIN, a_scores=args.a_scores)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    labs, cnt = np.unique(clips["label"], return_counts=True)
    print("classes:", dict(zip(labs.tolist(), cnt.tolist())), "| hidden-punch windows (A >= A_FLAG):", len(hidden))


if __name__ == "__main__":
    main()
