"""Shared helpers for the video3d scripts (alignment, dataset build, QC)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import punch_data as P  # noqa: E402

RAW = HERE / "raw"
# Dataset video id -> downloaded video file stem (video3d/raw/videos/<stem>.mp4; V{i}.mp4 was
# downloaded from the link listed for V{i} in RGB_videos/Meta_data.ods). Two metadata errors:
#  * the links for V2 and V3 are SWAPPED: dataset V2 matches the V3 link's video and vice versa;
#  * dataset V1 does not come from the V1 link at all; its true source is kbgkeTTSau8 (V1_true.mp4).
# Evidence: crossmatch.py, find_source.py, align.py; see REPORT.md.
VIDEO_OF = {v: f"V{v}" for v in range(1, 11)}
VIDEO_OF.update({1: "V1_true", 2: "V3", 3: "V2"})
# MediaPipe 33 -> COCO-17 (COCO order: nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_el, r_el,
# l_wr, r_wr, l_hip, r_hip, l_knee, r_knee, l_ank, r_ank)
MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
# joints used to measure 2D agreement: shoulders, elbows, wrists, hips (COCO 5..12).
# Face points are defined differently by different estimators; knees/ankles are often
# out of frame (both estimators then extrapolate), so they are reported but not used to align.
ALIGN_J = list(range(5, 13))
SWAP_LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def load_pose(v):
    """MediaPipe dump for downloaded video FILE v (not dataset id: use VIDEO_OF), reduced to COCO-17. Returns dict with
    xy (F,K,17,2) image-normalised, vis (F,K,17), pres (F,K,17), world (F,K,17,3), npose, W, H, fps."""
    d = np.load(RAW / "pose" / f"{v if isinstance(v, str) else f'V{v}'}.npz")
    img = d["img"][:, :, MP2COCO]
    return dict(xy=img[..., :2], vis=img[..., 3], pres=img[..., 4],
                world=d["world"][:, :, MP2COCO], npose=d["npose"], pts_ms=d["pts_ms"],
                W=int(d["width"]), H=int(d["height"]), fps=float(d["fps"]),
                mp_version=str(d["mediapipe_version"]), model_sha256=str(d["model_sha256"]))


def annotations(v):
    """Annotation rows exactly as punch_data.load_dataset sees them (before its filtering)."""
    ann = P._read_annotations(v)
    if v == 6:
        ann = ann.dropna(subset=["start", "end", "label"]).reset_index(drop=True)
    return ann


def clip_frames(v, ann):
    """For each annotation row, the list of ORIGINAL frame numbers of the clip's real frames,
    in the exact order punch_data puts them in the (25, 17, 2) clip. Also returns the 2D clips."""
    arr = np.load(ROOT / "Skeleton_data" / f"V{v}.npy")
    if v == 6:
        frames = arr[:, 0, 0, 0].astype(int)
        have = set(frames.tolist())
        fl = [[f for f in range(int(s), int(e) + 1) if f in have][:P.T]
              for s, e in zip(ann["start"], ann["end"])]
        clips = P._clips_from_stream(arr, ann)
    else:
        clips = arr
        real = np.abs(arr).sum(axis=(2, 3)) > 0
        fl = []
        for i, s in enumerate(ann["start"]):
            n = int(real[i].sum())
            fl.append([int(s) + k for k in range(n)] if n and np.isfinite(s) else [])
    return fl, clips


def dataset_frame_map(v):
    """{original frame number: (17, 2) dataset keypoints} over all annotated (real) frames."""
    ann = annotations(v)
    fl, clips = clip_frames(v, ann)
    m = {}
    for i, frames in enumerate(fl):
        for k, f in enumerate(frames):
            m.setdefault(f, clips[i, k])
    return m


def stream_frame_map():
    """V6 only: the full continuous stream (not just annotated frames)."""
    arr = np.load(ROOT / "Skeleton_data" / "V6.npy")
    return {int(f): arr[i, 0, :, 1:] for i, f in enumerate(arr[:, 0, 0, 0])}


def joint_err(ds, est, W, H, joints=ALIGN_J):
    """Per-candidate mean joint distance. ds (..., 17, 2), est (..., K, 17, 2).
    Returns (normalised units, pixels) each of shape (..., K); NaN where no candidate."""
    d = est[..., joints, :] - ds[..., None, joints, :]
    norm = np.sqrt((d ** 2).sum(-1)).mean(-1)
    px = np.sqrt((d[..., 0] * W) ** 2 + (d[..., 1] * H) ** 2).mean(-1)
    return norm, px
