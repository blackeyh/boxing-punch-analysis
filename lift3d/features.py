"""3D features for the classifier, from lifted Human3.6M-17 skeletons.

rotate_to_frontal: the key idea. Using the FIRST frame's hip line (left hip - right hip),
rotate the whole clip about the vertical (camera y) axis so the hips are parallel to the
image plane, i.e. every fighter faces the camera. A side-on jab then becomes the same
forward (depth) movement as a frontal jab. The first frame (the guard) is used so the
punch's own body rotation is kept.

H36M joints: 0 pelvis, 1 R hip, 4 L hip, 8 thorax (see lift.py for the full list).
"""
from pathlib import Path

import numpy as np

from baselines import resample

L_HIP, R_HIP, PELVIS, THORAX = 4, 1, 0, 8
LIFTED = Path(__file__).resolve().parent / "lifted_h36m.npz"


def real_lengths(X3):
    return (np.abs(X3).sum(axis=(2, 3)) > 0).sum(axis=1)


# joint indices per skeleton layout: (left hip, right hip, pelvis-or-None, thorax-or-shoulders)
LAYOUTS = {"h36m": dict(lhip=4, rhip=1, pelvis=[0], thorax=[8]),
           "coco": dict(lhip=11, rhip=12, pelvis=[11, 12], thorax=[5, 6])}   # MediaPipe -> COCO-17


def rotate_to_frontal(X3, layout="h36m"):
    """Rotate each clip about the y axis so its first-frame hip line points along +x."""
    J = LAYOUTS[layout]
    hip = X3[:, 0, J["lhip"]] - X3[:, 0, J["rhip"]]          # (N, 3)
    theta = np.arctan2(hip[:, 2], hip[:, 0])                  # angle of the hip line in the x-z plane
    c, s = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    out = X3.copy()
    x, z = X3[..., 0], X3[..., 2]
    out[..., 0] = x * c + z * s                               # rotate by -theta
    out[..., 2] = -x * s + z * c
    return out


def load_lifted(y, video_id):
    """Load lifted clips and check they line up row-for-row with punch_data.load_dataset()."""
    d = np.load(LIFTED)
    assert (d["y"] == np.asarray(y).astype(str)).all() and (d["video_id"] == video_id).all(), \
        "lifted file doesn't match the dataset rows; rerun lift3d/lift.py"
    assert "RANDOM" not in str(d["checkpoint"]), "lifted file was made with random weights"
    return d["X3"].astype(np.float32)


def smooth_time(X3, lengths, w=3):
    """Moving average over w frames within each clip's real frames (edges use the frames
    available). Reduces the frame-to-frame jitter of per-frame lifting."""
    out = X3.copy()
    h = w // 2
    for i, L in enumerate(lengths):
        seq = X3[i, :L]
        out[i, :L] = np.stack([seq[max(0, t - h): t + h + 1].mean(0) for t in range(L)])
    return out


def fill_low_conf(X3, conf, thr=0.5):
    """Replace each joint's 3D position on frames where its confidence < thr by linear
    interpolation (in time) from the same joint's confident frames in that clip; edges take
    the nearest confident value. A joint never confident in a clip is left unchanged.
    Returns (filled X3, fraction of real joint-frames replaced)."""
    lengths = real_lengths(X3)
    out = X3.copy(); n_rep = n_tot = 0
    for i, L in enumerate(lengths):
        t = np.arange(L)
        for j in range(X3.shape[2]):
            good = conf[i, :L, j] >= thr
            n_tot += L
            if good.all() or not good.any():
                continue
            n_rep += (~good).sum()
            for c in range(3):
                out[i, :L, j, c] = np.interp(t, t[good], X3[i, :L, j, c][good])
    return out, n_rep / max(n_tot, 1)


def seq_features_3d(X3, rotate=True, K=10, smooth=0, layout="h36m", conf=None):
    """(N, 25, 17, 3) 3D -> (N, K, 102 [+17]): per frame 51 positions + 51 velocities
    [+ 17 joint confidences if `conf` is given]. Root (pelvis) at the origin, scaled by the
    clip's median torso length, optionally smoothed over `smooth` frames and rotated to
    frontal, resampled to K frames."""
    lengths = real_lengths(X3)
    if smooth > 1:
        X3 = smooth_time(X3, lengths, smooth)
    J = LAYOUTS[layout]
    Z = X3 - X3[:, :, J["pelvis"]].mean(axis=2, keepdims=True)   # pelvis (or hip midpoint) at origin
    torso = np.linalg.norm(Z[:, :, J["thorax"]].mean(axis=2), axis=-1)
    real = np.arange(Z.shape[1])[None, :] < lengths[:, None]
    scale = np.nanmedian(np.where(real, torso, np.nan), axis=1)
    Z = Z / np.maximum(scale, 1e-3)[:, None, None, None]
    Z[~real] = 0
    if rotate:
        Z = rotate_to_frontal(Z, layout)
        Z[~real] = 0
    # resample() finds real frames by non-zero rows; the pelvis is 0 but other joints aren't
    if conf is not None:      # carry confidence along as a 4th coordinate so it is resampled identically
        Z4 = np.concatenate([Z, np.where(real[:, :, None, None], conf[..., None], 0)], axis=3)
        R4 = resample(Z4, K)
        R, C = R4[..., :3].reshape(len(Z), K, -1), R4[..., 3]   # (N, K, 51), (N, K, 17)
    else:
        R = resample(Z, K).reshape(len(Z), K, -1)                 # (N, K, 51)
    V = np.diff(R, axis=1, prepend=R[:, :1])
    parts = [R, V] + ([C] if conf is not None else [])
    return np.concatenate(parts, axis=2).astype(np.float32)


def load_mediapipe3d(y, video_id, with_conf=False):
    """MediaPipe 3D from the original videos (punch_data3d, merged from branch video3d),
    COCO-17 order, metres, y down. V4 (genuine southpaw) is mirrored to orthodox in 3D
    exactly like the 2D pipeline: negate x and swap left/right joints."""
    from baselines import MIRROR_IDX, SOUTHPAW_VIDEOS
    from punch_data3d import load_dataset_3d
    X3, _, conf, y3, v3, _ = load_dataset_3d()
    conf = conf.astype(np.float32).copy()
    assert (np.asarray(y3) == np.asarray(y)).all() and (v3 == video_id).all(), "rows don't match"
    X3 = X3.astype(np.float32).copy()
    for vid in SOUTHPAW_VIDEOS:
        m = v3 == vid
        Xm = X3[m][:, :, MIRROR_IDX]
        real = np.abs(Xm).sum(axis=(2, 3)) > 0
        Xm[..., 0] = np.where(real[:, :, None], -Xm[..., 0], 0.0)
        X3[m] = Xm
        conf[m] = conf[m][:, :, MIRROR_IDX]          # confidences follow their joints
    return (X3, conf) if with_conf else X3
