"""3D version of the boxing punch dataset, row-for-row aligned with punch_data.load_dataset().

Built from the original YouTube videos with MediaPipe Pose Landmarker (heavy); see
video3d/REPORT.md for how it was made and verified.

    from punch_data3d import load_dataset_3d
    X3d, X2d_new, conf, y, video_id, lengths = load_dataset_3d()

X3d      : (N, 25, 17, 3) float32  MediaPipe WORLD landmarks in metres, COCO-17 order.
           Origin = midpoint of the hips (per frame), x = person-image right, y = DOWN,
           z = depth (negative = towards the camera). Zero where nothing was detected
           and in the padding frames.
X2d_new  : (N, 25, 17, 2) float32  MediaPipe 2D for the same frames, x/width, y/height
           (same convention as Skeleton_data). Zero where nothing was detected.
conf     : (N, 25, 17) float32     MediaPipe per-joint visibility in [0, 1] (0 = no detection).
y, video_id, lengths               identical to punch_data.load_dataset() (asserted).

Row i is the same clip as row i of punch_data.load_dataset(): same punch, same original
frames (start..end inclusive), real frames first, zero padding after.

return_meta=True adds a 7th value, a dict of (N, ...) arrays:
  X2d_orig     the dataset's original 2D (== punch_data X)
  frame_ok     (N, 25) bool: detected AND 2D within 0.5 torso length of the original 2D
  match_err_px (N, 25) 2D error vs the original skeleton, pixels of the 1280x720 video
  video_frame  (N, 25) 0-based frame index in the downloaded video (-1 = padding)
  presence     (N, 25, 17) MediaPipe presence;  person (N, 25) which detected person was used
  aligned      (N,) bool: the clip's video passed the alignment check

Videos that FAILED alignment (see video3d/REPORT.md) are present as all-zero rows with
aligned=False; pass drop_unaligned=True to remove them.
"""
from pathlib import Path

import numpy as np

import punch_data as P

ROOT = Path(__file__).resolve().parent
D3 = ROOT / "Skeleton3D_data"


def _load_video(vid):
    X3 = np.load(D3 / f"V{vid}.npy")
    X2 = np.load(D3 / f"V{vid}_2d.npy")
    cf = np.load(D3 / f"V{vid}_conf.npy")
    m = dict(np.load(D3 / f"V{vid}_meta.npz"))
    return X3, X2, cf, m


def load_dataset_3d(min_frames: int = 3, verbose: bool = False, return_meta: bool = False,
                    drop_unaligned: bool = False):
    """drop_unaligned=True removes the rows of videos that failed alignment (their 3D is all
    zeros). Rows then match punch_data.load_dataset() with those videos removed."""
    parts = {k: [] for k in ["X3", "X2", "cf", "X2o", "y", "vid", "ok", "err", "vf", "pres", "person", "aligned"]}
    for vid in range(1, 11):
        # identical row selection to punch_data.load_dataset
        ann = P._read_annotations(vid)
        arr = np.load(P.ROOT / "Skeleton_data" / f"V{vid}.npy")
        if vid == 6:
            ann = ann.dropna(subset=["start", "end", "label"]).reset_index(drop=True)
            arr = P._clips_from_stream(arr, ann)
        dur = ann["end"] - ann["start"]
        ok = (ann["label"].isin(P.CLASSES) & (dur > 0) & (dur < P.T)).values
        if not (D3 / f"V{vid}.npy").exists():
            raise FileNotFoundError(f"Skeleton3D_data/V{vid}.npy missing: run video3d/build_dataset3d.py")
        X3, X2, cf, m = _load_video(vid)
        assert len(X3) == len(arr) == len(ann), f"V{vid}: 3D rows {len(X3)} vs 2D rows {len(arr)}"
        parts["X3"].append(X3[ok]); parts["X2"].append(X2[ok]); parts["cf"].append(cf[ok])
        parts["X2o"].append(arr[ok]); parts["y"].append(ann.loc[ok, "label"].values)
        parts["vid"].append(np.full(ok.sum(), vid)); parts["ok"].append(m["frame_ok"][ok])
        parts["err"].append(m["match_err_px"][ok]); parts["vf"].append(m["video_frame"][ok])
        parts["pres"].append(m["presence"][ok]); parts["person"].append(m["person"][ok])
        parts["aligned"].append(np.full(ok.sum(), bool(m["aligned"])))
        if verbose:
            real = m["video_frame"][ok] >= 0
            print(f"V{vid}: {ok.sum()} clips, aligned={bool(m['aligned'])}, "
                  f"frames ok {m['frame_ok'][ok][real].mean() if real.any() else 0:.1%}")
    c = {k: np.concatenate(v) for k, v in parts.items()}
    X2o = c["X2o"].astype(np.float32)
    lengths = (np.abs(X2o).sum(axis=(2, 3)) > 0).sum(axis=1)
    keep = lengths >= min_frames

    # row-match assertion against the 2D loader
    X_ref, y_ref, vid_ref, len_ref = P.load_dataset(min_frames=min_frames)
    assert np.array_equal(c["y"][keep], y_ref), "labels do not match punch_data.load_dataset()"
    assert np.array_equal(c["vid"][keep], vid_ref), "video ids do not match punch_data.load_dataset()"
    assert np.array_equal(lengths[keep], len_ref), "clip lengths do not match"
    assert np.array_equal(X2o[keep], X_ref), "original 2D does not match"
    # padding frames are zero in 3D too
    vf = c["vf"][keep]
    assert not np.abs(c["X3"][keep][vf < 0]).any(), "non-zero 3D in padding frames"

    al = c["aligned"][keep]
    r = al if drop_unaligned else np.ones(len(al), bool)
    out = (c["X3"][keep][r], c["X2"][keep][r], c["cf"][keep][r], y_ref[r], vid_ref[r], len_ref[r])
    if return_meta:
        meta = dict(X2d_orig=X_ref[r], frame_ok=c["ok"][keep][r], match_err_px=c["err"][keep][r],
                    video_frame=vf[r], presence=c["pres"][keep][r], person=c["person"][keep][r],
                    aligned=al[r])
        out = out + (meta,)
    return out


if __name__ == "__main__":
    X3, X2, conf, y, vid, lengths, meta = load_dataset_3d(verbose=True, return_meta=True)
    real = meta["video_frame"] >= 0
    print(f"rows {len(y)}, aligned rows {meta['aligned'].sum()}")
    print("X3d", X3.shape, "X2d_new", X2.shape, "conf", conf.shape, "| rows match punch_data: OK")
    print(f"real frames {real.sum()}, frame_ok {meta['frame_ok'][real].mean():.1%}, "
          f"median 2D error {np.nanmedian(meta['match_err_px'][real]):.1f} px")
