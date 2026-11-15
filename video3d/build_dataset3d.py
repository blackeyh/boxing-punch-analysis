"""Cut the MediaPipe per-frame dumps into clips that line up 1:1 with Skeleton_data/.

    python video3d/build_dataset3d.py        # needs video3d/alignment.json (from align.py)

For every annotation row i of video v (V6: rows after dropna, as in punch_data), the clip's
real frames are exactly the original frame numbers punch_data puts in row i (start..end
inclusive; V6 cut as punch_data._clips_from_stream does), each mapped to the video frame
f + offset[v]. Frames past the clip's real length stay zero, like the 2D data.

At each frame the person (MediaPipe returns up to 3) whose shoulders/elbows/wrists/hips are
closest to the dataset's own 2D skeleton for that frame is used.

Writes Skeleton3D_data/:
  V{v}.npy        (N, 25, 17, 3) float32  MediaPipe world landmarks (metres, origin = hip midpoint,
                                           x right, y down, z towards the camera is negative)
  V{v}_2d.npy     (N, 25, 17, 2) float32  MediaPipe image landmarks, x/width, y/height
  V{v}_conf.npy   (N, 25, 17)    float32  MediaPipe per-joint visibility (0..1)
  V{v}_meta.npz   presence (N,25,17), video_frame (N,25) int (-1 = padding),
                  match_err_px (N,25) (NaN = no detection / padding), torso_px (N,),
                  frame_ok (N,25) bool, person (N,25) int8, offset, aligned (bool)
Zero where nothing was detected (frame_ok False, conf 0).
A video that FAILED alignment gets all-zero placeholder files with aligned=False (no pose data).
"""
import json

import numpy as np

from common import ALIGN_J, HERE, ROOT, VIDEO_OF, annotations, clip_frames, joint_err, load_pose

T = 25
OUT = ROOT / "Skeleton3D_data"
OK_TORSO = 0.5   # a frame is "ok" if detected and its 2D error is below half a torso length


def build(v, align):
    P = load_pose(VIDEO_OF[v])
    W, H, F = P["W"], P["H"], P["xy"].shape[0]
    d = align["offset"]
    ann = annotations(v)
    fl, clips = clip_frames(v, ann)
    N = len(fl)
    X3 = np.zeros((N, T, 17, 3), np.float32); X2 = np.zeros((N, T, 17, 2), np.float32)
    conf = np.zeros((N, T, 17), np.float32); pres = np.zeros((N, T, 17), np.float32)
    vf = np.full((N, T), -1, np.int32); err = np.full((N, T), np.nan, np.float32)
    ok = np.zeros((N, T), bool); person = np.full((N, T), -1, np.int8); torso = np.full(N, np.nan, np.float32)
    for i, frames in enumerate(fl):
        if not frames:
            continue
        ds = clips[i, :len(frames)]
        sh = (ds[:, 5] + ds[:, 6]) / 2; hp = (ds[:, 11] + ds[:, 12]) / 2
        torso[i] = np.median(np.hypot((sh - hp)[:, 0] * W, (sh - hp)[:, 1] * H))
        for k, f in enumerate(frames):
            g = f + d
            vf[i, k] = g
            if not (0 <= g < F) or P["npose"][g] == 0:
                continue
            _, px = joint_err(ds[k], P["xy"][g], W, H, ALIGN_J)      # (K,)
            if not np.isfinite(px).any():
                continue
            p = int(np.nanargmin(px))
            person[i, k], err[i, k] = p, px[p]
            X3[i, k] = P["world"][g, p]; X2[i, k] = P["xy"][g, p]
            conf[i, k] = P["vis"][g, p]; pres[i, k] = P["pres"][g, p]
            ok[i, k] = px[p] < OK_TORSO * torso[i]
    OUT.mkdir(exist_ok=True)
    np.save(OUT / f"V{v}.npy", X3)
    np.save(OUT / f"V{v}_2d.npy", X2)
    np.save(OUT / f"V{v}_conf.npy", conf)
    np.savez_compressed(OUT / f"V{v}_meta.npz", presence=pres, video_frame=vf, match_err_px=err,
                        torso_px=torso, frame_ok=ok, person=person, offset=d,
                        aligned=align["verdict"] == "PASS", video_file=VIDEO_OF[v])
    real = vf >= 0
    return dict(rows=N, real_frames=int(real.sum()), detected=float((person >= 0)[real].mean()),
                frame_ok=float(ok[real].mean()), multi_person_frames=float((P["npose"][vf[real]] > 1).mean()),
                person_not_first=float((person[real] > 0).mean()))


def placeholder(v):
    """Video that failed alignment: zero arrays of the right shape, so rows stay 1:1 with
    punch_data. NO pose data. meta 'aligned' = False; punch_data3d can drop these rows."""
    N = len(clip_frames(v, annotations(v))[0])
    OUT.mkdir(exist_ok=True)
    np.save(OUT / f"V{v}.npy", np.zeros((N, T, 17, 3), np.float32))
    np.save(OUT / f"V{v}_2d.npy", np.zeros((N, T, 17, 2), np.float32))
    np.save(OUT / f"V{v}_conf.npy", np.zeros((N, T, 17), np.float32))
    np.savez_compressed(OUT / f"V{v}_meta.npz", presence=np.zeros((N, T, 17), np.float32),
                        video_frame=np.full((N, T), -1, np.int32), match_err_px=np.full((N, T), np.nan, np.float32),
                        torso_px=np.full(N, np.nan, np.float32), frame_ok=np.zeros((N, T), bool),
                        person=np.full((N, T), -1, np.int8), offset=0, aligned=False, video_file="")


def main():
    al = json.loads((HERE / "alignment.json").read_text())
    summary = {}
    for v in range(1, 11):
        a = al[f"V{v}"]
        if a["verdict"] != "PASS":
            placeholder(v)
            print(f"V{v}: alignment {a['verdict']} -> placeholder only (all zeros, aligned=False)", flush=True)
            summary[f"V{v}"] = dict(placeholder=True)
            continue
        summary[f"V{v}"] = s = build(v, a)
        print(f"V{v}: {s}", flush=True)
    (HERE / "build_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
