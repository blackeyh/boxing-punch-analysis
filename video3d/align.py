"""Find, per video, how the dataset's frame numbers map onto the downloaded video's frames,
and measure how well MediaPipe's 2D agrees with the dataset's 2D once aligned.

    python video3d/align.py            # all videos -> video3d/alignment.json (+ printed table)

Method (per video):
  1. Dataset frame map: {original frame number: dataset (17,2)} over every annotated real frame
     (overlapping clips are bit-identical on shared frames, so the map is unambiguous).
  2. Coarse search: offset d in [-COARSE, +COARSE] on a random subsample of annotated frames.
     cost(f, d) = min over detected people of the mean distance over shoulders/elbows/wrists/hips
     between dataset 2D at f and MediaPipe 2D at video frame f + d. Score(d) = median cost.
  3. Fine search: all annotated frames, d in best +- FINE.
  4. Drift / time-scaling check: best offset in each of N_SEG time segments, linear fit of
     offset vs frame number (a fps mismatch would show up as a slope).
  5. Per-clip check: best local offset of every clip in [d-3, d+3].
  6. Coordinate convention: least-squares fit dataset_x = a*mp_x + b (and y) at the best offset;
     a ~ 1, b ~ 0 means both are x/width, y/height in [0, 1].
  7. Mirror check: score of the horizontally flipped estimate (x -> 1-x, left/right swapped).
"""
import json
import sys

import numpy as np

from common import ALIGN_J, HERE, SWAP_LR, VIDEO_OF, dataset_frame_map, joint_err, load_pose, stream_frame_map

COARSE, FINE, N_SEG = 3000, 30, 8
rng = np.random.default_rng(0)


def costs(fr, ds, est, W, H, d, joints=ALIGN_J):
    """Per-frame best-person cost at offset d. fr (n,), ds (n,17,2). NaN where nothing detected."""
    F = est.shape[0]
    g = fr + d
    ok = (g >= 0) & (g < F)
    c = np.full(len(fr), np.nan)
    cpx = np.full(len(fr), np.nan)
    if ok.any():
        n, px = joint_err(ds[ok], est[g[ok]], W, H, joints)
        with np.errstate(all="ignore"):
            k = np.nanargmin(np.where(np.isnan(n), np.inf, n), axis=1)
        idx = np.arange(len(k))
        c[ok], cpx[ok] = n[idx, k], px[idx, k]
    return c, cpx


def score(c):
    return np.nanmedian(c) if np.isfinite(c).sum() >= 10 else np.inf


def search(fr, ds, est, W, H, lo, hi):
    ds_range = np.arange(lo, hi + 1)
    sc = np.array([score(costs(fr, ds, est, W, H, d)[0]) for d in ds_range])
    return ds_range, sc


def align_video(v, use_stream=False, video_file=None):
    video_file = VIDEO_OF[v] if video_file is None else video_file
    P = load_pose(video_file)
    W, H = P["W"], P["H"]
    est = P["xy"]                                     # (F, K, 17, 2)
    m = stream_frame_map() if use_stream else dataset_frame_map(v)
    fr = np.array(sorted(m))
    ds = np.stack([m[f] for f in fr])
    out = dict(video=v, video_file=video_file, n_frames_video=int(est.shape[0]), n_annot_frames=int(len(fr)),
               W=W, H=H, fps=P["fps"], first_annot_frame=int(fr.min()), last_annot_frame=int(fr.max()))

    # 2. coarse
    sub = rng.choice(len(fr), size=min(1500, len(fr)), replace=False)
    dr, sc = search(fr[sub], ds[sub], est, W, H, -COARSE, COARSE)
    d0 = int(dr[np.argmin(sc)])
    far = np.abs(dr - d0) > 10                      # best score clearly away from the winner
    out["coarse_best"] = d0
    out["coarse_best_score"] = float(sc.min())
    out["coarse_second_best_far"] = float(sc[far].min())
    out["coarse_second_best_far_offset"] = int(dr[far][np.argmin(sc[far])])

    # 3. fine
    dr, sc = search(fr, ds, est, W, H, d0 - FINE, d0 + FINE)
    d = int(dr[np.argmin(sc)])
    out["offset"] = d
    out["profile"] = {int(a): round(float(b), 5) for a, b in zip(dr, sc) if abs(a - d) <= 10}
    c, cpx = costs(fr, ds, est, W, H, d)
    det = np.isfinite(c)
    out["detect_rate"] = float(det.mean())
    out["median_err_norm"] = float(np.nanmedian(c))
    out["mean_err_norm"] = float(np.nanmean(c))
    out["median_err_px"] = float(np.nanmedian(cpx))
    out["mean_err_px"] = float(np.nanmean(cpx))
    out["p90_err_px"] = float(np.nanpercentile(cpx, 90))
    # body scale for context: dataset torso length (shoulder-mid to hip-mid) in pixels
    sh = (ds[:, 5] + ds[:, 6]) / 2; hp = (ds[:, 11] + ds[:, 12]) / 2
    torso = np.sqrt(((sh - hp)[:, 0] * W) ** 2 + ((sh - hp)[:, 1] * H) ** 2)
    out["median_torso_px"] = float(np.median(torso))
    out["median_err_over_torso"] = float(np.nanmedian(cpx / torso))
    out["frac_err_below_0.25_torso"] = float(np.mean(cpx / torso < 0.25))   # NaN (no detection) counts as fail
    for dd in (-2, -1, 1, 2):
        out[f"median_err_px_at_{dd:+d}"] = float(np.nanmedian(costs(fr, ds, est, W, H, d + dd)[1]))
    out["sharpness_pm1"] = min(out["median_err_px_at_-1"], out["median_err_px_at_+1"]) / out["median_err_px"]

    # all-17-joint error, for information
    c17, c17px = costs(fr, ds, est, W, H, d, joints=list(range(17)))
    out["median_err_px_all17"] = float(np.nanmedian(c17px))

    # 4. drift
    segs = np.array_split(np.argsort(fr), N_SEG)
    seg = []
    for s in segs:
        r, scs = search(fr[s], ds[s], est, W, H, d - FINE, d + FINE)
        seg.append(dict(frame_mid=int(np.median(fr[s])), offset=int(r[np.argmin(scs)]),
                        err_norm=round(float(scs.min()), 5)))
    out["segments"] = seg
    xs = np.array([q["frame_mid"] for q in seg]); ys = np.array([q["offset"] for q in seg])
    slope = np.polyfit(xs, ys, 1)[0] if len(set(ys)) > 1 else 0.0
    out["drift_slope_frames_per_1000"] = float(slope * 1000)

    # 5. per-clip local offsets (dataset map built clip by clip, using contiguous runs)
    if not use_stream:
        from common import annotations, clip_frames
        ann = annotations(v)
        fl, clips = clip_frames(v, ann)
        local = []
        for i, frames in enumerate(fl):
            if len(frames) < 3:
                continue
            f_ = np.array(frames); x_ = clips[i, :len(frames)]
            r = np.arange(d - 3, d + 4)
            cs = [costs(f_, x_, est, W, H, dd)[0] for dd in r]
            scs = np.array([np.nanmedian(q) if np.isfinite(q).sum() >= 2 else np.inf for q in cs])
            if np.isfinite(scs).any():
                local.append(int(r[np.argmin(scs)] - d))
        local = np.array(local)
        out["clip_local_offset_hist"] = {int(k): int((local == k).sum()) for k in range(-3, 4)}
        out["clip_frac_local_offset_0"] = float(np.mean(local == 0))
        out["clip_frac_local_offset_within_1"] = float(np.mean(np.abs(local) <= 1))

    # 6. coordinate convention (best person per frame, all joints with finite values)
    g = fr + d
    okf = (g >= 0) & (g < est.shape[0])
    n, _ = joint_err(ds[okf], est[g[okf]], W, H)
    k = np.nanargmin(np.where(np.isnan(n), np.inf, n), axis=1)
    e = est[g[okf]][np.arange(len(k)), k]                 # (n, 17, 2)
    good = np.isfinite(n[np.arange(len(k)), k])
    A, B = e[good][:, ALIGN_J].reshape(-1, 2), ds[okf][good][:, ALIGN_J].reshape(-1, 2)
    conv = {}
    for ax, name in ((0, "x"), (1, "y")):
        a, b = np.polyfit(A[:, ax], B[:, ax], 1)
        conv[name] = dict(scale=round(float(a), 4), shift=round(float(b), 4),
                          corr=round(float(np.corrcoef(A[:, ax], B[:, ax])[0, 1]), 4))
    out["coord_fit_dataset_from_mediapipe"] = conv

    # 7. mirror check
    flip = est[..., SWAP_LR, :].copy(); flip[..., 0] = 1 - flip[..., 0]
    out["median_err_px_mirrored"] = float(np.nanmedian(costs(fr, ds, flip, W, H, d)[1]))
    swap = est[..., SWAP_LR, :]
    out["median_err_px_lr_swapped"] = float(np.nanmedian(costs(fr, ds, swap, W, H, d)[1]))
    return out


def verdict(r):
    """Pass rule (see REPORT.md): the offset must be (a) unambiguous: a 3-frame neighbourhood
    and the best offset >10 frames away are clearly worse, (b) stable over the video (all
    segments agree, no drift), (c) tight: median error <= 0.15 torso lengths, and
    (d) most frames agree: >= 80% of annotated frames within 0.25 torso length."""
    why = []
    if r["sharpness_pm1"] < 1.1:
        why.append("offset not sharp at +-1")
    if r["coarse_second_best_far"] < 1.5 * r["coarse_best_score"]:
        why.append("ambiguous: another offset >10 frames away scores almost as well")
    if len({s["offset"] for s in r["segments"]}) > 1:
        why.append(f"segment offsets disagree {[s['offset'] for s in r['segments']]}")
    if r["median_err_over_torso"] > 0.15:
        why.append(f"median error {r['median_err_over_torso']:.2f} torso > 0.15")
    if r["frac_err_below_0.25_torso"] < 0.80:
        why.append(f"only {100 * r['frac_err_below_0.25_torso']:.0f}% of frames within 0.25 torso")
    return ("PASS" if not why else "FAIL"), why


def main():
    vids = [int(a) for a in sys.argv[1:]] or list(range(1, 11))
    path = HERE / "alignment.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for v in vids:
        r = align_video(v)
        r["verdict"], r["why"] = verdict(r)
        if v == 6:   # ground truth: V6 stores original frame numbers for the whole stream
            r["v6_full_stream_check"] = {k: align_video(6, use_stream=True)[k] for k in
                                         ("offset", "median_err_px", "detect_rate", "n_annot_frames",
                                          "drift_slope_frames_per_1000", "segments")}
        res[f"V{v}"] = r
        print(f"V{v}: offset {r['offset']:+d} | err {r['median_err_norm']:.4f} norm, "
              f"{r['median_err_px']:.1f} px ({r['median_err_over_torso']:.2f} torso) | "
              f"+-1: {r['median_err_px_at_-1']:.1f}/{r['median_err_px_at_+1']:.1f} px | "
              f"det {100 * r['detect_rate']:.1f}% | segs {[s['offset'] for s in r['segments']]} | "
              f"mirror {r['median_err_px_mirrored']:.0f}px | {r['verdict']} {r['why']}", flush=True)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
