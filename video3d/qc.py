"""Quality checks for Skeleton3D_data (run after build_dataset3d.py).

    /opt/anaconda3/envs/mediapipe/bin/python video3d/qc.py

Writes video3d/qc/:
  qc_summary.json           per-video numbers (bone stability, missing / low-confidence joints)
  alignment_profiles.png    2D error vs frame offset around the chosen offset, every video
  V{v}_clips.png            2 clips per video: video frames with the dataset 2D (green) and
                            MediaPipe 2D (red) overlaid, and the MediaPipe 3D (world) skeleton below
  V{v}_clip.gif             one clip animated: overlay | 3D
"""
import json
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common import HERE, ROOT, RAW, VIDEO_OF  # noqa: E402

sys.path.insert(0, str(ROOT))
from punch_data3d import load_dataset_3d  # noqa: E402

QC = HERE / "qc"
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
BONES = {"upper_arm": [(5, 7), (6, 8)], "forearm": [(7, 9), (8, 10)], "shoulder_width": [(5, 6)],
         "hip_width": [(11, 12)], "torso_side": [(5, 11), (6, 12)], "thigh": [(11, 13), (12, 14)],
         "shin": [(13, 15), (14, 16)]}
GROUPS = {"wrists": [9, 10], "elbows": [7, 8], "shoulders": [5, 6], "hips": [11, 12],
          "knees": [13, 14], "ankles": [15, 16], "face": [0, 1, 2, 3, 4]}


def bone_stats(X3, ok):
    """Within-clip coefficient of variation of each bone (frames with frame_ok), median over
    clips; and between-clip CV of each clip's median bone length."""
    out = {}
    for name, pairs in BONES.items():
        within, medians = [], []
        for a, b in pairs:
            L = np.linalg.norm(X3[:, :, a] - X3[:, :, b], axis=-1)    # (N, T)
            for i in range(len(L)):
                l = L[i][ok[i]]
                if len(l) >= 3:
                    within.append(l.std() / l.mean()); medians.append(np.median(l))
        medians = np.array(medians)
        out[name] = dict(within_clip_cv_median=round(float(np.median(within)), 4),
                         within_clip_cv_p90=round(float(np.percentile(within, 90)), 4),
                         median_length_m=round(float(np.median(medians)), 4),
                         between_clip_cv=round(float(medians.std() / medians.mean()), 4))
    return out


def read_frames(v, wanted):
    """Decode the video sequentially (same indexing as extraction) and keep the wanted frames."""
    cap = cv2.VideoCapture(str(RAW / "videos" / f"{v if isinstance(v, str) else f'V{v}'}.mp4"))
    wanted, got, k = set(wanted), {}, 0
    last = max(wanted)
    while k <= last:
        if k in wanted:
            ok, f = cap.read()
            if not ok:
                break
            got[k] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        elif not cap.grab():
            break
        k += 1
    cap.release()
    return got


def draw2d(ax, img, ds, mp, ok, W, H, box):
    x0, y0, x1, y1 = box
    ax.imshow(img[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
    for pts, col in ((ds, "lime"), (mp if ok else None, "red")):
        if pts is None or not np.abs(pts).sum():
            continue
        p = pts * [W, H]
        for a, b in EDGES:
            ax.plot(p[[a, b], 0], p[[a, b], 1], color=col, lw=1.2)
        ax.scatter(p[:, 0], p[:, 1], s=4, color=col, zorder=3)
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")


def draw3d(ax, p, ok):
    ax.cla()
    if ok and np.abs(p).sum():
        X, Y, Z = p[:, 0], p[:, 2], -p[:, 1]           # plot: x right, depth, up
        for a, b in EDGES:
            c = "tab:red" if a in (5, 7, 9, 11, 13, 15) and b in (5, 7, 9, 11, 13, 15) else \
                "tab:blue" if a in (6, 8, 10, 12, 14, 16) and b in (6, 8, 10, 12, 14, 16) else "gray"
            ax.plot(X[[a, b]], Y[[a, b]], Z[[a, b]], color=c, lw=1.5)
    ax.set_xlim(-0.8, 0.8); ax.set_ylim(-0.8, 0.8); ax.set_zlim(-1.0, 0.8)
    ax.view_init(elev=12, azim=-65)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1.1))


def crop_box(clips2d, W, H):
    p = clips2d.reshape(-1, 2)
    p = p[np.abs(p).sum(1) > 0] * [W, H]
    x0, y0 = p.min(0); x1, y1 = p.max(0)
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 0.6 + 20
    return (int(max(0, cx - s)), int(max(0, cy - s)), int(min(W, cx + s)), int(min(H, cy + s)))


def main():
    QC.mkdir(exist_ok=True)
    X3, X2, conf, y, vid, lengths, meta = load_dataset_3d(return_meta=True)
    al = json.loads((HERE / "alignment.json").read_text())
    real = meta["video_frame"] >= 0
    ok = meta["frame_ok"]
    summary = {}
    for v in range(1, 11):
        s = vid == v
        r = real[s]
        detected = (np.abs(X3[s]).sum((2, 3)) > 0)
        cv = conf[s][r]                                       # (frames, 17)
        det_cv = cv[detected[r]]
        summary[f"V{v}"] = dict(
            clips=int(s.sum()), real_frames=int(r.sum()),
            no_detection=round(float(1 - detected[r].mean()), 4),
            not_ok=round(float(1 - ok[s][r].mean()), 4),
            median_match_err_px=round(float(np.nanmedian(meta["match_err_px"][s][r])), 2),
            clips_all_frames_ok=round(float(np.mean([ok[s][i][r[i]].all() for i in range(s.sum())])), 4),
            low_vis_lt_0p5={g: round(float((det_cv[:, j] < 0.5).mean()), 4) for g, j in GROUPS.items()},
            median_vis={g: round(float(np.median(det_cv[:, j])), 3) for g, j in GROUPS.items()},
            bones=bone_stats(X3[s], ok[s]))
        b = summary[f"V{v}"]
        print(f"V{v}: no-det {b['no_detection']:.1%}  not-ok {b['not_ok']:.1%}  wrist vis<0.5 "
              f"{b['low_vis_lt_0p5']['wrists']:.1%}  forearm CV {b['bones']['forearm']['within_clip_cv_median']:.3f}",
              flush=True)
    (QC / "qc_summary.json").write_text(json.dumps(summary, indent=1))

    # alignment profiles
    fig, ax = plt.subplots(figsize=(7, 4))
    for v in range(1, 11):
        pr = al[f"V{v}"]["profile"]; d0 = al[f"V{v}"]["offset"]
        xs = np.array(sorted(int(k) for k in pr)); ys = np.array([pr[str(k)] for k in xs])
        ax.plot(xs - d0, ys / ys.min(), marker=".", label=f"V{v} (offset {d0:+d})")
    ax.set_xlabel("frames from chosen offset"); ax.set_ylabel("median 2D error / minimum")
    ax.set_title("Alignment: 2D error vs frame offset"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(QC / "alignment_profiles.png", dpi=110); plt.close(fig)

    # renders
    rng = np.random.default_rng(1)
    W, H = 1280, 720
    for v in range(1, 11):
        idx = np.where(vid == v)[0]
        straight = [i for i in idx if y[i] in ("Jab", "Cross") and lengths[i] >= 6]
        curved = [i for i in idx if y[i] not in ("Jab", "Cross") and lengths[i] >= 6]
        picks = [int(rng.choice(p)) for p in (straight, curved) if p]
        frames = {i: [int(f) for f in meta["video_frame"][i] if f >= 0] for i in picks}
        imgs = read_frames(VIDEO_OF[v], [f for fs in frames.values() for f in fs])
        fig = plt.figure(figsize=(10, 2.6 * 2 * len(picks)))
        for row, i in enumerate(picks):
            n = lengths[i]
            ks = np.linspace(0, n - 1, 4).round().astype(int)
            box = crop_box(np.concatenate([meta["X2d_orig"][i], X2[i]]), W, H)
            for c, k in enumerate(ks):
                ax = fig.add_subplot(2 * len(picks), 4, row * 8 + c + 1)
                draw2d(ax, imgs[frames[i][k]], meta["X2d_orig"][i, k], X2[i, k], ok[i, k], W, H, box)
                ax.set_title(f"{y[i]} f{frames[i][k]} k={k}", fontsize=7)
                ax3 = fig.add_subplot(2 * len(picks), 4, row * 8 + 4 + c + 1, projection="3d")
                draw3d(ax3, X3[i, k], ok[i, k])
        fig.suptitle(f"V{v}: green = dataset 2D, red = MediaPipe 2D; 3D = MediaPipe world "
                     f"(red = left side, blue = right)", fontsize=9)
        fig.tight_layout(); fig.savefig(QC / f"V{v}_clips.png", dpi=80); plt.close(fig)

        # GIF of the first pick
        i = picks[0]; n = lengths[i]
        box = crop_box(np.concatenate([meta["X2d_orig"][i], X2[i]]), W, H)
        gif = []
        fig = plt.figure(figsize=(6, 3))
        for k in range(n):
            fig.clf()
            a = fig.add_subplot(1, 2, 1)
            draw2d(a, imgs[frames[i][k]], meta["X2d_orig"][i, k], X2[i, k], ok[i, k], W, H, box)
            a.set_title(f"V{v} {y[i]} frame {frames[i][k]}", fontsize=7)
            draw3d(fig.add_subplot(1, 2, 2, projection="3d"), X3[i, k], ok[i, k])
            fig.canvas.draw()
            gif.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert(
                "P", palette=Image.ADAPTIVE, colors=128))
        plt.close(fig)
        gif[0].save(QC / f"V{v}_clip.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)
        print(f"V{v}: rendered clips {picks}", flush=True)


if __name__ == "__main__":
    main()
