"""Contact sheets for visual QC of a detector's extremes (runs locally on the Mac videos).

    python videodet/qc.py [detector]     # default videomae_probe

Per sheet: one row per window = full frame (middle frame, crop square drawn) + 8 crop frames
(every 2nd frame of the 16-frame window), captioned with video, start frame, time, score.
Writes videodet/qc/<detector>_<kind>.png for
  top_gap   highest-scoring GAP windows (candidate missed punches / false alarms)
  low_punch lowest-scoring LABELLED-punch windows (candidate misses / label errors)
Picks are spread across videos and de-duplicated (>= 32 frames apart).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import windows as Wd  # noqa: E402
from extract import cut  # noqa: E402

LOCAL_VIDEOS = Path("~/Downloads/Dataset_video3d/video3d/raw/videos")
FEATS = HERE / "cache" / "feats"
SCORES = HERE.parent / "detect" / "scores"
TH = 112


def pick(det, kind, per_video=2, total=20):
    out = []
    for v in range(1, 11):
        s = np.load(SCORES / det / f"V{v}.npz")
        lab = Wd.window_labels(v)
        m = lab["is_gap"] if kind == "top_gap" else lab["is_punch"]
        idx = np.where(m)[0]
        order = idx[np.argsort(-s["score"][idx] if kind == "top_gap" else s["score"][idx])]
        chosen = []
        for i in order:
            if all(abs(int(lab["start"][i]) - int(lab["start"][j])) >= 32 for j in chosen):
                chosen.append(i)
            if len(chosen) == per_video:
                break
        out += [(v, i, float(s["score"][i]), lab["label"][i], str(s["type"][i]) if "type" in s else "") for i in chosen]
    return out[:total]


def row(v, i, score, label, guess):
    st = int(Wd.starts(v)[i])
    crop = np.load(FEATS / f"V{v}.npz")["crop"][i]
    cap = cv2.VideoCapture(str(LOCAL_VIDEOS / f"{Wd.VIDEO_OF[v]}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, st)
    frames = [cap.read()[1] for _ in range(Wd.W)]
    cap.release()
    full = frames[8].copy()
    x, y, side = crop.astype(int)
    cv2.rectangle(full, (x, y), (x + side, y + side), (0, 255, 255), 4)
    full = cv2.resize(full, (TH * 16 // 9, TH))
    tiles = [cv2.resize(cv2.cvtColor(cut(f, crop), cv2.COLOR_RGB2BGR), (TH, TH)) for f in frames[::2]]
    r = np.hstack([full] + tiles)
    t = st / Wd.fps(v)
    txt = f"V{v} f{st} {int(t // 60)}:{t % 60:04.1f} s={score:.2f} {('lab=' + label) if label else ''} {('guess=' + guess) if guess else ''}"
    cv2.rectangle(r, (0, 0), (r.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(r, txt, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return r


def main():
    det = sys.argv[1] if len(sys.argv) > 1 else "videomae_probe"
    (HERE / "qc").mkdir(exist_ok=True)
    for kind in ["top_gap", "low_punch"]:
        picks = pick(det, kind)
        sheet = np.vstack([row(*p) for p in picks])
        f = HERE / "qc" / f"{det}_{kind}.png"
        from PIL import Image                       # 256-colour palette keeps the PNG ~0.6 MB
        Image.fromarray(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)).quantize(256, method=Image.Quantize.MEDIANCUT).save(f, optimize=True)
        print(f, sheet.shape)
        for p in picks:
            print("  ", kind, f"V{p[0]} start {int(Wd.starts(p[0])[p[1]])} score {p[2]:.3f} {p[3]} {p[4]}")


if __name__ == "__main__":
    main()
