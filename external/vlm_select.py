"""Benchmark selection for the VLM labeller: windows from OUR OWN labelled videos, where the truth is known.

Per video (seeded): 10 labelled punches (window centred on the punch) and 10 clean no-punch clips
(the strict no-punch rule of detect/build_7class.py), 16 frames each at the video's own frame rate.
Crop = square around the boxer's skeleton over the window (union of joint boxes, x1.5, from
detect/frames; V4's frames were mirrored for training, so its crop is mirrored back to the real video).
Output external/vlm_bench.json: video, file (video file stem), start, n, crop [x, y, side] in pixels,
label (class or "No Punch"), stance (V4 southpaw, others orthodox).
    /opt/anaconda3/bin/python external/vlm_select.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from video3d.common import VIDEO_OF  # noqa: E402

N = 16


def crop_for(fr, s, n, mirrored):
    xy = fr["xy"][s:s + n].astype(np.float32); pres = fr["present"][s:s + n]
    p = xy[pres].reshape(-1, 2)
    p = p[(p > 0).all(1)]
    if mirrored:
        p[:, 0] = 1 - p[:, 0]
    W, H = float(fr["W"]), float(fr["H"])
    x0, y0 = p.min(0) * [W, H]; x1, y1 = p.max(0) * [W, H]
    side = 1.5 * max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return [float(cx - side / 2), float(cy - side / 2), float(side)]


def main():
    rng = np.random.default_rng(0)
    d = np.load(ROOT / "detect" / "dataset7" / "clips.npz")
    sel = []
    for v in range(1, 11):
        fr = {k: np.asarray(x) for k, x in np.load(ROOT / "detect" / "frames" / f"V{v}.npz").items()}
        F = len(fr["present"])
        m = d["video"] == v
        pun = np.where(m & (d["source"] == "annotation"))[0]
        nop = np.where(m & (d["source"] == "nopunch_rule"))[0]
        for i in rng.choice(pun, 10, replace=False):
            c = int(d["start"][i] + d["length"][i] // 2); s = int(np.clip(c - N // 2, 0, F - N))
            sel.append(dict(video=v, file=VIDEO_OF[v], start=s, n=N, crop=crop_for(fr, s, N, v == 4),
                            label=str(d["label"][i]), stance="southpaw" if v == 4 else "orthodox"))
        for i in rng.choice(nop, 10, replace=False):
            s = int(np.clip(d["start"][i] + d["length"][i] // 2 - N // 2, 0, F - N))
            sel.append(dict(video=v, file=VIDEO_OF[v], start=s, n=N, crop=crop_for(fr, s, N, v == 4),
                            label="No Punch", stance="southpaw" if v == 4 else "orthodox"))
    (ROOT / "external" / "vlm_bench.json").write_text(json.dumps(sel, indent=0))
    print(len(sel), "windows")


if __name__ == "__main__":
    main()
