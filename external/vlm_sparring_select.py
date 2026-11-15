"""Sparring windows for the VLM labeller: A2's first 2 minutes (review/sparring_label/A2_label.mp4, which
starts at video frame 150), one row per (fighter, window).

Windows: 16 frames (0.53 s), every 8 frames. Fighter skeletons come from the identity tracker
(external/tracks/A2_sparring_ring.npz; F1 = RED headgear, F2 = GREEN headgear, checked on the video).
Crop = square around that fighter's joints over the window (x1.5). Windows where the fighter is
tracked in < 50% of frames are skipped (noted as not visible). The prompt names the fighter by
headgear colour, because the crop often contains the opponent too.
Output external/vlm_sparring.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N, STEP, OFFSET, SECONDS = 16, 8, 150, 120
WHO = {1: "wearing RED headgear", 2: "wearing GREEN headgear"}


def main():
    t = np.load(ROOT / "external" / "tracks" / "A2_sparring_ring.npz")
    W, H, fps = float(t["W"]), float(t["H"]), float(t["fps"])
    frames = t["frame"]
    n_clip = int(round(SECONDS * fps))
    sel, skipped = [], 0
    for k in (1, 2):
        xy, pres = t[f"f{k}_xy"], t[f"f{k}_present"]
        for s in range(0, n_clip - N + 1, STEP):
            i0 = int(np.searchsorted(frames, OFFSET + s))               # tracker index of clip frame s
            seg_p = pres[i0:i0 + N]
            if seg_p.mean() < 0.5:
                skipped += 1; continue
            p = xy[i0:i0 + N][seg_p].reshape(-1, 2)
            p = p[(p > 0).all(1)] * [W, H]
            x0, y0 = p.min(0); x1, y1 = p.max(0)
            side = 1.5 * max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            sel.append(dict(file="A2_label", start=s, n=N, crop=[float(cx - side / 2), float(cy - side / 2), float(side)],
                            fighter=k, who=WHO[k], video_frame=OFFSET + s))
    (ROOT / "external" / "vlm_sparring.json").write_text(json.dumps(sel, indent=0))
    print(f"{len(sel)} windows ({sum(w['fighter'] == 1 for w in sel)} red, {sum(w['fighter'] == 2 for w in sel)} green), "
          f"{skipped} skipped (fighter tracked < 50%)")


if __name__ == "__main__":
    main()
