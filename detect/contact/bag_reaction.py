"""Bag reaction after each labelled punch in the bag videos (V1-V4) -> automatic contact labels.

Per video (pixels only for the bag, the fighter's skeleton only to mask the fighter out):
  1. Decode once at 1/8 scale (160x90 grey). Keep frames e-2 .. e+10 around the end e of every
     labelled punch, plus pairs at random moments (the baseline: timers, overlays, camera noise).
  2. Bag mask, per 2-minute window (the V1 fighter switches between hanging bags): pixels whose
     change after punches exceeds their change at random moments, outside the fighter. Largest
     blob near the fighter = the bag being worked.
  3. Reaction per punch: max over k = 1..8 of the mean |frame(e+k) - frame(e-2)| inside the bag mask,
     fighter masked out at both frames, as a z-score against random moments of the same window.
  4. Labels: contact if z >= Z_HI, no_contact if z <= Z_LO, else ambiguous (excluded).
Outputs detect/contact/bag_reaction.json (+ per-video QC images in detect/contact/qc/).
    /opt/anaconda3/bin/python detect/contact/bag_reaction.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from progress import progress  # noqa: E402
from punch_data import CLASSES  # noqa: E402
from video3d.common import VIDEO_OF  # noqa: E402

VIDEOS_DIR = Path("~/Downloads/Dataset_video3d/video3d/raw/videos")
OUT = ROOT / "detect" / "contact"
A_DIR = Path("~/Downloads/Dataset_videodet/detect/scores/videomae_probe")
SC = 8                                   # downscale factor
PRE, POST = 2, 10
WIN_SEC = 120
Z_HI, Z_LO = 3.0, 1.0
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def body_mask(xy, present, f, w, h):
    m = np.zeros((h, w), np.uint8)
    if 0 <= f < len(present) and present[f]:
        p = (xy[f] * [w, h]).astype(int)
        for a, b in BONES:
            cv2.line(m, tuple(p[a]), tuple(p[b]), 1, max(3, w // 22))
        for j in (9, 10, 0):                                          # gloves and head are wide
            cv2.circle(m, tuple(p[j]), max(3, w // 30), 1, -1)
    return m.astype(bool)


def punches(label):
    out, f = [], 0
    while f < len(label):
        if label[f] >= 0:
            g = f
            while g + 1 < len(label) and label[g + 1] == label[f]:
                g += 1
            out.append((f, g, CLASSES[label[f]])); f = g + 1
        else:
            f += 1
    return out


def main():
    rng = np.random.default_rng(0)
    (OUT / "qc").mkdir(parents=True, exist_ok=True)
    results = {}
    progress.start("bag reaction", total=4)
    for v in (1, 2, 3, 4):
        fr = np.load(ROOT / "detect" / "frames" / f"V{v}.npz")
        xy = fr["xy"].astype(np.float32); present = fr["present"]; label = fr["label"]; fps = float(fr["fps"])
        if v == 4:                                                    # frames file is mirrored to orthodox; undo for pixels
            from baselines import MIRROR_IDX
            xy = xy[:, MIRROR_IDX]; xy[..., 0] = np.where(present[:, None], 1 - xy[..., 0], 0)
        P = punches(label); F = len(label)
        # baseline moments: CLEAN no-punch only (detector A < 0.05 and B < 0.2, away from labels). Random
        # gap moments are not enough: e.g. V2 has ~270 hidden-punch events that also move the bag.
        near = np.convolve((label >= 0).astype(float), np.ones(41), "same") > 0
        A = np.load(A_DIR / f"V{v}.npz"); B = np.load(ROOT / "detect" / "scores" / "pu_skeleton" / f"V{v}.npz")
        clean = np.zeros(F, bool)
        for st, sa, sb in zip(A["start"], A["score"], B["score"][:len(A["start"])]):
            if sa < 0.05 and sb < 0.2:
                clean[st + 8] = True                                    # window centre
        flagged = np.zeros(F, bool)
        for st, sa, sb in zip(A["start"], A["score"], B["score"][:len(A["start"])]):
            if sa >= 0.5 or sb >= 0.5:
                flagged[max(0, st - 8): st + 24] = True
        ok_rand = np.where(clean & ~near & ~flagged & present)[0]
        ok_rand = ok_rand[(ok_rand >= PRE) & (ok_rand < F - POST)]
        rand_e = rng.choice(ok_rand, size=min(1500, len(ok_rand)), replace=False)
        need = set()
        for _, e, _ in P:
            need.update(range(e - PRE, e + POST + 1))
        for e in rand_e:
            need.update(range(e - PRE, e + POST + 1))
        cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{VIDEO_OF[v]}.mp4"))
        W, H = int(fr["W"]) // SC, int(fr["H"]) // SC
        frames = {}
        last = max(need)
        for f in range(last + 1):
            if f in need:
                ok, img = cap.read()
                if ok:
                    frames[f] = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                cap.grab()
        cap.release()

        def change(e):
            """(H,W) max change after e vs e-PRE, fighter masked out at both frames; None if missing."""
            if e - PRE not in frames:
                return None
            base = frames[e - PRE]; mb = body_mask(xy, present, e - PRE, W, H)
            best = None
            for k in range(1, POST + 1):
                if e + k not in frames:
                    continue
                d = np.abs(frames[e + k] - base); d[mb | body_mask(xy, present, e + k, W, H)] = np.nan
                best = d if best is None else np.fmax(best, d)
            return best

        win = int(WIN_SEC * fps)
        pch = [(s, e, t, change(e)) for s, e, t in P]
        rnd = [(e, change(e)) for e in rand_e]
        rows = []
        for w0 in range(0, F, win):
            pw = [x for x in pch if w0 <= x[1] < w0 + win and x[3] is not None]
            rw = [x for x in rnd if w0 - win // 2 <= x[0] < w0 + win + win // 2 and x[1] is not None]
            if len(pw) < 5 or len(rw) < 20:
                continue
            mp = np.nanmean(np.stack([x[3] for x in pw]), 0); mr = np.nanmean(np.stack([x[1] for x in rw]), 0)
            contrast = np.nan_to_num(mp - mr)
            thr = np.quantile(contrast, 0.97)
            cand = (contrast > max(thr, 1.0)).astype(np.uint8)
            n, comp, stats, cent = cv2.connectedComponentsWithStats(cand)
            if n <= 1:
                continue
            # the bag being worked: large blob, closest to the fighter's median position in this window
            fx = np.nanmedian([np.mean(xy[e, [5, 6, 11, 12]], 0) * [W, H] for _, e, _, _ in pw if present[e]], axis=0)
            reach = 0.45 * W
            near_blobs = [i for i in range(1, n) if np.linalg.norm(cent[i] - fx) <= reach] or list(range(1, n))
            best = max(near_blobs, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            bag = comp == best
            bag = cv2.dilate(bag.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            base_scores = np.array([np.nanmean(x[1][bag]) for x in rw])
            mu, sd = np.nanmean(base_scores), np.nanstd(base_scores) + 1e-6
            for s, e, t, ch in pw:
                sc = float(np.nanmean(ch[bag]))
                z = (sc - mu) / sd
                lab = "contact" if z >= Z_HI else ("no_contact" if z <= Z_LO else "ambiguous")
                rows.append(dict(video=v, start=int(s), end=int(e), type=t, reaction=round(sc, 3), z=round(float(z), 2), label=lab))
            if True:
                vis = cv2.applyColorMap(np.clip(contrast * 12, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                vis[bag] = (0.5 * vis[bag] + [0, 127, 0]).astype(np.uint8)
                cv2.imwrite(str(OUT / "qc" / f"V{v}_bagmask_{w0 // win:02d}.png"), cv2.resize(vis, (W * 3, H * 3), interpolation=cv2.INTER_NEAREST))
        labs = [r["label"] for r in rows]
        results[f"V{v}"] = rows
        print(f"V{v}: punches {len(P)} scored {len(rows)} | contact {labs.count('contact')} | no_contact {labs.count('no_contact')} | "
              f"ambiguous {labs.count('ambiguous')} | median z {np.median([r['z'] for r in rows]):.1f}", flush=True)
        progress.tick(1, note=f"V{v}")
    progress.finish()
    (OUT / "bag_reaction.json").write_text(json.dumps(dict(rule=dict(Z_HI=Z_HI, Z_LO=Z_LO, PRE=PRE, POST=POST, WIN_SEC=WIN_SEC, SC=SC), punches=results), indent=0))


if __name__ == "__main__":
    main()
