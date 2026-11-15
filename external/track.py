"""Training-free identity tracker: which MediaPipe skeleton is which fighter, frame by frame.

MediaPipe returns an UNORDERED list of people per frame (measured order swaps 3-18%). Each frame:
  1. PREDICT each fighter's torso centre from its last position + smoothed velocity.
  2. SCORE every fighter x detection pair (lower = more likely the same person):
       position : |predicted - detected centre| / torso length          (down-weighted in clinches)
       appearance: Bhattacharyya distance between colour histograms (torso + thighs, HSV)
       shape    : difference in limb-length proportions (bone lengths / torso, 3D)
       pose     : mean keypoint distance to the fighter's previous pose / torso length
     Pairs too far apart (> GATE torso lengths) are impossible.
  3. ASSIGN jointly: the combination with the lowest total cost (brute force; <= 3 detections).
  4. UPDATE matched fighters; appearance learns slowly and only when the fighters are apart.
     Unmatched fighters coast on their prediction (present = False).
  5. CLINCHES: while the fighters overlap, rely on appearance + shape; after they separate, compare
     both labelings on appearance over the next frames and swap back retroactively if needed.
Init: the first frame with 2 well-separated people; Fighter 1 = left, Fighter 2 = right.

    /opt/anaconda3/bin/python external/track.py {id}            # real external clip
    /opt/anaconda3/bin/python external/track.py --synthetic      # self-test with known identities
"""
import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12)]
W_POS, W_APP, W_SHAPE, W_POSE = 1.0, 2.5, 1.5, 0.5
GATE, MISS_COST = 3.0, 4.0
LOST, LOST_APP = 30, 0.45           # unseen > LOST frames: re-acquire by appearance (Bhattacharyya < LOST_APP)
APP_LR, VEL_LR = 0.05, 0.5
SEP_CHECK = 12                        # frames after a clinch over which the labeling is re-checked


# ------------------------------------------------------------------ per-detection features
APP_SCALE = 0.5                       # colour histograms on a half-size frame (speed)


def hist_hsv(hsv, poly):
    """hsv: the (downscaled) HSV frame, converted ONCE per frame; poly in full-res pixels."""
    poly = poly * APP_SCALE
    x0, y0 = np.floor(poly.min(0)).astype(int); x1, y1 = np.ceil(poly.max(0)).astype(int) + 1
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, hsv.shape[1]), min(y1, hsv.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)                   # mask only over the polygon's box
    cv2.fillPoly(m, [(poly - [x0, y0]).astype(np.int32)], 1)
    if m.sum() < 8:
        return None
    h = cv2.calcHist([hsv[y0:y1, x0:x1]], [0, 1], m, [16, 4], [0, 180, 0, 256]).ravel()
    return h / max(h.sum(), 1e-6)


def appearance(hsv, xy_px):
    """Colour histograms of the torso (shoulders-hips quad) and the thighs."""
    torso = xy_px[[5, 6, 12, 11]]
    lt = np.array([xy_px[11], xy_px[13], xy_px[13] + [8, 0], xy_px[11] + [8, 0]])
    rt = np.array([xy_px[12], xy_px[14], xy_px[14] - [8, 0], xy_px[12] - [8, 0]])
    parts = [hist_hsv(hsv, torso), hist_hsv(hsv, lt), hist_hsv(hsv, rt)]
    parts = [p for p in parts if p is not None]
    return np.mean(parts, 0) if parts else None


def shape_vec(world):
    hip = (world[11] + world[12]) / 2; sh = (world[5] + world[6]) / 2
    torso = max(np.linalg.norm(sh - hip), 1e-3)
    return np.array([np.linalg.norm(world[a] - world[b]) for a, b in BONES]) / torso


def detections_from(pose, f, W, H, img=None):
    """Detections in frame f of a MediaPipe pose dump (any person order)."""
    dets = []
    hsv = None
    if img is not None and int(pose["npose"][f]) > 0:
        hsv = cv2.cvtColor(cv2.resize(img, None, fx=APP_SCALE, fy=APP_SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    for p in range(int(pose["npose"][f])):
        xy = pose["img"][f, p][MP2COCO, :2]; vis = pose["img"][f, p][MP2COCO, 3]; world = pose["world"][f, p][MP2COCO]
        if np.isnan(xy).any():
            continue
        px = xy * [W, H]
        hip = (px[11] + px[12]) / 2; sh = (px[5] + px[6]) / 2
        dets.append(dict(center=(hip + sh) / 2, scale=max(np.linalg.norm(sh - hip), 5.0), kp=px,
                         box=np.r_[px.min(0), px.max(0)], shape=shape_vec(world),
                         app=appearance(hsv, px) if hsv is not None else None,
                         xy=xy, vis=vis, world=world))
    return dets


# ------------------------------------------------------------------ tracker core
def bhatt(a, b):
    if a is None or b is None:
        return 0.5
    return float(np.sqrt(max(0.0, 1 - np.sum(np.sqrt(a * b)))))


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


class Tracker:
    def __init__(self, n=2):
        self.n = n; self.tr = None; self.assign = []; self.clinch_since = None; self.log = []

    def _init(self, dets):
        if len(dets) < self.n:
            return False
        ds = sorted(dets, key=lambda d: -d["scale"])[: self.n]           # the largest people
        ds = sorted(ds, key=lambda d: d["center"][0])                      # Fighter 1 = left
        if self.n == 2 and abs(ds[0]["center"][0] - ds[1]["center"][0]) < 1.5 * max(ds[0]["scale"], ds[1]["scale"]):
            return False
        self.tr = [dict(c=d["center"].copy(), v=np.zeros(2), s=d["scale"], app=d["app"], shape=[d["shape"]],
                        kp=d["kp"], box=d["box"], miss=0) for d in ds]
        return True

    def cost(self, t, d, clinch):
        # (fix 2026-11-13) position uncertainty grows while a fighter is unseen; after LOST frames
        # the fighter is re-acquired mainly by appearance (position nearly ignored, no gate), but
        # only if the clothing matches well (so a bystander is not grabbed instead)
        pos = np.linalg.norm(t["c"] - d["center"]) / t["s"] / (1 + 0.5 * t["miss"])
        app = bhatt(t["app"], d["app"])
        lost = t["miss"] > LOST
        if lost and app > LOST_APP:
            return np.inf
        if pos > GATE and not clinch and not lost:
            return np.inf
        shape = float(np.linalg.norm(np.median(t["shape"][-60:], 0) - d["shape"]))
        pose = float(np.mean(np.linalg.norm(t["kp"] - d["kp"], axis=1)) / t["s"])
        w_pos = 0.2 if clinch else (0.05 if lost else W_POS)
        return w_pos * pos + W_APP * app + W_SHAPE * shape + W_POSE * min(pose, 3) * (0 if lost else 1)

    def step(self, dets):
        if self.tr is None:
            ok = self._init(dets)
            self.assign.append({i: dets[i] for i in range(len(dets))} if False else {})
            if ok:
                self.assign[-1] = {k: d for k, d in enumerate(sorted(sorted(dets, key=lambda d: -d["scale"])[: self.n],
                                                                     key=lambda d: d["center"][0]))}
            return
        boxes = [t["box"] for t in self.tr]
        clinch = self.n == 2 and iou(boxes[0], boxes[1]) > 0.15
        if clinch and self.clinch_since is None:
            self.clinch_since = len(self.assign)
        best, best_c = {}, np.inf
        idx = list(range(len(dets)))
        for perm in itertools.permutations(idx + [None] * self.n, self.n):
            if any(p is not None and perm.count(p) > 1 for p in perm):
                continue
            c = sum(MISS_COST if p is None else self.cost(self.tr[k], dets[p], clinch) for k, p in enumerate(perm))
            if c < best_c:
                best_c, best = c, {k: dets[p] for k, p in enumerate(perm) if p is not None}
        for k, t in enumerate(self.tr):
            d = best.get(k)
            if d is None:
                t["miss"] += 1; t["c"] = t["c"] + t["v"]; t["v"] = 0.7 * t["v"]; continue   # coast, slowing to a stop
            t["v"] = (1 - VEL_LR) * t["v"] + VEL_LR * (d["center"] - t["c"]) / (1 + t["miss"])
            t["c"], t["s"], t["kp"], t["box"], t["miss"] = d["center"].copy(), 0.9 * t["s"] + 0.1 * d["scale"], d["kp"], d["box"], 0
            t["shape"].append(d["shape"])
            if not clinch and d["app"] is not None:
                t["app"] = d["app"] if t["app"] is None else (1 - APP_LR) * t["app"] + APP_LR * d["app"]
        self.assign.append(best)
        # after a clinch: re-check the labeling on appearance; swap retroactively if clearly better
        if self.clinch_since is not None and not clinch:
            end = len(self.assign)
            if end - self.clinch_since >= SEP_CHECK:
                keep = swap = 0.0
                for a in self.assign[end - SEP_CHECK:]:
                    if 0 in a and 1 in a:
                        keep += bhatt(self.tr[0]["app"], a[0]["app"]) + bhatt(self.tr[1]["app"], a[1]["app"])
                        swap += bhatt(self.tr[0]["app"], a[1]["app"]) + bhatt(self.tr[1]["app"], a[0]["app"])
                if swap + 0.15 * SEP_CHECK < keep:
                    for i in range(self.clinch_since, end):
                        a = self.assign[i]; self.assign[i] = {1 - k: d for k, d in a.items()}
                    for key in ("c", "v", "kp", "box", "shape"):
                        self.tr[0][key], self.tr[1][key] = self.tr[1][key], self.tr[0][key]
                    self.log.append(("swapped_back", self.clinch_since, end))
                self.clinch_since = None


# ------------------------------------------------------------------ synthetic self-test
def synthetic(seed=0, seconds=120):
    """Two REAL fighters (V8 and V9: real skeletons, real clothing pixels), moved so they cross
    each other again and again; per-frame order shuffled like MediaPipe, random dropouts.
    True identities are known, so swaps are measured exactly."""
    from video3d.common import VIDEO_OF
    rng = np.random.default_rng(seed)
    VD = Path("~/Downloads/Dataset_video3d/video3d/raw/videos")
    srcs = []
    for v, f0 in ((8, 1300), (9, 3400)):
        pose = dict(np.load(ROOT / "video3d" / "raw" / "pose" / f"{VIDEO_OF[v]}.npz"))   # into memory ONCE (npz re-decompresses on every access)
        cap = cv2.VideoCapture(str(VD / f"{VIDEO_OF[v]}.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        srcs.append((pose, cap, f0))
    W, H, n = 1280, 720, int(seconds * 24)
    truth, got, crossings = [], [], 0
    tk = Tracker(2)
    prev_side = None
    for k in range(n):
        dets = []
        # horizontal paths: they swap sides every ~6 s, with close encounters (clinch-like)
        phase = 2 * np.pi * k / (24 * 12)
        offs = [-260 * np.cos(phase), 260 * np.cos(phase)]
        side = np.sign(offs[0] - offs[1])
        if prev_side is not None and side != prev_side:
            crossings += 1
        prev_side = side
        for who, (pose, cap, f0) in enumerate(srcs):
            ok, img = cap.read()
            ds = detections_from(pose, f0 + k, W, H, img)
            if not ds or rng.random() < 0.03:                              # 3% random dropouts
                continue
            d = max(ds, key=lambda d: d["scale"])
            shift = np.array([offs[who] + (W / 2 - d["center"][0]), 0.0])  # move the fighter on a crossing path
            for key in ("center", "kp"):
                d[key] = d[key] + shift
            d["box"] = np.r_[d["kp"].min(0), d["kp"].max(0)]
            d["who"] = who
            dets.append(d)
        rng.shuffle(dets)                                                  # unordered, like MediaPipe
        tk.step(dets)
        truth.append({d["who"] for d in dets})
        got.append({k2: d["who"] for k2, d in tk.assign[-1].items()})
    # identity mapping fixed at init; count frames where a fighter label points at the other person
    mapping = next((g for g in got if len(g) == 2), None)
    wrong = sum(1 for g in got if g and mapping and any(g.get(k) is not None and g[k] != mapping[k] for k in g))
    frames_both = sum(1 for g in got if len(g) == 2)
    print(f"synthetic: {n} frames, {crossings} crossings | frames with both fighters assigned {frames_both} | "
          f"frames with a swapped identity {wrong} ({wrong / max(frames_both, 1):.1%}) | retroactive fixes {len(tk.log)}")
    return wrong / max(frames_both, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("id", nargs="?")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if args.synthetic:
        synthetic()
        return
    import json
    man = json.loads((HERE / "manifest.json").read_text())[args.id]
    pose = dict(np.load(HERE / "pose" / f"{args.id}.npz")); W, H = int(pose["width"]), int(pose["height"])
    cap = cv2.VideoCapture(str(HERE / "raw" / f"{args.id}.mp4"))
    f0 = int(pose["frame"][0])
    for _ in range(f0):
        cap.grab()
    tk = Tracker(2)
    pos = f0                                              # next video frame the reader will return
    for i in range(len(pose["frame"])):
        while pos < int(pose["frame"][i]):                # subsampled clips (60 fps, step 2): skip frames
            cap.grab(); pos += 1
        ok, img = cap.read(); pos += 1
        tk.step(detections_from(pose, i, W, H, img if ok else None))
    F = len(pose["frame"])
    out = {}
    for k in range(2):
        xy = np.zeros((F, 17, 2), np.float32); world = np.zeros((F, 17, 3), np.float32); vis = np.zeros((F, 17), np.float32)
        present = np.zeros(F, bool)
        for i, a in enumerate(tk.assign):
            if k in a:
                xy[i], world[i], vis[i], present[i] = a[k]["xy"], a[k]["world"], a[k]["vis"], True
        out[f"f{k + 1}"] = dict(xy=xy, world=world, vis=vis, present=present)
    (HERE / "tracks").mkdir(exist_ok=True)
    np.savez_compressed(HERE / "tracks" / f"{args.id}.npz", frame=pose["frame"], fps=pose["fps"], W=W, H=H,
                        **{f"{f}_{key}": val for f, d in out.items() for key, val in d.items()})
    print(f"[{args.id}] tracked {F} frames | fighter 1 present {out['f1']['present'].mean():.0%}, "
          f"fighter 2 present {out['f2']['present'].mean():.0%} | clinch re-checks that swapped back: {len(tk.log)}")


if __name__ == "__main__":
    main()
