"""Real-time simulation on a HELD-OUT video, rendered to an annotated MP4.

1. Train the 7-class model (6 punches + No Punch, detect/dataset7) on the OTHER 9 videos only.
2. Stream through the chosen excerpt as a live camera would: every STEP frames, classify the
   LAST 8, 12 and 16 frames (no future frames) and average the three opinions.
3. Turn the stream of probabilities into counted punches (fixed rules, not tuned on this video):
     P(punch) = 1 - P(No Punch), smoothed over 3 steps;
     an event starts when P(punch) > ON and ends when it falls below OFF;
     events shorter than MIN_STEPS steps are dropped; each event counts once, with the punch type
     whose probability summed highest during the event.
4. Compare with the annotations (a labelled punch is "detected" if an event overlaps it) and
   render the original video with skeleton, live call, per-type counters, probability trace and
   the annotators' labels underneath.

Pose comes from the stored MediaPipe per-frame results (detect/frames), i.e. exactly what a live
MediaPipe would have produced frame by frame. Not for V4 (its frame data is mirrored).
    /opt/anaconda3/bin/python detect/realtime_demo.py --video 9 --start 3400 --seconds 180
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mem_guard  # noqa: F401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from deep import TorchSeqClassifier  # noqa: E402
from detect.train7 import features  # noqa: E402
from progress import progress  # noqa: E402
from punch_data import CLASSES, T  # noqa: E402
from video3d.common import VIDEO_OF  # noqa: E402

VIDEOS_DIR = Path("~/Downloads/Dataset_video3d/video3d/raw/videos")
SCALES, STEP = (8, 12, 16), 2
ON, OFF, MIN_STEPS = 0.6, 0.4, 2
NP_ = "No Punch"
SHORT = {"Jab": "JAB", "Cross": "CROSS", "Lead Hook": "LEAD HOOK", "Rear Hook": "REAR HOOK",
         "Lead Uppercut": "LEAD UPPERCUT", "Rear Uppercut": "REAR UPPERCUT"}
EDGES = [(5, 7, 0), (7, 9, 0), (11, 13, 0), (13, 15, 0), (6, 8, 1), (8, 10, 1), (12, 14, 1), (14, 16, 1),
         (5, 6, 2), (5, 11, 2), (6, 12, 2), (11, 12, 2)]


def train_model(video, seed):
    d = np.load(ROOT / "detect" / "dataset7" / "clips.npz")
    tr = d["video"] != video
    F = features(d)[tr]
    m = TorchSeqClassifier("cnn", epochs=60, seed=seed).fit(F, d["label"][tr])
    return m


def window_batch(fr, ends, W):
    n = len(ends)
    xy = np.zeros((n, T, 17, 2), np.float16); world = np.zeros((n, T, 17, 3), np.float16); vis = np.zeros((n, T, 17), np.float16)
    for i, t in enumerate(ends):
        s = t - W + 1
        xy[i, :W], world[i, :W], vis[i, :W] = fr["xy"][s:t + 1], fr["world"][s:t + 1], fr["vis"][s:t + 1]
        # frames with no detected person: copy the nearest present frame of the window
        empty = np.abs(xy[i, :W].astype(np.float32)).sum(axis=(1, 2)) == 0
        if empty.any() and not empty.all():
            good = np.where(~empty)[0]
            for k in np.where(empty)[0]:
                g = good[np.argmin(np.abs(good - k))]; xy[i, k], world[i, k], vis[i, k] = xy[i, g], world[i, g], vis[i, g]
    ok = np.abs(xy.astype(np.float32)).sum(axis=(1, 2, 3)) > 0
    return dict(xy=xy, world=world, vis=vis), ok


LEAD_IDX, REAR_IDX = [0, 2, 4], [1, 3, 5]      # Jab, Lead Hook, Lead Uppercut | Cross, Rear Hook, Rear Uppercut
DIP = 0.15                                    # split where P(punch) dips this far below both neighbouring peaks


def split_points(sm, p_types, s0, s1):
    """Combinations (e.g. jab-cross) keep P(punch) high, so one detection can hold several punches.
    Split a detection [s0, s1) (1) where the punching hand switches (lead <-> rear), sustained for
    2 steps, and (2) at clear dips in P(punch) between two peaks."""
    cuts = set()
    side = p_types[s0:s1, LEAD_IDX].sum(1) - p_types[s0:s1, REAR_IDX].sum(1)
    sgn = np.sign(np.convolve(side, np.ones(2) / 2, "same"))
    for k in range(1, len(sgn) - 1):
        if sgn[k] != sgn[k - 1] and sgn[k] == sgn[k + 1] and sgn[k] != 0:
            cuts.add(s0 + k)
    seg = sm[s0:s1]
    for k in range(1, len(seg) - 1):
        if seg[k] <= seg[k - 1] and seg[k] < seg[k + 1]:
            if seg[:k].max() - seg[k] >= DIP and seg[k + 1:].max() - seg[k] >= DIP:
                cuts.add(s0 + k)
    pts = sorted(c for c in cuts if s0 < c < s1)
    bounds = [s0] + pts + [s1]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= 1]


def events_from(times, p_punch, p_types):
    sm = np.convolve(p_punch, np.ones(3) / 3, "same")
    ev, on, s0 = [], False, 0
    for i, p in enumerate(sm):
        if not on and p > ON:
            on, s0 = True, i
        elif on and p < OFF:
            on = False
            if i - s0 >= MIN_STEPS:
                for a, b in split_points(sm, p_types, s0, i):
                    tp = p_types[a:b].sum(0)
                    ev.append(dict(start=int(times[a]), end=int(times[b - 1]), type=CLASSES[int(tp.argmax())],
                                   conf=float(sm[a:b].max()), type_share=float(tp.max() / tp.sum())))
    return ev, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, default=9)
    ap.add_argument("--start", type=int, default=3400)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()
    v = args.video
    assert v != 4, "V4 frame data is mirrored to orthodox; not supported in the demo"
    out = ROOT / "runs" / f"realtime_V{v}"; out.mkdir(parents=True, exist_ok=True)
    fr = dict(np.load(ROOT / "detect" / "frames" / f"V{v}.npz")); fps = float(fr["fps"])
    a, b = args.start, min(args.start + int(args.seconds * fps), len(fr["label"]))

    print(f"training 7-class model without V{v} ...", flush=True)
    t0 = time.time(); model = train_model(v, args.seed); print(f"  trained in {time.time() - t0:.0f}s")

    # ---- streaming inference (causal: each decision uses only frames <= t)
    times = np.arange(a + max(SCALES) - 1, b, STEP)
    probs = np.zeros((len(times), 7), np.float32); t_inf = 0.0
    for W in SCALES:
        batch, ok = window_batch(fr, times, W)
        t1 = time.time(); F = features(batch); P = model.predict_proba(F); t_inf += time.time() - t1
        order = [list(model.classes_).index(c) for c in CLASSES + [NP_]]
        P = P[:, order]; P[~ok] = np.eye(7)[6]               # nobody in view -> no punch
        probs += P / len(SCALES)
    ms_per_decision = 1000 * t_inf / len(times)
    p_punch = 1 - probs[:, 6]
    p_types = probs[:, :6] / np.maximum(probs[:, :6].sum(1, keepdims=True), 1e-6)
    events, sm = events_from(times, p_punch, probs[:, :6])

    # ---- compare with annotations inside the excerpt
    lab = fr["label"]; segs = []
    f = a
    while f < b:
        if lab[f] >= 0:
            g = f
            while g + 1 < b and lab[g + 1] == lab[f]:
                g += 1
            segs.append(dict(start=f, end=g, type=CLASSES[lab[f]])); f = g + 1
        else:
            f += 1
    # one-to-one matching in time order: each labelled punch takes the first unused counted punch
    # that overlaps it (+-4 frames)
    used = set()
    for s in segs:
        s["detected"], s["model_type"] = False, None
        for j, e in enumerate(events):
            if j not in used and e["start"] <= s["end"] + 4 and e["end"] >= s["start"] - 4:
                used.add(j); s["detected"], s["model_type"] = True, e["type"]; break
    for j, e in enumerate(events):
        e["matches_label"] = j in used
    det = [s for s in segs if s["detected"]]
    summary = dict(video=v, frames=[a, b], fps=fps, seconds=(b - a) / fps, labelled_punches=len(segs),
                   model_count=len(events), matched=len(det), missed=len(segs) - len(det),
                   type_correct=sum(s["model_type"] == s["type"] for s in det),
                   extra_counts_without_label=sum(not e["matches_label"] for e in events),
                   ms_per_decision=round(ms_per_decision, 2), decisions_per_second_needed=fps / STEP,
                   rules=dict(scales=SCALES, step=STEP, on=ON, off=OFF, min_steps=MIN_STEPS))
    (out / "summary.json").write_text(json.dumps(dict(summary=summary, events=events, labelled=segs), indent=1))
    print(json.dumps(summary, indent=1), flush=True)

    # ---- render
    cap = cv2.VideoCapture(str(VIDEOS_DIR / f"{VIDEO_OF[v]}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    Wd = args.width; Hd = int(round(Wd * float(fr["H"]) / float(fr["W"])))
    path = out / f"realtime_V{v}_{a}-{b}.mp4"
    vw = None
    for codec in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (Wd, Hd))
        if vw.isOpened():
            break
    counts = {c: 0 for c in CLASSES}; ann_counts = {c: 0 for c in CLASSES}
    ev_by_end = {e["end"]: e for e in events}; seg_by_start = {s["start"]: s for s in segs}
    last_ev, last_ev_frame = None, -999
    font = cv2.FONT_HERSHEY_SIMPLEX
    progress.start(f"render V{v}", total=b - a)
    hist_len = int(3 * fps / STEP)
    for f in range(a, b):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
        # skeleton
        if fr["present"][f]:
            p = fr["xy"][f].astype(np.float32) * [Wd, Hd]
            colors = [(60, 60, 230), (230, 140, 60), (230, 230, 230)]
            for i, j, c in EDGES:
                cv2.line(img, tuple(map(int, p[i])), tuple(map(int, p[j])), colors[c], 2, cv2.LINE_AA)
        k = np.searchsorted(times, f, side="right") - 1
        if f in ev_by_end:
            e = ev_by_end[f]; counts[e["type"]] += 1; last_ev, last_ev_frame = e, f
        if f in seg_by_start:
            ann_counts[seg_by_start[f]["type"]] += 1
        # live call panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (Wd, 64), (20, 20, 20), -1)
        cv2.rectangle(overlay, (Wd - 300, 64), (Wd, 64 + 30 * 8), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, Hd - 70), (Wd, Hd), (20, 20, 20), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        if k >= 0:
            pp = sm[k]
            if pp > ON:
                t_now = CLASSES[int(probs[k, :6].argmax())]
                cv2.putText(img, f"{SHORT[t_now]}  {p_types[k].max():.0%}", (16, 44), font, 1.15, (80, 220, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(img, "no punch", (16, 44), font, 1.0, (170, 170, 170), 2, cv2.LINE_AA)
            # probability trace (last 3 s)
            x0, w_tr = Wd // 2 - 40, Wd // 2 - 280
            seg = sm[max(0, k - hist_len): k + 1]
            pts = [(x0 + int(w_tr * i / max(1, hist_len)), 52 - int(40 * q)) for i, q in enumerate(seg)]
            cv2.line(img, (x0, 52 - int(40 * ON)), (x0 + w_tr, 52 - int(40 * ON)), (90, 90, 90), 1)
            for p1, p2 in zip(pts, pts[1:]):
                cv2.line(img, p1, p2, (80, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(img, "P(punch), last 3 s", (x0, 62), font, 0.4, (170, 170, 170), 1, cv2.LINE_AA)
        # counters
        cv2.putText(img, f"MODEL COUNT  {sum(counts.values())}", (Wd - 288, 92), font, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        for i, c in enumerate(CLASSES):
            hl = last_ev is not None and last_ev["type"] == c and f - last_ev_frame < fps * 0.6
            cv2.putText(img, f"{SHORT[c]:<14}{counts[c]:>4}", (Wd - 288, 122 + 26 * i), font, 0.58,
                        (80, 220, 255) if hl else (225, 225, 225), 2 if hl else 1, cv2.LINE_AA)
        # annotators' strip
        cur = CLASSES[lab[f]] if lab[f] >= 0 else None
        txt = f"ANNOTATORS: {SHORT[cur] if cur else '-'}   labelled so far {sum(ann_counts.values())}"
        cv2.putText(img, txt, (16, Hd - 40), font, 0.6, (150, 230, 150) if cur else (170, 170, 170), 1, cv2.LINE_AA)
        if last_ev is not None and f - last_ev_frame < fps * 0.8 and not last_ev["matches_label"]:
            cv2.putText(img, "last model punch has NO annotator label (hidden punch or false alarm?)", (16, Hd - 14),
                        font, 0.5, (90, 170, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, f"V{v} held out from training | {(f - a) / fps:5.1f}s | decision every {STEP} frames from past frames only",
                        (16, Hd - 14), font, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        vw.write(img)
        if (f - a) % 200 == 0:
            progress.tick(200, note=f"frame {f - a}/{b - a}")
    vw.release(); cap.release(); progress.finish()
    print(f"video: {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
