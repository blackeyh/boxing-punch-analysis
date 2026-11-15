"""Run the FINAL punch counter on an external TEST clip and render an annotated video.

External footage is used for TESTING ONLY: nothing here is ever trained on it.

1. Model: the 7-class 1D-CNN (6 punches + No Punch, detect/dataset7, final-A no-punch rule) trained
   on ALL 10 of our videos, 3 seeds averaged. Cached in runs/final7/ (state dicts; loaded with
   weights_only=True).
2. Fighters: single-fighter clips (bag/BOB) follow the largest person; sparring clips use the
   identity tracker's output (external/track.py -> external/tracks/{id}.npz).
3. Stance per fighter BY THE MODEL: more punches found as-is or mirrored (10/10 own videos, LOVO).
   Southpaws are mirrored to orthodox before the model sees them, exactly like V4 in training, so
   "Jab" always means the LEAD hand.
4. Streaming counter: identical rules to detect/realtime_demo.py (causal 8/12/16-frame windows every
   2 frames, ON 0.6 / OFF 0.4, combination splitting).
5. Output runs/external/{id}/: summary.json (counts per fighter and type, every counted punch with its
   time) and {id}_annotated.mp4.
    /opt/anaconda3/bin/python external/run_test.py [ids...] [--seconds N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import mem_guard  # noqa: F401,E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from baselines import MIRROR_IDX  # noqa: E402
from deep import ARCHS, DEVICE, TorchSeqClassifier  # noqa: E402
from detect.build_frames import fill_low_vis  # noqa: E402
from detect.realtime_demo import EDGES, NP_, ON, SCALES, SHORT, STEP, events_from, window_batch  # noqa: E402
from detect.train7 import features  # noqa: E402
from external.stance import stance_track  # noqa: E402
from external.stance_by_model import mirror  # noqa: E402
from progress import progress  # noqa: E402
from punch_data import CLASSES  # noqa: E402

MP2COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
MODEL_DIR = ROOT / "runs" / "final7"
SEEDS = (0, 1, 2)
COLORS = [(40, 170, 255), (255, 200, 60)]            # BGR: fighter 1 orange, fighter 2 cyan-blue


# ------------------------------------------------------------------ model
class Final7:
    """Average of SEEDS 7-class CNNs trained on all 10 videos."""

    def __init__(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.members = []
        missing = [s for s in SEEDS if not (MODEL_DIR / f"cnn7_seed{s}.pt").exists()]
        if missing:
            d = np.load(ROOT / "detect" / "dataset7" / "clips.npz")
            F, y = features(d), d["label"]
            progress.start("final7 train (all 10 videos)", total=len(missing))
            for s in missing:
                t0 = time.time()
                m = TorchSeqClassifier("cnn", epochs=60, seed=s).fit(F, y)
                torch.save(dict(state=m.model_.state_dict(), mu=torch.tensor(m.mu_), sd=torch.tensor(m.sd_), classes=[str(c) for c in m.classes_],
                                in_ch=int(F.shape[2]), hidden=int(m.hidden), dropout=float(m.dropout), n_train=int(len(y))),
                           MODEL_DIR / f"cnn7_seed{s}.pt")
                print(f"  trained seed {s} on {len(y)} clips in {time.time() - t0:.0f}s", flush=True)
            progress.finish()
        for s in SEEDS:
            ck = torch.load(MODEL_DIR / f"cnn7_seed{s}.pt", weights_only=True)
            net = ARCHS["cnn"](ck["in_ch"], len(ck["classes"]), ck["hidden"], ck["dropout"]).to(DEVICE)
            net.load_state_dict(ck["state"]); net.eval()
            order = [ck["classes"].index(c) for c in CLASSES + [NP_]]
            self.members.append((net, ck["mu"].numpy(), ck["sd"].numpy(), order))

    @torch.no_grad()
    def predict_proba(self, X):
        """(n, 7) probabilities in CLASSES + [No Punch] order."""
        out = 0
        for net, mu, sd, order in self.members:
            Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=DEVICE)
            logits = torch.cat([net(Xt[i:i + 1024]) for i in range(0, len(Xt), 1024)])
            out = out + torch.softmax(logits, 1).cpu().numpy()[:, order]
        return out / len(self.members)


# ------------------------------------------------------------------ fighters
def single_fighter(pose):
    """Largest person in the first frame, then the person whose hips are nearest (like build_frames)."""
    img = pose["img"][:, :, MP2COCO]; F = len(img)
    xy, vis, world = img[..., :2], img[..., 3], pose["world"][:, :, MP2COCO]
    out = dict(xy=np.zeros((F, 17, 2), np.float32), world=np.zeros((F, 17, 3), np.float32),
               vis=np.zeros((F, 17), np.float32), present=np.zeros(F, bool))
    last = None
    for f in range(F):
        n = int(pose["npose"][f])
        if n == 0:
            continue
        hips = (xy[f, :n, 11] + xy[f, :n, 12]) / 2
        if last is None:
            size = [np.nanmax(xy[f, p, :, 1]) - np.nanmin(xy[f, p, :, 1]) for p in range(n)]
            p = int(np.argmax(size))
        else:
            p = int(np.argmin(np.linalg.norm(hips - last, axis=1)))
        last = hips[p]
        out["xy"][f], out["world"][f], out["vis"][f], out["present"][f] = xy[f, p], world[f, p], vis[f, p], True
    return [out]


def tracked_fighters(cid):
    t = np.load(HERE / "tracks" / f"{cid}.npz")
    return [dict(xy=t[f"f{k}_xy"], world=t[f"f{k}_world"], vis=t[f"f{k}_vis"], present=t[f"f{k}_present"]) for k in (1, 2)]


UNSURE = 0.10        # stance margin (a - b) / (a + b) below this -> "stance uncertain" (own videos: lowest V1 0.08, V10 0.12)


def prepare(model, fighter, W, H, fps):
    """Fill low-confidence 3D joints, then decide stance BY THE MODEL: count punches in the skeleton
    as-is and mirrored; the reading with more punches is the real stance (10/10 whole videos right on
    our own data, leave-one-video-out: external/stance_by_model.py). Southpaws are fed mirrored, like
    V4 in training. The geometric rule (external/stance.py) is kept as a second opinion."""
    present = fighter["present"]
    world = fill_low_vis(fighter["world"], fighter["vis"], present)
    as_is = dict(xy=fighter["xy"], world=world, vis=fighter["vis"], present=present)
    flipped = mirror(as_is)
    r_as, r_fl = stream(model, as_is), stream(model, flipped)
    a, b = len(r_as[3]), len(r_fl[3])
    margin = abs(a - b) / max(a + b, 1)
    # near-tie -> orthodox (most boxers are). Added 2026-11-13 AFTER B1 (BOB) came out a wrong
    # near-tie southpaw (314 vs 325; on-screen "DOUBLE JAB-CROSS-R.BODY HOOK" = orthodox), so B1 is
    # no longer an independent stance test.
    southpaw = b > a and margin >= UNSURE
    _, geo, _ = stance_track(fighter["xy"], world, fighter["vis"], present, W, H, fps)
    stance = dict(call="southpaw" if southpaw else "orthodox", punches_as_is=a, punches_mirrored=b,
                  margin=round(margin, 3), uncertain=bool(margin < UNSURE),
                  geometric_rule="southpaw" if geo < 0 else "orthodox", geometric_score=round(float(geo), 2))
    return (r_fl if southpaw else r_as), stance


def stream(model, fr):
    """Causal multi-scale windows every STEP frames -> probabilities and counted punches."""
    F = len(fr["present"])
    times = np.arange(max(SCALES) - 1, F, STEP)
    probs = np.zeros((len(times), 7), np.float32)
    for W in SCALES:
        batch, ok = window_batch(fr, times, W)
        P = np.tile(np.eye(7)[6], (len(times), 1)).astype(np.float32)      # nobody in the window -> no punch
        if ok.any():
            P[ok] = model.predict_proba(features({k: x[ok] for k, x in batch.items()}))
        probs += P / len(SCALES)
    events, sm = events_from(times, 1 - probs[:, 6], probs[:, :6])
    return times, probs, sm, events


# ------------------------------------------------------------------ render
def render(cid, pose, fighters_raw, results, out_path, width, n_frames):
    src = HERE / "raw" / f"{cid}.mp4"
    cap = cv2.VideoCapture(str(src)); frames = pose["frame"]
    pos = 0
    while pos < int(frames[0]):
        cap.grab(); pos += 1
    W0, H0 = int(pose["width"]), int(pose["height"]); fps = float(pose["fps"])
    Wd = width; Hd = int(round(Wd * H0 / W0))
    vw = None
    for codec in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (Wd, Hd))
        if vw.isOpened():
            break
    font = cv2.FONT_HERSHEY_SIMPLEX
    nf = len(results)
    counts = [{c: 0 for c in CLASSES} for _ in range(nf)]
    by_end = [{e["end"]: e for e in r["events"]} for r in results]
    last = [(None, -999)] * nf
    progress.start(f"render {cid}", total=n_frames)
    for i in range(n_frames):
        while pos < int(frames[i]):
            cap.grab(); pos += 1
        ok, img = cap.read(); pos += 1
        if not ok:
            break
        img = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
        overlay = img.copy()
        panel_w, panel_h = 250, 34 + 24 * 7
        for k in range(nf):
            x0 = 0 if k == 0 else Wd - panel_w
            cv2.rectangle(overlay, (x0, 0), (x0 + panel_w, panel_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, Hd - 26), (Wd, Hd), (20, 20, 20), -1)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
        for k, (raw, r) in enumerate(zip(fighters_raw, results)):
            col = COLORS[k]
            if i in by_end[k]:
                e = by_end[k][i]; counts[k][e["type"]] += 1; last[k] = (e, i)
            # skeleton (original, un-mirrored coordinates)
            if raw["present"][i]:
                p = raw["xy"][i] * [Wd, Hd]
                for a, b, _ in EDGES:
                    cv2.line(img, tuple(map(int, p[a])), tuple(map(int, p[b])), col, 2, cv2.LINE_AA)
                head = p[0].astype(int)
                j = np.searchsorted(r["times"], i, side="right") - 1
                tag = f"F{k + 1}" if nf > 1 else ""
                if j >= 0 and r["sm"][j] > ON:
                    tag += f" {SHORT[CLASSES[int(r['probs'][j, :6].argmax())]]}"
                if tag:
                    cv2.putText(img, tag.strip(), (head[0] - 30, max(head[1] - 40, 14)), font, 0.7, col, 2, cv2.LINE_AA)
            # counters
            x0 = 0 if k == 0 else Wd - panel_w
            title = (f"FIGHTER {k + 1} " if nf > 1 else "") + f"({r['stance']['call']}{'?' if r['stance']['uncertain'] else ''})"
            cv2.putText(img, title, (x0 + 10, 22), font, 0.55, col, 2, cv2.LINE_AA)
            cv2.putText(img, f"TOTAL {sum(counts[k].values())}", (x0 + 10, 46), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            le, lf = last[k]
            for t_i, c in enumerate(CLASSES):
                hl = le is not None and le["type"] == c and i - lf < fps * 0.6
                cv2.putText(img, f"{SHORT[c]:<14}{counts[k][c]:>4}", (x0 + 10, 70 + 24 * t_i), font, 0.52,
                            col if hl else (225, 225, 225), 2 if hl else 1, cv2.LINE_AA)
        cv2.putText(img, f"{cid} | EXTERNAL TEST CLIP (never trained on) | {i / fps:5.1f}s | model trained on our 10 videos only",
                    (10, Hd - 8), font, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        vw.write(img)
        if i % 200 == 0:
            progress.tick(200, note=f"frame {i}/{n_frames}")
    vw.release(); cap.release(); progress.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--seconds", type=float, default=None, help="only the first N seconds (quick look)")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--no-video", action="store_true", help="summary.json only (no annotated render)")
    args = ap.parse_args()
    man = json.loads((HERE / "manifest.json").read_text())
    ids = args.ids or [k for k in man if (HERE / "pose" / f"{k}.npz").exists()]
    model = Final7()
    for cid in ids:
        pose = dict(np.load(HERE / "pose" / f"{cid}.npz"))
        W, H, fps = float(pose["width"]), float(pose["height"]), float(pose["fps"])
        sparring = cid.startswith("A") or man[cid].get("category") == "O"      # O = Olympic bouts (olympic/)
        fighters_raw = tracked_fighters(cid) if sparring else single_fighter(pose)
        n = len(pose["frame"]) if args.seconds is None else min(len(pose["frame"]), int(args.seconds * fps))
        results = []
        for k, raw in enumerate(fighters_raw):
            raw = {key: v[:n] for key, v in raw.items()}; fighters_raw[k] = raw
            (times, probs, sm, events), stance = prepare(model, raw, W, H, fps)
            c = {t: sum(e["type"] == t for e in events) for t in CLASSES}
            results.append(dict(times=times, probs=probs, sm=sm, events=events, stance=stance, counts=c,
                                present=float(raw["present"].mean())))
            print(f"[{cid}] fighter {k + 1}: {stance['call']}{' (UNCERTAIN)' if stance['uncertain'] else ''} "
                  f"(punches as-is {stance['punches_as_is']} vs mirrored {stance['punches_mirrored']}; geometric rule says {stance['geometric_rule']}) | seen in {raw['present'].mean():.0%} "
                  f"of frames | {len(events)} punches counted: " + ", ".join(f"{SHORT[t]} {v}" for t, v in c.items()), flush=True)
        out = ROOT / "runs" / "external" / cid; out.mkdir(parents=True, exist_ok=True)
        summ = dict(clip=cid, fps=fps, seconds=n / fps, url=man[cid].get("url"), segment=man[cid].get("segment"),
                    fighters=[dict(fighter=k + 1, stance=r["stance"], seen_fraction=r["present"], counts=r["counts"],
                                   total=len(r["events"]),
                                   events=[dict(e, t_sec=round(e["start"] / fps, 2)) for e in r["events"]])
                              for k, r in enumerate(results)])
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        if args.no_video:
            continue
        path = out / f"{cid}_annotated.mp4"
        render(cid, pose, fighters_raw, results, path, args.width, n)
        print(f"[{cid}] video: {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
