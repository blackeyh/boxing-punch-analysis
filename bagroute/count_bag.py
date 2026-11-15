"""BAG / BOB ROUTE (separate from the sparring chain): one boxer hitting a bag or a BOB dummy.

  skeletons (RTMPose, external/pose_rtm/{id}.npz)  +  target detections (OWLv2 every 2 s, runs/bagroute/*_det.json)
  -> the TARGET = the persistent detected place the boxer's gloves keep reaching (never "only one person seen")
  -> the BOXER  = the largest skeleton that is not the target
  -> the target's box FOLLOWS it between detections (a heavy bag swings; the bag nearest the boxer is the one being hit)
  -> punches + type: the sparring punch detector's no-opponent version (runs/punch-tcn-noopp, BoxingWeb, 3 seeds)
     + a rescue rule: a glove coming fast into the target outside every counted punch is a punch too (the detector misses many bag jabs)
  -> contact moment = greatest extension of the punching arm inside the counted punch
  -> TOUCHED or not: (a) reach: the glove point gets inside the target outline around the contact moment  = touched
                     (b) reaction: the target's pixels move more just after the contact than just before  = "clear"
     TOUCHED = reach + clear reaction; "touched, soft" = reach only (a swinging bag hides the reaction); both count as
     touched. "no touch" = the glove never reaches the target: a miss or a false punch count.
Nothing is trained here and there are no labels: thresholds are fixed by looking at the distributions, and the
result is checked by eye on contact sheets (runs/bagroute/{id}_check.jpg) and in the rendered video.

    python bagroute/count_bag.py ID [--no-render]     or     python pipeline/count_video.py ID --bob
    ID = any clip with external/raw/ID.mp4 and external/pose_rtm/ID.npz (the bag is detected first if runs/bagroute/ID_det.json is missing)
"""
import argparse, glob, json, os, subprocess, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for d in ("", "boxingweb", "pipeline"):
    sys.path.insert(0, os.path.join(ROOT, d))
import mem_guard  # noqa: F401
import cv2, numpy as np, torch
from progress import progress
import punch_data as pd, punch_tcn as pt
OUT = os.path.join(ROOT, "runs", "bagroute")
SEG = {"B1_bob_seg": ("B1_bob_miri", 15.0), "C1_heavybag_seg": ("C1_heavybag_miri", 20.0), "C2_standingbag_seg": ("C2_standingbag_gosia", 280.0)}   # segment -> (raw clip with detections, start s)
USE = (0, 2, 4)                                              # prompts used: "a punching bag", "a free standing punching bag", "a training dummy torso"
NAMES = ["", "L straight", "R straight", "L hook", "R hook", "L uppercut", "R uppercut"]
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
REACH_M, REACT_T = 0.15, 1.35                                # glove may be this share of the target width outside the box; reaction ratio for "clear reaction"

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(x2 - x1, 0) * max(y2 - y1, 0)
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9)

def places(cid, dur):
    """Persistent detected places inside the segment: [(times, boxes (n,4), median box, share present, median score)]."""
    raw, t0 = SEG.get(cid, (cid, 0.0)); own = os.path.join(OUT, cid + "_det.json")            # the segment's own detections (every 0.5 s) if they exist, else the raw clip's (every 2 s)
    if os.path.exists(own):
        raw, t0 = cid, 0.0
    D = json.load(open(os.path.join(OUT, raw + "_det.json"))); fr = [f for f in D["frames"] if t0 <= f["t"] < t0 + dur]; cl = []
    for f in fr:
        for d in f["dets"]:
            if d["p"] in USE and d["s"] >= 0.2:
                b = np.array(d["b"]); best = max(cl, key=lambda c: iou(np.median(c["b"], 0), b), default=None)
                if best is not None and iou(np.median(best["b"], 0), b) >= 0.5:
                    if best["t"][-1] != f["t"] - t0:
                        best["t"].append(f["t"] - t0); best["b"].append(b); best["s"].append(d["s"])
                else:
                    cl.append(dict(t=[f["t"] - t0], b=[b], s=[d["s"]]))
    return [(np.array(c["t"]), np.array(c["b"]), np.median(c["b"], 0), len(c["t"]) / max(len(fr), 1), float(np.median(c["s"]))) for c in cl if len(c["t"]) / max(len(fr), 1) >= 0.4]

def follow(cid, dur, tg, bx_t):
    """The target's box at every detection time, FOLLOWING it when it swings. The persistent place only says WHICH object
    is the target; a heavy bag leaves that place for seconds, and a second bag hanging behind it stays there (seen on C1:
    the box sat on the back bag while the punches landed on the swinging one). Per sampled frame: detections of about the
    target's size whose centre is within 1.5 target widths of the place; among those scoring at least 0.7 x the best,
    the one nearest the boxer (bx_t(t) = the boxer's shoulder-centre x): the bag being hit is the one in front of him."""
    raw, t0 = SEG.get(cid, (cid, 0.0))
    if os.path.exists(os.path.join(OUT, cid + "_det.json")):
        raw, t0 = cid, 0.0
    D = json.load(open(os.path.join(OUT, raw + "_det.json"))); med = tg[2]; mw, mh = med[2] - med[0], med[3] - med[1]; mc = np.array([(med[0] + med[2]) / 2, (med[1] + med[3]) / 2]); ts, bs = [], []; away = 0
    for f in [f for f in D["frames"] if t0 <= f["t"] < t0 + dur]:
        cand = {}
        for d in f["dets"]:
            b = np.array(d["b"], float)
            if d["p"] in USE and d["s"] >= 0.2 and 0.6 <= (b[2] - b[0]) / mw <= 1.6 and 0.6 <= (b[3] - b[1]) / mh <= 1.6 and np.linalg.norm([(b[0] + b[2]) / 2 - mc[0], (b[1] + b[3]) / 2 - mc[1]]) <= 1.5 * mw:
                k = tuple(np.round(b).astype(int)); cand[k] = max(cand.get(k, 0), d["s"])
        if not cand:
            continue
        top = max(cand.values()); good = [np.array(k, float) for k, v in cand.items() if v >= 0.7 * top]; x = bx_t(f["t"] - t0)
        pick = min(good, key=lambda b: max(b[0] - x, x - b[2], 0)) if np.isfinite(x) else max(cand, key=cand.get); pick = np.array(pick, float)
        ts.append(f["t"] - t0); bs.append(pick); away += iou(pick, med) < 0.5
    return np.array(ts), np.array(bs), away / max(len(ts), 1)

def glove(K, w):
    return K[..., w, :] + 0.4 * (K[..., w, :] - K[..., w - 2, :])

def inside(g, b, m=0.0):
    wd = (b[..., 2] - b[..., 0]) * m
    return (g[..., 0] >= b[..., 0] - wd) & (g[..., 0] <= b[..., 2] + wd) & (g[..., 1] >= b[..., 1] - wd) & (g[..., 1] <= b[..., 3] + wd)

def analyse(cid):
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); kp, sc, fr, bx = R["kp"].astype(np.float64), R["score"], R["frame"], R["box"]; F = len(fr); fps = float(R["fps"]); W = float(R["width"])
    P = places(cid, F / fps); assert P, "no persistent bag / dummy place detected: this video does not go to the bag route"
    # boxer = per frame the largest skeleton that does not coincide with any persistent place (a BOB can be seen as a person), with legs seen
    role = np.zeros((F, kp.shape[1]), np.int8); nper = np.zeros(F, int); bob_as_person = 0
    for i in range(F):
        cand = []
        for p in range(int(R["n"][i])):
            if (sc[i, p] > 0.3).sum() < 8:
                continue
            pa = max((bx[i, p, 2] - bx[i, p, 0]) * (bx[i, p, 3] - bx[i, p, 1]), 1.0)                     # a "person" that sits on a persistent detected place is the dummy / bag, not a boxer
            def cover(b):                                    # share of the detected place that lies inside this person's box (the pose model draws a BOB with its stand as "legs")
                return max(min(bx[i, p, 2], b[2]) - max(bx[i, p, 0], b[0]), 0) * max(min(bx[i, p, 3], b[3]) - max(bx[i, p, 1], b[1]), 0) / max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
            if any(cover(pl[2]) >= 0.7 and (bx[i, p, 2] - bx[i, p, 0]) <= 1.6 * (pl[2][2] - pl[2][0]) for pl in P):
                bob_as_person += 1; continue
            cand.append(((bx[i, p, 2] - bx[i, p, 0]) * (bx[i, p, 3] - bx[i, p, 1]), p))
        nper[i] = len(cand)
        if cand:
            role[i, max(cand)[1]] = 1
    K, C, M = pd.stream(kp, sc, role, 1); G = np.stack([glove(K, 9), glove(K, 10)], 1)                                   # (F, 2, 2)
    # the target = the place the gloves reach most (share of seconds with a glove inside)
    def reach_share(pl):
        ins = inside(G, pl[2][None, None], REACH_M).any(1); return float(np.mean([ins[int(s * fps):int((s + 1) * fps)].any() for s in range(int(F / fps))]))
    rs = [reach_share(pl) for pl in P]; tg = P[int(np.argmax(rs))]; hb = np.nanmedian(np.nanmax(K[..., 1], 1) - np.nanmin(K[..., 1], 1))
    shx = np.nanmean(K[:, [5, 6], 0], 1); ft, fb, away = follow(cid, F / fps, tg, lambda t: shx[int(np.clip(t * fps, 0, F - 1))]); tb = np.stack([np.interp(np.arange(F) / fps, ft, fb[:, j]) for j in range(4)], 1)       # target box per frame, following the swing
    info = dict(id=cid, seconds=F / fps, places=[dict(box=[round(float(v)) for v in pl[2]], present=round(pl[3], 2), score=round(pl[4], 2), reach=round(r, 2)) for pl, r in zip(P, rs)],
                target=dict(box=[round(float(v)) for v in tg[2]], reach=round(max(rs), 2), present=round(tg[3], 2), followed=round(len(ft) / max(F / fps * 2, 1), 2), away_from_place=round(float(away), 2), height_vs_boxer=round(float((tg[2][3] - tg[2][1]) / hb), 2)),
                boxer_present=float(M.mean()), other_people=dict(zero=float((nper == 0).mean()), one=float((nper == 1).mean()), two_plus=float((nper >= 2).mean())),
                bob_seen_as_person_frames=float(bob_as_person / F))
    # punches: no-opponent detector
    X = pd.feats(K, C, M, np.full_like(K, np.nan), np.zeros(F, bool))[None]; res = sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn-noopp", "results_seed*.json")))
    thr = float(np.mean([json.load(open(f))["threshold"] for f in res])); models = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn-noopp", "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); models.append(m)
    Xn = X.copy(); Xn[..., pt.OPP] = 0; Xn[..., -1] = 0; Pm = np.mean([pt.predict(m, Xn) for m in models], 0)[0]; punches = []
    for a, e, c in pt.events(Pm, M, thr):
        w = 9 if c % 2 == 1 else 10; h = 0 if w == 9 else 1; a2, e2 = max(a - 2, 0), min(e + 2, F - 1); ext = np.linalg.norm(K[a2:e2 + 1, w] - K[a2:e2 + 1, w - 4], axis=1)
        t = a2 + (int(np.nanargmax(ext)) if np.isfinite(ext).any() else (e2 - a2) // 2); win = slice(max(t - 2, 0), min(t + 3, F))
        g = G[win, h]; b = tb[win]; wd = b[:, 2] - b[:, 0]
        dx = np.maximum(np.maximum(b[:, 0] - g[:, 0], g[:, 0] - b[:, 2]), 0); dy = np.maximum(np.maximum(b[:, 1] - g[:, 1], g[:, 1] - b[:, 3]), 0); dist = np.sqrt(dx ** 2 + dy ** 2) / wd
        punches.append(dict(cls=int(c), type=NAMES[c], i0=int(a), i1=int(e), ic=int(t), tc=float(t / fps), hand=h, dist=float(np.nanmin(dist)) if np.isfinite(dist).any() else None,
                            gx=float(G[t, h, 0]) if np.isfinite(G[t, h]).all() else None, gy=float(G[t, h, 1]) if np.isfinite(G[t, h]).all() else None, conf=float((1 - Pm[a:e + 1, 0]).mean())))
    # RESCUE (user: "many of the punches that land on it don't count"; contact sheets: the detector, trained on bouts, misses many
    # jabs on a bag): a glove that comes fast INTO the target, moving towards it, outside every counted punch, is a punch.
    cov = np.zeros(F, bool); wdt = tb[:, 2] - tb[:, 0]; bc = np.stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2], 1); cand = []
    for p in punches:
        cov[max(p["i0"] - 6, 0):p["i1"] + 7] = True
    for h in (0, 1):
        ins = inside(G[:, h], tb, REACH_M); v = np.r_[np.zeros((1, 2)), np.diff(G[:, h], axis=0)] * fps / wdt[:, None]; sp = np.nan_to_num(np.linalg.norm(v, axis=1)); toward = np.nan_to_num((v * (bc - G[:, h])).sum(1)) > 0; last = -99
        for i in range(6, F - 1):
            if ins[i] and not ins[i - 6:i].any() and sp[i - 6:i + 1].max() > 2.0 and toward[i - 3:i + 1].any() and not cov[i] and i - last > 8:
                cand.append((i, h, float(sp[i - 6:i + 1].max()))); last = i
    cand.sort(); keep = []
    for c in cand:                                             # both gloves arriving together = one punch: keep the faster glove
        if keep and c[0] - keep[-1][0] <= 4:
            if c[2] > keep[-1][2]:
                keep[-1] = c
        else:
            keep.append(c)
    for i, h, _ in keep:
        w = 9 + h; a, e = max(i - 4, 0), min(i + 4, F - 1); ext = np.linalg.norm(K[a:e + 1, w] - K[a:e + 1, w - 4], axis=1); t = a + (int(np.nanargmax(ext)) if np.isfinite(ext).any() else i - a)
        fam = int(Pm[a:e + 1, 1:].sum(0)[[h, h + 2, h + 4]].argmax()); c = 1 + h + 2 * fam
        punches.append(dict(cls=int(c), type=NAMES[c], i0=int(a), i1=int(e), ic=int(t), tc=float(t / fps), hand=h, dist=0.0, gx=float(G[t, h, 0]) if np.isfinite(G[t, h]).all() else None,
                            gy=float(G[t, h, 1]) if np.isfinite(G[t, h]).all() else None, conf=float((1 - Pm[a:e + 1, 0]).mean()), rescued=True))
    punches.sort(key=lambda p: p["tc"]); info["rescued"] = len(keep)
    return info, punches, K, M, tb, fr, fps

def reaction(cid, punches, K, tb, fr):
    """Target motion just after vs just before each contact: mean |frame difference| inside the target box, leaving out
    a band around the boxer's arms and head (their motion is not the target's)."""
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); step = int(fr[1] - fr[0]); need = {}
    for n, p in enumerate(punches):
        for j in range(-9, 9):
            i = p["ic"] + j
            if 0 <= i < len(fr):
                need.setdefault(i, []).append((n, j))
    E = np.full((len(punches), 18), np.nan); prev = None; f = 0; last = max(need) if need else -1
    for i in range(last + 1):
        while f < fr[i]:
            cap.grab(); f += 1
        ok, im = cap.read(); f += 1
        if not ok:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        if i in need and prev is not None:
            x1, y1, x2, y2 = (int(v) for v in np.clip(tb[i], 0, [g.shape[1], g.shape[0], g.shape[1], g.shape[0]])); d = np.abs(g - prev)[y1:y2, x1:x2]
            if d.size:
                mask = np.ones(d.shape, np.uint8); S = np.linalg.norm(K[i, [5, 6]].mean(0) - K[i, [11, 12]].mean(0)); r = int(max(0.35 * (S if np.isfinite(S) else 60), 12))
                for a, b in ((5, 7), (7, 9), (6, 8), (8, 10)):
                    if np.isfinite(K[i, [a, b]]).all():
                        pa = K[i, a]; pb = K[i, b] + (0.5 * (K[i, b] - K[i, a]) if b in (9, 10) else 0)
                        cv2.line(mask, (int(pa[0] - x1), int(pa[1] - y1)), (int(pb[0] - x1), int(pb[1] - y1)), 0, 2 * r)
                if np.isfinite(K[i, 0]).all():
                    cv2.circle(mask, (int(K[i, 0, 0] - x1), int(K[i, 0, 1] - y1)), 2 * r, 0, -1)
                v = float(d[mask > 0].mean()) if (mask > 0).mean() > 0.2 else float(d.mean())
                for n, j in need[i]:
                    E[n, j + 9] = v
        prev = g
    cap.release()
    for n, p in enumerate(punches):
        before, after = np.nanmean(E[n, 1:6]), np.nanmax(E[n, 9:15]) if np.isfinite(E[n, 9:15]).any() else np.nan       # frames -8..-4 vs 0..+5
        p["react"] = float(after / max(before, 0.3)) if np.isfinite(before) and np.isfinite(after) else None
    return punches

def decide(punches):
    for p in punches:
        reach = p["dist"] is not None and p["dist"] <= REACH_M; react = p["react"] is not None and p["react"] >= REACT_T
        p["reach"], p["reacted"] = bool(reach), bool(react); p["verdict"] = "TOUCHED" if reach and react else ("touched, soft" if reach else "no touch")
    return punches

def summary(info, punches):
    n = len(punches); t = sum(p["verdict"] == "TOUCHED" for p in punches); q = sum(p["verdict"] == "touched, soft" for p in punches)
    L = [f"{info['id']}: {info['seconds'] / 60:.1f} min | target box {info['target']['box']} (on its usual place {info['target']['present']:.0%} of the time, followed away from it {info['target'].get('away_from_place', 0):.0%}, gloves reach it in {info['target']['reach']:.0%} of seconds, height {info['target']['height_vs_boxer']} x boxer)",
         f"  boxer skeleton present {info['boxer_present']:.0%}; other people: none {info['other_people']['zero']:.0%} / one (the boxer) {info['other_people']['one']:.0%} / two or more {info['other_people']['two_plus']:.0%}; skeletons drawn on a detected bag / dummy and ignored: {info['bob_seen_as_person_frames']:.1f} per frame",
         f"  punches counted {n} ({n / info['seconds'] * 60:.0f} per minute): touched {t + q} ({(t + q) / max(n, 1):.0%}) = with a clear reaction {t} + soft {q}; no touch {n - t - q} ({(n - t - q) / max(n, 1):.0%}); {info.get('rescued', 0)} of the counted punches were added by the glove-reach rule (missed by the punch detector)",
         "  " + " ".join(f"{x:>11s}" for x in ["type"] + NAMES[1:])]
    for row, sel in (("thrown", lambda p: True), ("touched", lambda p: p["verdict"] != "no touch")):
        L.append("  " + f"{row:>11s} " + " ".join(f"{sum(p['cls'] == c and sel(p) for p in punches):>11d}" for c in range(1, 7)))
    return "\n".join(L)

def check_sheet(cid, punches, K, tb, fr, n=24):
    """Contact sheet for judging by eye: for n evenly spaced punches, the frames 3 before, at, and 3 after the contact."""
    sel = [punches[i] for i in np.linspace(0, len(punches) - 1, min(n, len(punches))).astype(int)]; cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); rows = []
    for p in sel:
        tiles = []
        for j in (-3, 0, 3):
            i = int(np.clip(p["ic"] + j, 0, len(fr) - 1)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[i])); ok, im = cap.read()
            if not ok:
                im = np.zeros((720, 1280, 3), np.uint8)
            b = tb[i]; cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 255), 2); w = 9 if p["hand"] == 0 else 10
            if np.isfinite(K[i, w]).all():
                g = glove(K[i], w); cv2.circle(im, (int(g[0]), int(g[1])), 14, (255, 0, 255), 2)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = int(max(b[2] - b[0], b[3] - b[1]) * 0.9); x1, y1 = int(np.clip(cx - s, 0, im.shape[1] - 1)), int(np.clip(cy - s, 0, im.shape[0] - 1))
            tiles.append(cv2.resize(im[y1:int(cy + s), x1:int(cx + s)], (300, 300)))
        row = np.hstack(tiles); col = (80, 220, 80) if p["verdict"] == "TOUCHED" else ((0, 200, 255) if p["verdict"] == "touched, soft" else (170, 170, 170))
        cv2.putText(row, f"{p['tc']:.1f}s {p['type']} {p['verdict']} d={p['dist'] if p['dist'] is None else round(p['dist'], 2)} r={p['react'] if p['react'] is None else round(p['react'], 2)}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(row, f"{p['tc']:.1f}s {p['type']} {p['verdict']} d={p['dist'] if p['dist'] is None else round(p['dist'], 2)} r={p['react'] if p['react'] is None else round(p['react'], 2)}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2); rows.append(row)
    half = (len(rows) + 1) // 2; left, right = np.vstack(rows[:half]), np.vstack(rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:])))
    cv2.imwrite(os.path.join(OUT, cid + "_check.jpg"), np.hstack([left, right]), [cv2.IMWRITE_JPEG_QUALITY, 80])

def render(cid, punches, K, M, tb, fr, fps):
    """Annotated video in the sparring renderer's style: boxer skeleton, target box, impact effects at the glove, screen shake, synthesised sound,
    a health bar for the BOXER (stays full: nothing hits back) and one for the TARGET (100 points, fixed damage per touched punch by type as in the sparring route; at zero: K.O., counted, bar refills)."""
    import count_video as cvv
    DMG = cvv.DMG; touched = lambda p: p["verdict"] != "no touch"; name = "BOB" if "bob" in cid.lower() else "BAG"
    snd = [dict(t_hit=p["tc"], cls=p["cls"], verdict="LANDED" if touched(p) else "missed") for p in punches]
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); W, H = int(cap.get(3)), int(cap.get(4)); raw = os.path.join(OUT, cid + "_raw.mp4"); FONT = cv2.FONT_HERSHEY_SIMPLEX
    z = W / 1280; BAR = int(124 * z); S = lambda v: int(round(v * z)); rng = np.random.default_rng(1)
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)); f = 0; progress.start(f"render {cid}", total=len(fr))
    hits = sorted((p["tc"], DMG[p["cls"]]) for p in punches if touched(p)); cum = np.cumsum([d for _, d in hits]); ht = np.array([h for h, _ in hits])
    ko_t = [ht[int(np.searchsorted(cum, 100 * (n + 1)))] for n in range(int(cum[-1] // 100))] if len(cum) else []      # moments the bar reaches zero
    def dealt(t):
        n = int(np.searchsorted(ht, t, side="right")); return float(cum[n - 1]) if n else 0.0
    def health(t):
        d = dealt(t); return 100.0 - (d % 100.0) if d % 100.0 or d == 0 else 0.0
    def txt(im, s, x, y, sc_, col, th=2):
        cv2.putText(im, s, (x, y), FONT, sc_ * z, (0, 0, 0), max(int((th + 3) * z), 1), cv2.LINE_AA); cv2.putText(im, s, (x, y), FONT, sc_ * z, col, max(int(round(th * z)), 1), cv2.LINE_AA)
    for i in range(len(fr)):
        while f < fr[i]:
            cap.grab(); f += 1
        ok, im = cap.read(); f += 1
        if not ok:
            break
        t = i / fps; b = tb[i]; shake = 0.0; hit = min([t - p["tc"] for p in punches if touched(p) and 0 <= t - p["tc"] <= 0.25], default=None)
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 60, 255) if hit is not None else (0, 255, 255), S(5) if hit is not None else S(2)); txt(im, name, int(b[0]) + S(4), int(b[1]) + S(24), 0.7, (0, 255, 255), 2)
        if M[i]:
            for a, c in EDGES:
                if np.isfinite(K[i, [a, c]]).all():
                    cv2.line(im, tuple(int(v) for v in K[i, a]), tuple(int(v) for v in K[i, c]), (0, 140, 255), S(2), cv2.LINE_AA)
        hx = int(np.nanmean(K[i, :, 0])) if M[i] else W // 2; hy = int(np.nanmin(K[i, :, 1])) if M[i] else S(200)
        for p in punches:
            dt = t - p["tc"]
            if not 0 <= dt <= 1.2:
                continue
            d = DMG[p["cls"]]; tc_ = touched(p)
            if p["gx"] is not None and dt <= 0.45:
                c = (int(p["gx"]), int(p["gy"])); u = dt / 0.45; ov = im.copy()
                if tc_:
                    if dt <= 0.07:
                        cv2.circle(ov, c, S(26 + 4 * d), (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(ov, c, S(14 + u * (60 + 10 * d)), (60, 200, 255), S(5), cv2.LINE_AA); cv2.circle(ov, c, S(8 + u * (34 + 6 * d)), (255, 255, 255), S(3), cv2.LINE_AA)
                    for q in range(10):
                        ang = q * 0.628 + p["tc"]; r0, r1 = S(18 + u * 50), S(30 + u * (80 + 8 * d))
                        cv2.line(ov, (int(c[0] + r0 * np.cos(ang)), int(c[1] + r0 * np.sin(ang))), (int(c[0] + r1 * np.cos(ang)), int(c[1] + r1 * np.sin(ang))), (80, 230, 255), S(3), cv2.LINE_AA)
                    if dt < 0.18:
                        shake = max(shake, (1 - dt / 0.18) * (2 + d) * z)
                else:
                    for q in (-1, 0, 1):
                        cv2.ellipse(ov, (c[0], c[1] + q * S(14)), (S(30 + u * 70), S(10)), 0, 200, 340, (200, 200, 200), S(2), cv2.LINE_AA)
                im = cv2.addWeighted(ov, 1 - u ** 1.5, im, u ** 1.5, 0)
            txt(im, f"{p['type']}: {'TOUCHED' if tc_ else 'no touch'}", int(np.clip(hx - S(120), 5, W - S(330))), max(hy - S(24) - S(dt * 25), BAR + S(28)), 0.75, (80, 220, 80) if tc_ else (170, 170, 170))
            if tc_ and dt <= 0.9:                                                               # floating damage number over the target
                txt(im, f"-{d}", int((b[0] + b[2]) / 2) + S(10), max(int(b[1]) - S(10) - S(dt * 60), BAR + S(28)), 1.0, (60, 60, 255), 3)
        if shake > 0.5:
            dx, dy = (int(v) for v in rng.integers(-int(shake), int(shake) + 1, 2)); im = cv2.warpAffine(im, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
        kos = sum(k <= t for k in ko_t); since = min([t - k for k in ko_t if k <= t], default=9)
        if since <= 1.5:
            txt(im, "K.O.!", W // 2 - S(150), H // 2, 3.2, (60, 60, 255), 8)
        # ---- top bar: target health, counts
        ov = im.copy(); cv2.rectangle(ov, (0, 0), (W, BAR), (0, 0, 0), -1); im = cv2.addWeighted(ov, 0.62, im, 0.38, 0); done = [p for p in punches if p["tc"] <= t]; tch = [p for p in done if touched(p)]
        hp = 0.0 if since <= 1.5 else health(t); ghost = max(100.0 - (dealt(t - 0.6) % 100.0), hp) if since > 2.1 else hp; lastd = min([t - h for h in ht if h <= t], default=9)
        txt(im, "BOXER", S(14), S(30), 0.8, (0, 140, 255)); bx0, bx1 = S(110), W // 2 - S(60)                         # the boxer's bar (user's request): shown although nothing hits back
        cv2.rectangle(im, (bx0, S(12)), (bx1, S(34)), (40, 220, 60), -1); cv2.rectangle(im, (bx0, S(12)), (bx1, S(34)), (220, 220, 220), 1); txt(im, "100", bx1 - S(44), S(30), 0.6, (255, 255, 255), 1)
        txt(im, name, W // 2 - S(40), S(30), 0.8, (0, 255, 255)); x0, x1, y0, y1 = W // 2 + S(30), W - S(190), S(12), S(34); wbar = x1 - x0
        cv2.rectangle(im, (x0, y0), (x1, y1), (40, 40, 40), -1); cv2.rectangle(im, (x0, y0), (x0 + int(wbar * ghost / 100), y1), (230, 230, 255), -1)
        g = hp / 100; colh = (40, int(60 + 180 * min(g * 2, 1)), int(60 + 180 * min((1 - g) * 2, 1))); cv2.rectangle(im, (x0, y0), (x0 + int(wbar * hp / 100), y1), colh, -1)
        cv2.rectangle(im, (x0, y0), (x1, y1), (60, 60, 255) if lastd < 0.2 else (220, 220, 220), S(4) if lastd < 0.2 else 1)
        txt(im, f"{hp:.0f}" if hp > 0 else "K.O.", x1 - S(44 if hp > 0 else 64), S(30), 0.6, (255, 255, 255) if hp > 0 else (60, 60, 255), 1 if hp > 0 else 2)
        txt(im, f"K.O. x{kos}", W - S(170), S(32), 0.8, (60, 60, 255) if kos else (150, 150, 150))
        txt(im, f"TOUCHED {len(tch)}   thrown {len(done)}   no touch {len(done) - len(tch)}   damage dealt {dealt(t):.0f}", S(90), S(66), 0.66, (0, 140, 255))
        n = [sum(p["cls"] == c for p in tch) for c in range(1, 7)]; txt(im, f"touched:  straight L{n[0]} R{n[1]}   hook L{n[2]} R{n[3]}   uppercut L{n[4]} R{n[5]}", S(90), S(96), 0.58, (230, 230, 230), 1)
        txt(im, f"{int(t // 60)}:{t % 60:04.1f}", W - S(100), S(116), 0.5, (255, 255, 255), 1); vw.write(im)
        if i % 100 == 0:
            progress.tick(100, note=f"frame {i}/{len(fr)}")
    vw.release(); cap.release(); progress.finish(); wav = os.path.join(OUT, cid + "_fx.wav"); dst = os.path.join(OUT, cid + ".mp4"); cvv.sounds(snd, 0.0, len(fr) / fps, wav)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", dst], check=True)
    os.remove(raw); os.remove(wav); return dst

def run(cid, no_render=False):
    """The whole bag / BOB route on one clip (also reached by: python pipeline/count_video.py ID --bob)."""
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(OUT, cid + "_det.json")) and cid not in SEG:      # a new clip: find the bag / dummy first (OWLv2 every 0.5 s)
        print("detecting the bag / dummy (every 0.5 s) ...", flush=True); subprocess.run([sys.executable, os.path.join(HERE, "detect_target.py"), cid, "--every", "0.5"], check=True)
    info, punches, K, M, tb, fr, fps = analyse(cid); punches = decide(reaction(cid, punches, K, tb, fr)); s = summary(info, punches); print(s)
    d = np.array([p["dist"] for p in punches if p["dist"] is not None]); r = np.array([p["react"] for p in punches if p["react"] is not None])
    print(f"  glove-to-target distance (share of target width) 10/50/90 pct: {np.percentile(d, [10, 50, 90]).round(2).tolist()} | reaction ratio 10/50/90 pct: {np.percentile(r, [10, 50, 90]).round(2).tolist()}")
    json.dump(dict(info=info, punches=punches), open(os.path.join(OUT, cid + "_punches.json"), "w"), indent=0); open(os.path.join(OUT, cid + "_summary.txt"), "w").write(s + "\n"); check_sheet(cid, punches, K, tb, fr)
    if not no_render:
        print("video:", render(cid, punches, K, M, tb, fr, fps))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("--no-render", action="store_true"); a = ap.parse_args(); run(a.id, a.no_render)

if __name__ == "__main__":
    main()
