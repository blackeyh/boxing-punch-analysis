"""THE WHOLE CHAIN ON ONE VIDEO, rendered for judging by eye.

  RTMPose skeletons (external/pose_rtm) -> learned boxer selector -> identity A / B (OSNet-AIN, self-updating)
  -> per-boxer streams -> punch detector + type (TCN, BoxingWeb, 3 seeds) -> rough contact moment (skeleton contact
  model, Olympic, trained here on all 29 bouts) -> 7-position pixel scan (13 positions gave the same accuracy: 66.5% vs 66.3%) with the contact + landed video model
  (VideoMAE-Large, runs/landed_model/cl_all_large.pt) -> landed / blocked / missed per counted punch.

Nothing here was trained on the video being judged. Clips for the landed model are resampled to the 50 fps time base
it was trained on (nearest video frame), whatever the video's frame rate.

Outputs in runs/pipeline_demo/: {id}_punches.json (every counted punch), {id}_summary.txt, {id}.mp4 (annotated).
Cosmetics in the video (user's request): a health bar per boxer (a landed punch drains the boxer who was hit, by punch
type; fixed points out of 100, so the bar can reach zero before the end), an impact effect for landed, a shield for blocked, streaks for missed, and synthesised sounds.
Stages are cached: delete {id}_punches.json / {id}_scan.npz to redo them.
    python pipeline/count_video.py ID [--no-render] [--start S --seconds N]
    python pipeline/count_video.py ID --bob [--no-render]      one boxer on a BOB dummy / bag: the bag route (bagroute/count_bag.py, outputs in runs/bagroute/); --bag is the same
"""
import argparse, glob, json, os, subprocess, sys, time, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for d in ("", "boxingweb", "olympic"):
    sys.path.insert(0, os.path.join(ROOT, d))
import mem_guard  # noqa: F401
import cv2, numpy as np, torch
from progress import progress
import punch_data as pd, punch_tcn as pt, contact_tcn as ct
OUT = os.path.join(ROOT, "runs", "pipeline_demo"); THR, GROW = 0.40, 4
OFFS = list(range(-12, 13, 4)); G0, GN = 20, 40                       # stored clip: 40 frames on a 50 fps grid, contact at index 20
NAMES = ["", "L straight", "R straight", "L hook", "R hook", "L uppercut", "R uppercut"]; VERD = ["LANDED", "blocked", "missed"]
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COL = {0: (0, 140, 255), 1: (255, 220, 0)}; VCOL = [(80, 220, 80), (0, 200, 255), (170, 170, 170)]         # boxer A orange, B cyan (BGR)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def contact_models():
    """Skeleton contact model trained on ALL Olympic bouts (the scored versions each left one gym out); "allp" = corrected puncher rule."""
    d = os.path.join(ROOT, "runs", "contact-tcn"); os.makedirs(d, exist_ok=True); fs = [os.path.join(d, f"model_allp_seed{s}.pt") for s in (0, 1)]
    if not all(os.path.exists(f) for f in fs):
        ev = json.load(open(os.path.join(ROOT, "olympic", "events.json")))
        ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "chain_streams", "O_*.npz"))); D = [ct.load(c, ev) for c in ids]
        for s, f in enumerate(fs):
            if not os.path.exists(f):
                torch.save(ct.train(D, seed=s).state_dict(), f)
    out = []
    for f in fs:
        m = pt.TCN(157, ncls=2).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); out.append(m)
    return out

def stage_punches(cid):
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); kp, sc, fr = R["kp"].astype(np.float64), R["score"], R["frame"]; F = len(fr); W = float(R["width"])
    idf = os.path.join(ROOT, "runs", "identity_sparring", cid + ".npz")
    if not os.path.exists(idf):
        import identity_sparring as isp; isp.main(cid)
    ident = np.load(idf)["ident"]; role = np.where(ident == 1, 1, np.where(ident == 0, 2, 0)).astype(np.int8); st = {r: pd.stream(kp, sc, role, r) for r in (1, 2)}; X = []
    for mir in (False, True):
        for r in (1, 2):
            K, C, M = st[r]; Ko, Co, Mo = st[3 - r]
            if mir:
                K, C = pd.mirror(K, C, W); Ko, Co = pd.mirror(Ko, Co, W)
            X.append(pd.feats(K, C, M, Ko, Mo))
    X = np.stack(X); M = np.stack([st[1][2], st[2][2]]); models = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn", "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); models.append(m)
    P = np.mean([pt.predict(m, X[:2]) for m in models], 0); H = np.mean([ct.heat(m, dict(X=X, M=M)) for m in contact_models()], 0)        # (2, F, 7), (2, F, 2)
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); vfps = cap.get(cv2.CAP_PROP_FPS); cap.release(); step = int(fr[1] - fr[0]); punches = []
    for b, r in ((0, 1), (1, 2)):
        K = st[r][0]
        for a, e, c in pt.events(P[b], M[b], THR):
            h = 0 if c % 2 == 1 else 1; w = 9 if h == 0 else 10; a2, e2 = max(a - GROW, 0), min(e + GROW, F - 1); t = a2 + int(H[b, a2:e2 + 1, h].argmax()); off = 0.0
            if 0 < t < F - 1:
                y0, y1, y2 = H[b, t - 1, h], H[b, t, h], H[b, t + 1, h]; den = y0 - 2 * y1 + y2; off = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if abs(den) > 1e-6 else 0.0
            g = K[t, w] + 0.4 * (K[t, w] - K[t, w - 2]); S = max(np.linalg.norm(K[t, [5, 6]].mean(0) - K[t, [11, 12]].mean(0)), 20.0); ok = bool(np.isfinite(g).all() and np.isfinite(S))
            m = (a + e) // 2; km = K[m]; gm = km[w] + 0.4 * (km[w] - km[w - 2]); okm = bool(np.isfinite(gm).all() and np.isfinite(km[[5, 6, 11, 12]]).all())
            Sm = max(np.linalg.norm(km[[5, 6]].mean(0) - km[[11, 12]].mean(0)), 20.0) if okm else 0.0; pf = P[b, a:e + 1, 1:].sum(0)[[h, h + 2, h + 4]]; pf = pf / max(pf.sum(), 1e-9)
            punches.append(dict(boxer="AB"[b], cls=int(c), type=NAMES[c], i0=int(a), i1=int(e), tm=float(fr[m] / vfps), tx=float(gm[0]) if okm else None, ty=float(gm[1]) if okm else None,
                                tside=float(2.4 * Sm), sk=[round(float(v), 4) for v in pf], t0=float(fr[a] / vfps), t1=float(fr[e] / vfps), tc=float((fr[t] + off * step) / vfps),
                                cx=float(g[0]) if ok else None, cy=float(g[1]) if ok else None, side=float(2.4 * S) if ok else None, conf=float((1 - P[b, a:e + 1, 0]).mean())))
    punches.sort(key=lambda p: p["tc"]); return punches

def crop(im, cx, cy, side):
    h, w = im.shape[:2]; s = int(max(side, 64)); x1, y1 = int(cx - s / 2), int(cy - s / 2); out = np.zeros((s, s, 3), np.uint8)
    xa, ya, xb, yb = max(x1, 0), max(y1, 0), min(x1 + s, w), min(y1 + s, h)
    if xb > xa and yb > ya:
        out[ya - y1:yb - y1, xa - x1:xb - x1] = im[ya:yb, xa:xb]
    return cv2.cvtColor(cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

def stage_scan(cid, punches, tmpdir):
    idx = [n for n, p in enumerate(punches) if p["cx"] is not None]; N = len(idx); S = np.full((len(punches), len(OFFS), 4), np.nan, np.float32)
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); vfps = cap.get(cv2.CAP_PROP_FPS); nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tmp = os.path.join(tmpdir, cid + "_clips.npy"); A = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(N, GN, 224, 224, 3)); need = {}
    tidx = [n for n, p in enumerate(punches) if p["tx"] is not None]; NT = len(tidx); PX = np.full((len(punches), 3), np.nan, np.float32)      # type clips: 16 frames at 30 fps around the middle of the punch
    tmp2 = os.path.join(tmpdir, cid + "_tclips.npy"); T = np.lib.format.open_memmap(tmp2, mode="w+", dtype=np.uint8, shape=(max(NT, 1), 16, 224, 224, 3)); tfilled = np.zeros((NT, 16), bool)
    for k, n in enumerate(idx):
        for j in range(GN):
            need.setdefault(int(np.clip(round((punches[n]["tc"] + (j - G0) / 50) * vfps), 0, nf - 1)), []).append((k, j, 0))
    for k, n in enumerate(tidx):
        for j in range(16):
            need.setdefault(int(np.clip(round((punches[n]["tm"] + (j - 8) / 30) * vfps), 0, nf - 1)), []).append((k, j, 1))
    last = max(need); f = 0; t0 = time.time(); progress.start(f"cut clips {cid}", total=last + 1); filled = np.zeros((N, GN), bool)
    while f <= last:
        if not cap.grab():
            break
        if f in need:
            ok, im = cap.retrieve()
            if ok:
                for k, j, which in need[f]:
                    if which == 0:
                        p = punches[idx[k]]; A[k, j] = crop(im, p["cx"], p["cy"], p["side"]); filled[k, j] = True
                    else:
                        p = punches[tidx[k]]; T[k, j] = crop(im, p["tx"], p["ty"], p["tside"]); tfilled[k, j] = True
        f += 1
        if f % 500 == 0:
            progress.tick(500, note=f"frame {f}/{last}")
    progress.finish(); cap.release(); print(f"{N} clips cut in {time.time() - t0:.0f}s (video {vfps:.2f} fps -> 50 fps grid); incomplete clips {int((~filled).any(1).sum())}", flush=True)
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification
    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16, tubelet_size=2, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
                         use_mean_pooling=True, num_labels=4, attn_implementation="sdpa")
    net = VideoMAEForVideoClassification(cfg); sd = torch.load(os.path.join(ROOT, "runs", "landed_model", "cl_all_large.pt"), map_location="cpu", weights_only=True)
    sd = {k.replace(".q_bias", ".query.bias").replace(".v_bias", ".value.bias"): v for k, v in sd.items()}       # saved with an older transformers (same weights, other names)
    net.load_state_dict(sd, strict=True); dev = pt.DEV; net = net.to(dev).eval(); mean, std = torch.tensor(MEAN, device=dev).view(1, 1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 1, 3, 1, 1)
    progress.start(f"landed scan {cid}", total=N * len(OFFS)); t0 = time.time()
    with torch.no_grad():
        for k0 in range(0, N, 8):
            kk = list(range(k0, min(k0 + 8, N)))
            for j, d in enumerate(OFFS):
                x = torch.from_numpy(np.stack([A[k, G0 - 8 + d:G0 + 8 + d] for k in kk])).to(dev).permute(0, 1, 4, 2, 3).float() / 255
                S[[idx[k] for k in kk], j] = torch.softmax(net(pixel_values=(x - mean) / std).logits.float(), 1).cpu().numpy()
            progress.tick(len(kk) * len(OFFS), note=f"punch {k0}/{N}")
    progress.finish(); print(f"scan of {N} punches x {len(OFFS)} positions: {time.time() - t0:.0f}s on {dev} ({(time.time() - t0) / max(N * len(OFFS), 1) * 1000:.0f} ms per clip)", flush=True)
    del A, net; os.remove(tmp)
    for k in range(NT):                                              # frames past the end of the video: repeat the nearest one
        if not tfilled[k].all() and tfilled[k].any():
            good = np.flatnonzero(tfilled[k])
            for j in np.flatnonzero(~tfilled[k]):
                T[k, j] = T[k, good[np.abs(good - j).argmin()]]
    if NT:
        import type_chain as tc; t0 = time.time(); PX[tidx] = tc.type_probs(tc.type_model(dev), T[:NT], dev, note=f"punch type {cid}")
        print(f"type model on {NT} punches: {time.time() - t0:.0f}s ({(time.time() - t0) / NT * 1000:.0f} ms per punch)", flush=True)
    del T; os.remove(tmp2); return S, PX

def verdicts(punches, S):
    cen = np.nan_to_num(1 - S[:, :, 3], nan=-1); k = cen.argmax(1)
    for n, p in enumerate(punches):
        if np.isnan(S[n, 0, 0]):
            p.update(verdict=None, p=None, t_hit=p["tc"], centred=None); continue
        p3 = S[n, k[n], :3] / S[n, k[n], :3].sum()
        p.update(verdict=VERD[int(p3.argmax())], p=[round(float(v), 3) for v in p3], t_hit=float(p["tc"] + OFFS[k[n]] / 50), centred=round(float(cen[n, k[n]]), 3))
    return punches

def fuse_types(punches, PX):
    """Punch family (straight / hook / uppercut) = mean of the skeleton detector's and the pixel model's probabilities; the hand stays the detector's."""
    for n, p in enumerate(punches):
        p["cls_det"] = p["cls"]
        if np.isnan(PX[n, 0]):
            continue
        f3 = 0.5 * np.array(p["sk"]) + 0.5 * PX[n]; h = (p["cls"] - 1) % 2; p["cls"] = int(1 + h + 2 * int(f3.argmax())); p["type"] = NAMES[p["cls"]]; p["ptype"] = [round(float(v), 3) for v in f3]
    return punches

def summary(cid, punches, secs):
    L = [f"{cid}: {secs / 60:.1f} min, {len(punches)} punches counted ({len(punches) / secs * 60:.0f} per minute)", ""]
    for b in "AB":
        mine = [p for p in punches if p["boxer"] == b]; v = {x: sum(p["verdict"] == x for p in mine) for x in VERD}; soft = sum(p["p"][0] for p in mine if p["p"])
        L.append(f"Boxer {b} ({'orange' if b == 'A' else 'cyan'}): thrown {len(mine)} | landed {v['LANDED']} ({v['LANDED'] / max(len(mine), 1):.0%}), blocked {v['blocked']}, missed {v['missed']}, "
                 f"not judged {sum(p['verdict'] is None for p in mine)} | sum of P(landed) {soft:.0f}")
        L.append("   " + " ".join(f"{'type':>12s}" if i == 0 else f"{n:>11s}" for i, n in enumerate(["type"] + NAMES[1:])))
        for row, sel in (("thrown", lambda p: True), ("landed", lambda p: p["verdict"] == "LANDED")):
            L.append("   " + f"{row:>12s} " + " ".join(f"{sum(p['cls'] == c and sel(p) for p in mine):>11d}" for c in range(1, 7)))
        L.append("")
    return "\n".join(L)

DMG = [0, 3, 5, 6, 7, 6, 8]                                           # damage units per LANDED punch type (L/R straight, L/R hook, L/R uppercut); blocked and missed deal none

def sounds(punches, t_start, dur, path, sr=44100):
    """Synthesised effects, one per punch at its contact time: landed = thump (deeper and louder for heavier punches),
    blocked = dull slap, missed = whoosh. Written as a wav for muxing."""
    import wave
    from scipy.signal import butter, lfilter
    rng = np.random.default_rng(0); a = np.zeros(int((dur + 1) * sr), np.float32)
    def band(x, lo, hi):
        b, c = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band"); return lfilter(b, c, x)
    for p in punches:
        t = p["t_hit"] - t_start
        if not 0 <= t < dur:
            continue
        if p["verdict"] == "LANDED":
            d = DMG[p["cls"]]; n = int(0.28 * sr); x = np.arange(n) / sr; f = (150 - 9 * d) * np.exp(-x * 9) + 45
            s = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-x * 16) + 0.6 * band(rng.standard_normal(n), 300, 2500) * np.exp(-x * 60); s *= 0.45 + 0.06 * d
        elif p["verdict"] == "blocked":
            n = int(0.12 * sr); x = np.arange(n) / sr; s = 0.5 * band(rng.standard_normal(n), 500, 1600) * np.exp(-x * 45) + 0.2 * np.sin(2 * np.pi * 210 * x) * np.exp(-x * 40)
        else:
            n = int(0.24 * sr); x = np.arange(n) / sr; s = 0.3 * band(rng.standard_normal(n), 700, 3500) * np.hanning(n); t -= 0.10
        i = max(int(t * sr), 0); s = s[:len(a) - i]; a[i:i + len(s)] += s.astype(np.float32)
    a = np.clip(a / max(np.abs(a).max(), 1e-6) * 0.9, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((a * 32767).astype(np.int16).tobytes())

def results_stats(A, punches, t_end):
    """Numbers for the results card, all from the real counts up to t_end. Winner: the boxer who scored the first K.O.;
    otherwise more health left (health = landed punches weighted by type); then more landed punches; else a draw."""
    done = [[p for p in punches if p["boxer"] == "AB"[b] and p["t_hit"] <= t_end] for b in (0, 1)]; land = [[p for p in d if p["verdict"] == "LANDED"] for d in done]
    hp = [A.hp(b, t_end) for b in (0, 1)]; kos = [x for x in A.kos if x[1] <= t_end]
    if kos:
        winner, how = 1 - min(kos, key=lambda x: x[1])[0], "BY K.O."
    elif round(hp[0]) != round(hp[1]):
        winner, how = (0 if hp[0] > hp[1] else 1), "ON HEALTH LEFT"
    elif len(land[0]) != len(land[1]):
        winner, how = (0 if len(land[0]) > len(land[1]) else 1), "ON LANDED PUNCHES"
    else:
        winner, how = None, "EVEN ON EVERY COUNT"
    combos = [[r for r in A.combos[b] if r[-1]["t"] <= t_end] for b in (0, 1)]
    def row(label, a, b_, fmt=str, high=True):
        better = None if a == b_ else ((0 if a > b_ else 1) if high else (0 if a < b_ else 1)); return (label, fmt(a), fmt(b_), better)
    acc = [len(land[b]) / max(len(done[b]), 1) for b in (0, 1)]; cnt = lambda b, c: sum(p["cls"] == c for p in land[b])
    rows = [row("LANDED", len(land[0]), len(land[1])), row("THROWN", len(done[0]), len(done[1])), row("ACCURACY", acc[0], acc[1], lambda v: f"{v:.0%}"),
            row("BLOCKED BY OPPONENT", sum(p["verdict"] == "blocked" for p in done[0]), sum(p["verdict"] == "blocked" for p in done[1]), high=False),
            row("MISSED", sum(p["verdict"] == "missed" for p in done[0]), sum(p["verdict"] == "missed" for p in done[1]), high=False),
            row("BIGGEST COMBO", max([len(r) for r in combos[0]] + [0]), max([len(r) for r in combos[1]] + [0])),
            row("BEST COMBO DAMAGE", max([sum(e["dmg"] for e in r) for r in combos[0]] + [0]), max([sum(e["dmg"] for e in r) for r in combos[1]] + [0])),
            row("HEALTH LEFT", round(hp[0]), round(hp[1]))]
    for name, c in (("STRAIGHTS LANDED", 1), ("HOOKS LANDED", 3), ("UPPERCUTS LANDED", 5)):
        rows.append((name + "  (L / R)", f"{cnt(0, c)} / {cnt(0, c + 1)}", f"{cnt(1, c)} / {cnt(1, c + 1)}", None))
    return dict(winner=winner, how=how, rows=rows)

def render(cid, punches, start, seconds, reduce_fx=False, cards=True):
    """Annotated video in the arcade style (pipeline/arcade_fx.py, user's spec): skeletons, then hit-stop, impact effects, camera, HUD.
    cards: intro card before the footage and results card after it (the sound is shifted to match)."""
    import arcade_fx as fx
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); kp, sc, fr = R["kp"], R["score"], R["frame"]; sfps = float(R["fps"])
    ident = np.load(os.path.join(ROOT, "runs", "identity_sparring", cid + ".npz"))["ident"]
    src = os.path.join(ROOT, "external", "raw", cid + ".mp4"); cap = cv2.VideoCapture(src); vfps = cap.get(cv2.CAP_PROP_FPS); W, Hh = int(cap.get(3)), int(cap.get(4)); k = 1280 / W; oh = int(round(Hh * k / 2)) * 2
    i0 = int(start * sfps); i1 = len(fr) if seconds is None else min(len(fr), i0 + int(seconds * sfps)); raw = os.path.join(OUT, cid + "_raw.mp4")
    vw = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), sfps, (1280, oh)); f = 0; progress.start(f"render {cid}", total=i1 - i0); FONT = cv2.FONT_HERSHEY_SIMPLEX
    A = fx.Arcade(punches, 1280, oh, sfps, DMG, cols=(COL[0], COL[1]), reduce=reduce_fx); trail = {}; head = {}; BAR = int(A.geo["y0"] + A.geo["h"] + 150)
    if i0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[i0])); f = int(fr[i0])
    lead, tail = (fx.P["intro_t"], fx.P["outro_t"]) if cards else (0.0, 0.0); t_first = fr[i0] / vfps; last = None
    if cards:                                                                                       # intro card over the first frame
        c2 = cv2.VideoCapture(src); c2.set(cv2.CAP_PROP_POS_FRAMES, int(fr[i0])); ok, base = c2.read(); c2.release(); base = cv2.resize(base, (1280, oh), interpolation=cv2.INTER_AREA)
        for j in range(int(round(lead * sfps))):
            vw.write(A.intro(base, j / sfps, t_first))
    def txt(im, s, x, y, sc_, col, th=2):
        cv2.putText(im, s, (x, y), FONT, sc_, (0, 0, 0), th + 3, cv2.LINE_AA); cv2.putText(im, s, (x, y), FONT, sc_, col, th, cv2.LINE_AA)
    for i in range(i0, i1):
        while f < fr[i]:
            cap.grab(); f += 1
        ok, im = cap.read(); f += 1
        if not ok:
            break
        im = cv2.resize(im, (1280, oh), interpolation=cv2.INTER_AREA); t = fr[i] / vfps
        for p in np.flatnonzero(ident[i] >= 0):
            b = 0 if ident[i, p] == 1 else 1; xy = kp[i, p] * k
            for a, c in EDGES:
                if sc[i, p, a] > .3 and sc[i, p, c] > .3:
                    cv2.line(im, tuple(int(v) for v in xy[a]), tuple(int(v) for v in xy[c]), COL[b], 2, cv2.LINE_AA)
            top = xy[sc[i, p] > .3]
            if len(top):
                head[b] = (int(top[:, 0].mean()), int(top[:, 1].min()) - 14); txt(im, "AB"[b], head[b][0] - 8, head[b][1], 0.9, COL[b])
            for h, (w, e) in enumerate(((9, 7), (10, 8))):                                              # glove points of the last frames, for the afterimages
                if sc[i, p, w] > .3 and sc[i, p, e] > .3:
                    trail.setdefault((b, h), []).append(xy[w] + 0.4 * (xy[w] - xy[e])); trail[(b, h)] = trail[(b, h)][-fx.P['trail_n']:]
        im = A.hitstop(im, t); im = A.trails(im, t, trail); im = A.effects(im, t, k, head, trail)
        for e in A.ev:                                                                                  # what the system decided, next to the puncher
            dt = t - e["t"]
            if 0 <= dt <= 1.2:
                b = e["b"]; v = VERD.index(e["v"]) if e["v"] else 2; hx, hy = head.get(b, (200 + 640 * b, 220)); y = max(hy - 34 - int(dt * 25), BAR)
                txt(im, f"{e['type']}: {e['v'] or 'not judged'}" + (f" {max(e['p']):.0%}" if e["p"] else ""), int(np.clip(hx - 260 if b == 0 else hx + 20, 5, 940)), y + (0 if b == 0 else 24), 0.62, VCOL[v])
        im = A.camera(im, t, k); im = A.focus(im, t, k); A.pip_capture(im, t); im = A.ko_scene(im, t, k); im = A.vignette(im, t); stats = []
        for b in (0, 1):
            done = [p for p in punches if p["boxer"] == "AB"[b] and p["t_hit"] <= t]; land = [p for p in done if p["verdict"] == "LANDED"]; n = [sum(p["cls"] == c for p in land) for c in range(1, 7)]
            stats.append([f"LANDED {len(land)}   thrown {len(done)}   blocked {sum(p['verdict'] == 'blocked' for p in done)}   missed {sum(p['verdict'] == 'missed' for p in done)}",
                          f"landed:  straight L{n[0]} R{n[1]}   hook L{n[2]} R{n[3]}   uppercut L{n[4]} R{n[5]}"])
        im = A.hud(im, t, stats); im = A.callouts(im, t, k); im = A.pip_draw(im, t); im = A.ko_text(im, t); vw.write(im); last = (im, t)
        if (i - i0) % 100 == 0:
            progress.tick(100, note=f"frame {i - i0}/{i1 - i0}")
    t_start = fr[i0] / vfps; dur = (i1 - i0) / sfps
    if cards and last is not None:                                                                  # results card over the last frame
        R = results_stats(A, punches, last[1])
        for j in range(int(round(tail * sfps))):
            vw.write(A.results(last[0], j / sfps, R))
    vw.release(); cap.release(); progress.finish(); dst = os.path.join(OUT, cid + ".mp4"); wav = os.path.join(OUT, cid + "_fx.wav")
    cues = ([(fx.P["fight_at"], "bell"), (lead + dur, "bell3")] if cards else []) + [(lead + tko - t_start, kind) for _, tko, _ in A.kos if 0 <= tko - t_start < dur for kind in ("boom", "bell3")]
    fx.sounds(A.ev, t_start, dur, wav, lead=lead, tail=tail, cues=cues)
    has_audio = "audio" in subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", src], capture_output=True, text=True).stdout
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", raw, "-i", wav]
    if has_audio:                                                                                   # keep the original sound underneath, quieter
        ms = int(round(lead * 1000)); dl = f",adelay={ms}|{ms}" if ms else ""
        cmd += ["-ss", f"{t_start:.3f}", "-t", f"{dur:.3f}", "-i", src, "-filter_complex", f"[2:a]volume=0.5{dl}[o];[1:a][o]amix=inputs=2:duration=first:normalize=0[a]", "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    subprocess.run(cmd + ["-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", dst], check=True); os.remove(raw); os.remove(wav); return dst

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("--no-render", action="store_true"); ap.add_argument("--start", type=float, default=0); ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--reduce-fx", action="store_true", help="accessibility: screen shake and flashes reduced by 75%")
    ap.add_argument("--no-cards", action="store_true", help="no intro card before / results card after the footage")
    ap.add_argument("--bob", "--bag", dest="bob", action="store_true", help="one boxer hitting a BOB dummy or a bag: use the bag route (chosen by hand, no automatic routing)")
    ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp")); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True); cid = a.id
    if a.bob:
        sys.path.insert(0, os.path.join(ROOT, "bagroute")); import count_bag; count_bag.run(cid, a.no_render); return
    pj, sn = os.path.join(OUT, cid + "_punches.json"), os.path.join(OUT, cid + "_scan.npz")
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); secs = len(R["frame"]) / float(R["fps"])
    if os.path.exists(pj) and os.path.exists(sn):
        punches = json.load(open(pj))
    else:
        punches = stage_punches(cid); print(f"{len(punches)} punches counted; cutting clips and scanning", flush=True)
        S, PX = stage_scan(cid, punches, a.tmp); np.savez_compressed(sn, scan=S, offsets=np.array(OFFS), type=PX); punches = fuse_types(verdicts(punches, S), PX); json.dump(punches, open(pj, "w"), indent=0)
    s = summary(cid, punches, secs); open(os.path.join(OUT, cid + "_summary.txt"), "w").write(s + "\n"); print(s)
    if not a.no_render:
        print("video:", render(cid, punches, a.start, a.seconds, a.reduce_fx, cards=not a.no_cards))

if __name__ == "__main__":
    main()
