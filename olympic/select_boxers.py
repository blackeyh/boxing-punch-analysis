"""Boxer selector: which 2 of the people RTMPose finds are the boxers (not the referee, not the crowd).

1. TRACKLETS: link people frame to frame by keypoint distance (Hungarian), so each person has a short history.
2. FEATURES per person and frame, from a +-1 s window of that person's tracklet. Kit-independent (pose and
   motion only, no colour), so it can transfer to footage without red/blue kits:
     guard_lo / guard_hi : height of the lower / higher wrist between hip (0) and shoulder (1)
     mobility            : how much the hips move (torso lengths)
     arm_speed           : wrist speed relative to the hips (torso lengths / frame)
     leg_extent          : hip-to-ankle height / torso (standing ~2, seated ~1)
     kp_score            : mean keypoint confidence
     size                : torso length relative to the largest torso in the frame
     near                : distance to the nearest similar-sized person (torso lengths)
     lifetime            : tracklet length (s, capped)
3. LABELS from the referee, free: at a labelled CONTACT punch (head/body/block), the person with a wrist in
   the box and a second person with head/torso in the box are boxers; everyone else in that frame is not.
4. MODEL: logistic regression -> P(boxer) per person-frame, smoothed along the tracklet; per frame keep the
   top 2 above THR.
    python olympic/select_boxers.py fit            # leave-one-venue-out test + final model -> olympic/boxer_selector.json
    python olympic/select_boxers.py apply ID...    # -> external/boxers_rtm/{id}.npz
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
WIN, GATE, MAXGAP, THR, MARGIN = 25, 0.35, 12, 0.5, 30
THR_LIFE, THR_LOW = 0.6, 0.25
NAMES = ["guard_lo", "guard_hi", "mobility", "arm_speed", "leg_extent", "kp_score", "size", "near", "lifetime"]
WR, HEADTORSO = [9, 10], [0, 1, 2, 3, 4, 5, 6, 11, 12]


def tracklets(kp, sc, n):
    """-> tid (F,K) track id per detection (-1 = none)."""
    F, K = n.shape[0], kp.shape[1]
    tid = np.full((F, K), -1, np.int32); nxt = 0
    live = {}                                                 # id -> (kp, score, frame)
    for f in range(F):
        m = int(n[f])
        ids = list(live)
        if m and ids:
            C = np.full((len(ids), m), 1e3)
            for a, i in enumerate(ids):
                pk, ps, _ = live[i]
                for b in range(m):
                    ok = (ps > 0.3) & (sc[f, b] > 0.3)
                    if ok.sum() >= 4:
                        scale = max(np.ptp(kp[f, b][ok], 0).max(), 20.0)
                        C[a, b] = np.linalg.norm(pk[ok] - kp[f, b][ok], axis=1).mean() / scale
            r, c = linear_sum_assignment(C)
            for a, b in zip(r, c):
                if C[a, b] < GATE:
                    tid[f, b] = ids[a]
        for b in range(m):
            if tid[f, b] < 0:
                tid[f, b] = nxt; nxt += 1
            live[tid[f, b]] = (kp[f, b], sc[f, b], f)
        live = {i: v for i, v in live.items() if f - v[2] <= MAXGAP}
    return tid


def features(kp, sc, n, tid):
    """-> X (F,K,len(NAMES)), NaN where there is no person."""
    F, K = tid.shape
    sh = (kp[:, :, 5] + kp[:, :, 6]) / 2; hip = (kp[:, :, 11] + kp[:, :, 12]) / 2
    torso = np.maximum(np.linalg.norm(sh - hip, axis=-1), 8.0)
    h = (hip[..., 1:2] - kp[:, :, WR, 1]) / torso[..., None]                 # wrist height: 0 hip, 1 shoulder
    ank = np.nanmax(kp[:, :, [15, 16], 1], axis=-1)
    raw = dict(guard_lo=h.min(-1), guard_hi=h.max(-1), leg=(ank - hip[..., 1]) / torso, score=sc.mean(-1),
               size=torso / np.nanmax(np.where(tid >= 0, torso, np.nan), axis=1, keepdims=True))
    near = np.full((F, K), np.nan)
    for f in range(F):
        m = int(n[f])
        if m >= 2:
            d = np.linalg.norm(hip[f, :m, None] - hip[f, None, :m], axis=-1) / torso[f, :m, None]
            ratio = torso[f, :m, None] / torso[f, None, :m]
            d[(ratio < 0.6) | (ratio > 1.67) | np.eye(m, dtype=bool)] = np.inf
            near[f, :m] = np.minimum(d.min(1), 20.0)
        elif m == 1:
            near[f, 0] = 20.0
    raw["near"] = near
    X = np.full((F, K, len(NAMES)), np.nan, np.float32)
    fs, ks = np.nonzero(tid >= 0); t = tid[fs, ks]
    for i in np.unique(t):
        sel = t == i; f, k = fs[sel], ks[sel]                                 # frames ascending
        hp, to, wr = hip[f, k], torso[f, k], kp[f, k][:, WR] - hip[f, k][:, None]
        dt = np.maximum(np.diff(f), 1)[:, None]
        hv = np.r_[np.zeros((1, 2)), np.diff(hp, axis=0) / dt] / to[:, None]
        wv = np.r_[np.zeros((1, 2, 2)), np.diff(wr, axis=0) / dt[:, None]] / to[:, None, None]
        aspd = np.linalg.norm(wv, axis=-1).max(-1)
        for j in range(len(f)):
            a, b = np.searchsorted(f, f[j] - WIN), np.searchsorted(f, f[j] + WIN, "right")
            w = slice(a, b)
            X[f[j], k[j]] = [raw["guard_lo"][f[w], k[w]].mean(), raw["guard_hi"][f[w], k[w]].mean(),
                             np.linalg.norm(hp[w].std(0)) / to[w].mean(), aspd[w].mean(),
                             raw["leg"][f[w], k[w]].mean(), raw["score"][f[w], k[w]].mean(),
                             raw["size"][f[w], k[w]].mean(), np.median(raw["near"][f[w], k[w]]),
                             min((f[-1] - f[0]) / 25.0, 20.0)]
    return np.nan_to_num(X, nan=0.0, posinf=20.0, neginf=-20.0) + np.where(tid[..., None] >= 0, 0, np.nan)


def load(cid):
    d = np.load(ROOT / "external" / "pose_rtm" / f"{cid}.npz")
    kp, sc, n, fr = d["kp"].astype(np.float64), d["score"], d["n"].astype(int), d["frame"]
    cache = ROOT / "external" / "pose_rtm" / f"{cid}.feat.npz"
    if cache.exists():
        c = np.load(cache); tid, X = c["tid"], c["X"]
    else:
        tid = tracklets(kp, sc, n); X = features(kp, sc, n, tid)
        np.savez_compressed(cache, tid=tid, X=X)
    return dict(kp=kp, sc=sc, n=n, frame=fr, tid=tid, X=X)


def inside(pts, box, m=MARGIN):
    return (pts[..., 0] >= box[0] - m) & (pts[..., 0] <= box[2] + m) & (pts[..., 1] >= box[1] - m) & (pts[..., 1] <= box[3] + m)


def event_box(e):
    b = np.array(e["boxes"]); return [b[:, 1].min(), b[:, 2].min(), b[:, 3].max(), b[:, 4].max()]


def weak_labels(D, events):
    """Person-frames labelled boxer (1) / not (0) from contact punches."""
    rows = []
    for e in events:
        if e["outcome"] == "miss":
            continue
        i = int(np.clip(np.searchsorted(D["frame"], (e["start"] + e["end"]) // 2), 0, len(D["frame"]) - 1))
        m = D["n"][i]; box = event_box(e)
        if m < 2:
            continue
        ok = D["sc"][i, :m] > 0.3
        w = (inside(D["kp"][i, :m][:, WR], box) & ok[:, WR]).any(1)
        b = (inside(D["kp"][i, :m][:, HEADTORSO], box) & ok[:, HEADTORSO]).any(1)
        touching = (inside(D["kp"][i, :m], box, 60) & ok).any(1)
        if w.sum() < 1 or (w | b).sum() != 2:                 # need exactly two people at the box, one with a wrist in it
            continue
        for p in range(m):
            if w[p] or b[p]:
                rows.append((i, p, 1))
            elif not touching[p]:
                rows.append((i, p, 0))
    return rows


def fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit((X - mu) / sd, y)
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=lr.coef_[0].tolist(), b=float(lr.intercept_[0]), names=NAMES)


def prob(model, X):
    z = ((X - np.array(model["mu"])) / np.array(model["sd"])) @ np.array(model["w"]) + model["b"]
    return 1 / (1 + np.exp(-z))


def select(D, model):
    """-> pick (F,2) person index per frame (-1 = none), P (F,K) smoothed boxer probability."""
    F, K = D["tid"].shape
    P = np.where(D["tid"] >= 0, prob(model, np.nan_to_num(D["X"])), np.nan)
    fs, ks = np.nonzero(D["tid"] >= 0); t = D["tid"][fs, ks]; S = P.copy()
    for i in np.unique(t):                                    # smooth along the tracklet (+-1 s)
        sel = t == i; f, k = fs[sel], ks[sel]; p = P[f, k]
        for j in range(len(f)):
            a, b = np.searchsorted(f, f[j] - WIN), np.searchsorted(f, f[j] + WIN, "right")
            S[f[j], k[j]] = p[a:b].mean()
    life = S.copy()                                           # whole-tracklet mean: a person who is a boxer overall
    for i in np.unique(t):                                    # stays selected through short dips (hysteresis)
        sel = t == i; life[fs[sel], ks[sel]] = P[fs[sel], ks[sel]].mean()
    keep = (S >= THR) | ((life >= THR_LIFE) & (S >= THR_LOW))
    pick = np.full((F, 2), -1, np.int32)
    for f in range(F):
        order = [k for k in np.argsort(-np.nan_to_num(S[f], nan=-1)) if keep[f, k]][:2]
        pick[f, :len(order)] = order
    return pick, S


def punch_test(D, pick, events, shift=0):
    """Share of labelled punches where a SELECTED person has a wrist in the box / both selected are at the box."""
    last = int(D["frame"][-1]); a = b = na = nb = 0
    for e in events:
        s = (e["start"] + shift) % (last - 30); box = event_box(e)
        i0 = max(np.searchsorted(D["frame"], s) - 1, 0); i1 = min(np.searchsorted(D["frame"], s + e["end"] - e["start"], "right") + 1, len(D["frame"]))
        pu = bo = False
        for i in range(i0, i1):
            ps = [p for p in pick[i] if p >= 0]
            w = [bool((inside(D["kp"][i, p][WR], box) & (D["sc"][i, p][WR] > 0.3)).any()) for p in ps]
            h = [bool((inside(D["kp"][i, p][HEADTORSO], box) & (D["sc"][i, p][HEADTORSO] > 0.3)).any()) for p in ps]
            pu |= any(w); bo |= len(ps) == 2 and ((w[0] and h[1]) or (w[1] and h[0]))
        a += pu; na += 1
        if e["outcome"] != "miss":
            b += bo; nb += 1
    return a / max(na, 1), b / max(nb, 1), na


def white_shirt_rate(cid, D, pick, n=40, seed=0):
    """Direct referee check, independent of the selector's own cues: on n random frames, the share of SELECTED
    people whose torso is white (low saturation, bright) = the referee's shirt. Boxers wear red/blue/orange vests."""
    import cv2
    cap = cv2.VideoCapture(str(ROOT / "external" / "raw" / f"{cid}.mp4")); rng = np.random.default_rng(seed)
    white = tot = 0
    for i in sorted(rng.choice(len(D["frame"]), n, replace=False)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(D["frame"][i])); ok, im = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        for p in pick[i]:
            if p < 0:
                continue
            q = D["kp"][i, p][[5, 6, 12, 11]]; c = q.mean(0); q = (c + 0.6 * (q - c)).astype(np.int32)   # shrunk torso quad
            mask = np.zeros(hsv.shape[:2], np.uint8); cv2.fillConvexPoly(mask, q, 1)
            px = hsv[mask > 0]
            if len(px) < 30:
                continue
            tot += 1; white += np.mean((px[:, 1] < 60) & (px[:, 2] > 140)) > 0.5
    return white, tot


def main():
    ev = json.load(open(ROOT / "olympic" / "events.json"))
    if sys.argv[1] == "fit":
        ids = sorted(p.name[:-4] for p in (ROOT / "external" / "pose_rtm").glob("O_*.npz") if p.name.count(".") == 1)
        data = {}
        for cid in ids:
            D = load(cid); es = [e for e in ev if "O_" + e["video"] == cid and e["end"] <= D["frame"][-1]]
            rows = weak_labels(D, es)
            data[cid] = (D, es, np.array([D["X"][i, p] for i, p, _ in rows]), np.array([y for *_, y in rows]))
            print(f"{cid}: {len(es)} punches, weak labels boxer {int(data[cid][3].sum())} / other {int((1 - data[cid][3]).sum())}, tracklets {D['tid'].max() + 1}")
        report = {}; agg = {}
        for held in ("kam2", "kam4"):
            tr = [c for c in ids if held not in c]; te = [c for c in ids if held in c]
            model = fit_lr(np.vstack([data[c][2] for c in tr]), np.concatenate([data[c][3] for c in tr]))
            Xt, yt = np.vstack([data[c][2] for c in te]), np.concatenate([data[c][3] for c in te])
            p = prob(model, Xt); acc = ((p >= 0.5) == yt).mean(); rec = (p[yt == 1] >= 0.5).mean(); spec = (p[yt == 0] < 0.5).mean()
            print(f"\nheld-out venue {held} (trained on the other): person-frame accuracy {acc:.1%} | boxers kept {rec:.1%} | others rejected {spec:.1%}")
            print("  weights: " + ", ".join(f"{n} {w:+.2f}" for n, w in zip(NAMES, model["w"])))
            for c in te:
                D, es = data[c][:2]; pick, S = select(D, model)
                real = punch_test(D, pick, es); ch = np.mean([punch_test(D, pick, es, s)[:2] for s in (350, 700, 1050)], axis=0)
                two = (pick >= 0).sum(1)
                print(f"  {c}: puncher among the selected {real[0]:.0%} (shifted labels {ch[0]:.0%}) | both at the box {real[1]:.0%} (shifted {ch[1]:.0%}) | frames with 2 selected {np.mean(two == 2):.0%}, 1: {np.mean(two == 1):.0%}, 0: {np.mean(two == 0):.0%}")
                wh, wt = white_shirt_rate(c, D, pick)
                print(f"      white-shirted (referee) among the selected, 40 random frames: {wh}/{wt}")
                report[c] = dict(puncher=real[0], both=real[1], chance_puncher=float(ch[0]), chance_both=float(ch[1]), two=float(np.mean(two == 2)),
                                 white=int(wh), white_n=int(wt))
                contact = sum(e["outcome"] != "miss" for e in es)
                a = agg.setdefault(held, np.zeros(8)); a += [real[0] * len(es), len(es), real[1] * contact, contact, wh, wt, np.mean(two == 2) * len(D["frame"]), len(D["frame"])]
        for held, a in agg.items():
            print(f"\nSUMMARY held-out {held}: puncher among the selected {a[0] / a[1]:.1%} of {int(a[1])} punches | both at the box {a[2] / a[3]:.1%} of {int(a[3])} contact punches | "
                  f"referee among the selected {a[4] / max(a[5], 1):.1%} ({int(a[4])}/{int(a[5])}) | frames with 2 selected {a[6] / a[7]:.1%}")
        final = fit_lr(np.vstack([d[2] for d in data.values()]), np.concatenate([d[3] for d in data.values()]))
        final["leave_one_venue_out"] = report
        json.dump(final, open(ROOT / "olympic" / "boxer_selector.json", "w"), indent=1)
    else:
        model = json.load(open(ROOT / "olympic" / "boxer_selector.json"))
        out = ROOT / "external" / "boxers_rtm"; out.mkdir(exist_ok=True)
        for cid in sys.argv[2:]:
            D = load(cid); pick, S = select(D, model)
            np.savez_compressed(out / f"{cid}.npz", frame=D["frame"], pick=pick, prob=S, tid=D["tid"])
            print(cid, "frames with 2 selected", f"{np.mean((pick >= 0).sum(1) == 2):.0%}")


if __name__ == "__main__":
    main()
