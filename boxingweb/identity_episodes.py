"""Which boxer is which, skeleton-only: LEARNED re-association at each separation.

A round = stretches where the two boxers are clearly APART (boxes barely overlap: following each by motion is
reliable) separated by OVERLAP episodes (clinch, exchange, a boxer briefly undetected), where RTMPose's skeletons
cannot be trusted. Identity only has to be decided once per episode: did the two boxers come out the same way round
("keep") or swapped? Cues, all from skeletons (each = cost of "swap" minus cost of "keep"):
  continuity vote : follow the skeletons through the episode frame by frame (the earlier baseline) + its margin
  position        : extrapolated / held position before vs after
  apparent size   : body scale before vs after (who is nearer the camera)
  feet height     : ankle height in the image (depth on the floor)
  body signature  : limb-length ratios
  stance          : which foot is nearer the opponent
  order           : left/right order kept? front/back (feet) order kept?   + episode length
Truth: BoxingWeb red/blue roles. Train on data_train, test on data_test. The two boxers are given (truth roles), so
only identity is tested. Camera cuts (both boxers jump) are reported separately.
    python boxingweb/identity_episodes.py
"""
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
IOU_APART, HOLE, MINLEN, W = 0.15, 3, 8, 12
NAMES = ["cont_vote", "cont_margin", "pos_extrap", "pos_hold", "size", "feet", "signature", "stance", "side_kept", "depth_kept", "gap_s", "min_iou_gap"]

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)

def pdist(pa, sa, pb, sb_):
    ok = (sa > 0.3) & (sb_ > 0.3)
    if ok.sum() < 3:
        return 9.0
    return np.linalg.norm(pa[ok] - pb[ok], axis=1).mean() / max(np.ptp(pb[ok], 0).max(), 20.0)

def segments(R, role, cut=None):
    """-> list of segments: dict(frames [i], tr [[p0...],[p1...]], cut_before bool). `cut` (F,) bool: camera cuts
    found in the pixels (boxingweb/detect_cuts.py); a stretch never spans one."""
    kp, sc, box = R["kp"].astype(np.float64), R["score"], R["box"]; F = len(R["frame"])
    segs = []; cur = None; hole = 0; pix_cut = False
    for i in range(F):
        if cut is not None and cut[i]:
            if cur:
                segs.append(cur); cur = None
            pix_cut = True
        ps = np.flatnonzero(role[i] > 0)
        ap = len(ps) == 2 and iou(box[i, ps[0]], box[i, ps[1]]) < IOU_APART
        if not ap:
            hole += 1
            if cur and hole > HOLE:
                segs.append(cur); cur = None
            continue
        a, b = ps
        if cur is None:
            cur = dict(frames=[i], tr=[[a], [b]], cut_before=pix_cut); pix_cut = False; hole = 0; continue
        j = cur["frames"][-1]; q0, q1 = cur["tr"][0][-1], cur["tr"][1][-1]
        keep = pdist(kp[j, q0], sc[j, q0], kp[i, a], sc[i, a]) + pdist(kp[j, q1], sc[j, q1], kp[i, b], sc[i, b])
        swap = pdist(kp[j, q0], sc[j, q0], kp[i, b], sc[i, b]) + pdist(kp[j, q1], sc[j, q1], kp[i, a], sc[i, a])
        if min(keep, swap) > 1.6:                                  # both jump: camera cut
            segs.append(cur); cur = dict(frames=[i], tr=[[a], [b]], cut_before=True); hole = 0; continue
        if swap < keep:
            a, b = b, a
        cur["frames"].append(i); cur["tr"][0].append(a); cur["tr"][1].append(b); hole = 0
    if cur:
        segs.append(cur)
    out = []; pending_cut = False
    for s in segs:                                                 # drop very short stretches, carry the cut flag forward
        pending_cut |= s["cut_before"]
        if len(s["frames"]) >= MINLEN:
            s["cut_before"] = pending_cut; pending_cut = False; out.append(s)
    return out

def summary(R, fr, ps, other, end):
    """Skeleton summary of one boxer over the last (end=True) or first W frames of a stretch."""
    kp, sc = R["kp"].astype(np.float64), R["score"]
    sl = slice(-W, None) if end else slice(0, W); f = np.array(fr[sl]); p = np.array(ps[sl]); o = np.array(other[sl])
    k = kp[f, p]; s = sc[f, p]; hip = k[:, [11, 12]].mean(1); sh = k[:, [5, 6]].mean(1)
    L = np.stack([np.linalg.norm(k[:, a] - k[:, b], axis=-1) for a, b in LIMBS], -1)
    scale = np.median(np.maximum(np.linalg.norm(sh - hip, axis=1), 8.0))
    sig = np.median(L / np.maximum(L.sum(1, keepdims=True), 1e-6), 0)
    ox = kp[f, o][:, [11, 12], 0].mean(1)                                                    # opponent's hip x
    lead_left = np.mean(np.abs(k[:, 15, 0] - ox) < np.abs(k[:, 16, 0] - ox))                 # left ankle nearer the opponent
    edge = slice(-3, None) if end else slice(0, 3); c = hip[edge].mean(0)
    n = len(f); v = (hip[-3:].mean(0) - hip[:3].mean(0)) / max(f[-1] - f[0], 1) if n >= 6 else np.zeros(2)
    return dict(c=c, v=v, scale=scale, feet=np.median(k[:, [15, 16], 1].max(1)), sig=sig, stance=lead_left, i=f[-1] if end else f[0],
                pose=k[-1] if end else k[0], psc=s[-1] if end else s[0])

def features(R, role, A, B):
    kp, sc, box = R["kp"].astype(np.float64), R["score"], R["box"]
    a = [summary(R, A["frames"], A["tr"][t], A["tr"][1 - t], True) for t in (0, 1)]
    b = [summary(R, B["frames"], B["tr"][t], B["tr"][1 - t], False) for t in (0, 1)]
    gap = b[0]["i"] - a[0]["i"]; S = (a[0]["scale"] + a[1]["scale"]) / 2
    def diff(fn):                                                 # cost(swap) - cost(keep): positive supports "keep"
        return (fn(a[0], b[1]) + fn(a[1], b[0])) - (fn(a[0], b[0]) + fn(a[1], b[1]))
    g = min(gap, 15)
    # continuity through the episode on whatever boxer skeletons exist
    st = [dict(pose=a[t]["pose"].copy(), psc=a[t]["psc"].copy()) for t in (0, 1)]; miou = 0.0
    for i in range(a[0]["i"] + 1, b[0]["i"] + 1):
        ps = list(np.flatnonzero(role[i] > 0))
        if len(ps) == 2:
            miou = max(miou, iou(box[i, ps[0]], box[i, ps[1]]))
            C = [[pdist(st[t]["pose"], st[t]["psc"], kp[i, p], sc[i, p]) for p in ps] for t in (0, 1)]
            asg = {0: ps[0], 1: ps[1]} if C[0][0] + C[1][1] <= C[0][1] + C[1][0] else {0: ps[1], 1: ps[0]}
            margin = abs((C[0][1] + C[1][0]) - (C[0][0] + C[1][1]))
        elif len(ps) == 1:
            c0, c1 = (pdist(st[t]["pose"], st[t]["psc"], kp[i, ps[0]], sc[i, ps[0]]) for t in (0, 1)); asg = {0 if c0 <= c1 else 1: ps[0]}; miou = 1.0
        else:
            miou = 1.0; continue
        for t, p in asg.items():
            st[t] = dict(pose=kp[i, p].copy(), psc=sc[i, p].copy())
    i1 = B["frames"][0]; ps = [B["tr"][0][0], B["tr"][1][0]]
    C = [[pdist(st[t]["pose"], st[t]["psc"], kp[i1, p], sc[i1, p]) for p in ps] for t in (0, 1)]
    cont = (C[0][1] + C[1][0]) - (C[0][0] + C[1][1])
    x = [np.sign(cont), cont,
         diff(lambda p, q: np.linalg.norm(p["c"] + p["v"] * g - q["c"]) / S), diff(lambda p, q: np.linalg.norm(p["c"] - q["c"]) / S),
         diff(lambda p, q: abs(np.log(p["scale"] / q["scale"]))), diff(lambda p, q: abs(p["feet"] - q["feet"]) / S),
         diff(lambda p, q: np.abs(p["sig"] - q["sig"]).sum()), diff(lambda p, q: abs(p["stance"] - q["stance"])),
         float(np.sign(a[0]["c"][0] - a[1]["c"][0]) == np.sign(b[0]["c"][0] - b[1]["c"][0])) * 2 - 1,
         float(np.sign(a[0]["feet"] - a[1]["feet"]) == np.sign(b[0]["feet"] - b[1]["feet"])) * 2 - 1,
         gap / 30.0, miou]
    return np.array(x, np.float32)

def load_round(c):
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", c + ".npz")); role = np.load(os.path.join(ROOT, "external", "boxers_gt", c + ".npz"))["role"]
    cp = os.path.join(ROOT, "external", "cuts", c + ".npz")
    cut = np.load(cp)["cut"][R["frame"]] if os.path.exists(cp) else None
    segs = segments(R, role, cut); rows = []
    for s in segs:
        r0 = role[s["frames"], s["tr"][0]]; s["r0"] = 1 if (r0 == 1).mean() >= 0.5 else 2; s["purity"] = float((r0 == s["r0"]).mean())
    for A, B in zip(segs[:-1], segs[1:]):
        rows.append(dict(x=features(R, role, A, B), y=int(A["r0"] == B["r0"]), cut=B["cut_before"], gap=B["frames"][0] - A["frames"][-1]))
    return dict(id=c, segs=segs, rows=rows, sec=len(R["frame"]) / float(R["fps"]))

def chain(rd, decide):
    """Frame-weighted identity correctness over the apart stretches; a wrong decision flips everything after it.
    Identity is re-set from the truth at camera cuts (one continuous camera would not have them)."""
    ok = tot = wrong = 0; state = True
    n = len(rd["segs"][0]["frames"]) if rd["segs"] else 0; ok += n; tot += n
    for row, s in zip(rd["rows"], rd["segs"][1:]):
        if row["cut"]:
            state = True
        elif decide(row) != row["y"]:
            state = not state; wrong += 1
        n = len(s["frames"]); tot += n; ok += n * state
    return ok, tot, wrong

def main():
    from multiprocessing import Pool
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ids = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", pre + "*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as pool:
        tr = pool.map(load_round, ids("W_train")); te = pool.map(load_round, ids("W_test"))
    def XY(rds):
        r = [row for rd in rds for row in rd["rows"] if not row["cut"]]; return np.array([q["x"] for q in r]), np.array([q["y"] for q in r])
    Xtr, ytr = XY(tr); Xte, yte = XY(te)
    segs = [s for rd in te for s in rd["segs"]]; fr = sum(len(s["frames"]) for s in segs); mins = sum(rd["sec"] for rd in te) / 60
    print(f"TEST: {len(te)} rounds, {mins:.0f} min | apart stretches {len(segs)} covering {fr / sum(rd['sec'] * 30 for rd in te):.0%} of frames, purity (one boxer per track) {np.average([s['purity'] for s in segs], weights=[len(s['frames']) for s in segs]):.2%}")
    print(f"      episodes to decide {len(yte)} ({len(yte) / mins:.1f}/min), truth says 'swapped' in {1 - yte.mean():.1%}; camera cuts {sum(row['cut'] for rd in te for row in rd['rows'])}")
    print(f"TRAIN: {len(tr)} rounds, {len(ytr)} episodes")
    gbm = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(np.clip(Xtr, -10, 10), ytr)
    methods = {"always 'keep'": lambda r: 1, "left/right order kept": lambda r: int(r["x"][8] > 0), "held position": lambda r: int(r["x"][3] > 0),
               "continuity through the overlap (baseline)": lambda r: int(r["x"][0] >= 0),
               "learned (logistic)": lambda r: int(lr.predict(np.clip(r["x"], -10, 10)[None])[0]), "learned (boosted trees)": lambda r: int(gbm.predict(r["x"][None])[0])}
    res = {}
    for name, fn in methods.items():
        acc = np.mean([fn(row) == row["y"] for rd in te for row in rd["rows"] if not row["cut"]])
        c = np.array([chain(rd, fn) for rd in te]).sum(0)
        print(f"  {name:44s} episode decisions right {acc:.1%} | wrong decisions {c[2] / mins:.2f}/min | right identity in {c[0] / c[1]:.1%} of apart frames")
        res[name] = dict(acc=float(acc), wrong_per_min=float(c[2] / mins), frames_right=float(c[0] / c[1]))
    lw = lr[-1].coef_[0]; print("  logistic weights: " + ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(NAMES, lw), key=lambda t: -abs(t[1]))))
    for i, n in enumerate(NAMES[:8]):
        print(f"    cue alone '{n}': right {np.mean((Xte[:, i] > 0) == yte):.1%}")
    json.dump(res, open(os.path.join(ROOT, "runs", "identity_episodes.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
