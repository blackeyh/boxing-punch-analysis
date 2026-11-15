"""How often is the identity SWAPPED, including inside overlaps? And does carrying from both sides fix it?

Identity in a clip = recognition in the "apart" stretches + CARRYING by motion through the overlaps between them.
  forward carry (so far): follow the skeletons from the stretch BEFORE the overlap. A mistake stays until the next stretch.
  two-sided carry (fix) : also follow them BACKWARDS from the stretch AFTER the overlap. Where the two agree, done.
                          Where they disagree, a swap happened somewhere inside: each frame takes the label carried
                          from the nearer stretch in time.
Measured on BoxingWeb data_test (truth for every frame):   python boxingweb/identity_swaps.py bw
Estimated on a clip without truth (A1, A2):                python boxingweb/identity_swaps.py clip ID
  proxy = HEAD colour (headgear): a per-clip A-vs-B classifier on the head patch, trained on the stretch frames and
  checked on held-out stretches; then applied to the overlap frames. It uses different evidence from the whole-body
  recognition model, so a disagreement inside an overlap marks a carried swap.
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "olympic"))
import mem_guard  # noqa: F401
import cv2, numpy as np
import identity_episodes as ie

def carry(kp, sc, pick, lab, in_seg, cut, reverse=False):
    F = len(pick); c = lab.copy(); st = None
    for i in (range(F - 1, -1, -1) if reverse else range(F)):
        if cut is not None and cut[i] and not reverse:
            st = None
        ps = [p for p in pick[i] if p >= 0]
        if in_seg[i]:
            st = {int(lab[i, p]): (kp[i, p].copy(), sc[i, p].copy()) for p in ps if lab[i, p] >= 0}
        elif st is not None and len(st) == 2 and ps:
            C = {(a, p): ie.pdist(st[a][0], st[a][1], kp[i, p], sc[i, p]) for a in (0, 1) for p in ps}
            asg = ({1: ps[0], 0: ps[1]} if C[(1, ps[0])] + C[(0, ps[1])] <= C[(1, ps[1])] + C[(0, ps[0])] else {1: ps[1], 0: ps[0]}) if len(ps) == 2 \
                else {1 if C[(1, ps[0])] <= C[(0, ps[0])] else 0: ps[0]}
            for a, p in asg.items():
                c[i, p] = a; st[a] = (kp[i, p].copy(), sc[i, p].copy())
        if cut is not None and cut[i] and reverse:
            st = None
    return c

def two_sided(Fw, Bw, in_seg):
    out = Fw.copy(); n = len(in_seg); i = 0; disagree_runs = 0
    while i < n:
        if in_seg[i]:
            i += 1; continue
        j = i
        while j < n and not in_seg[j]:
            j += 1
        run_dis = False
        for k in range(i, j):
            f, b = Fw[k], Bw[k]; both = (f >= 0) & (b >= 0); dis = both & (f != b); run_dis |= bool(dis.any())
            out[k] = np.where(f < 0, b, np.where(b < 0, f, np.where(f == b, f, f if (k - i) < (j - k) else b)))
        disagree_runs += run_dis; i = j
    return out, disagree_runs

def episodes(wrong, fps):
    """Runs of consecutive wrong frames -> (count, durations in s)."""
    d = []; n = 0
    for w in list(wrong) + [False]:
        if w:
            n += 1
        elif n:
            d.append(n / fps); n = 0
    return d

def bw():
    from multiprocessing import Pool
    from sklearn.linear_model import LogisticRegression
    import identity_pipeline as ip, identity_reid2 as ir2
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_test*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(5) as pool:
        prs = pool.map(ip.prepare, ids)
    emb = ir2.make_models()["OSNet-AIN (re-identification)"]
    T = {m: dict(seg_ok=0, seg_n=0, ov_ok=0, ov_n=0, dur=[], pun_ok=0, pun_n=0) for m in ("forward carry", "two-sided carry")}; mins = 0; dis = runs = 0
    for pr in prs:
        R = np.load(os.path.join(ROOT, "external", "pose_rtm", pr["id"] + ".npz")); kp, sc = R["kp"].astype(np.float64), R["score"]; F = len(R["frame"])
        segs, pick, role = pr["segs"], pr["pick"], pr["role"]; E = emb(pr["scrops"], False); per = {}
        for (si, t, *_), e in zip(pr["scrops"], E):
            per.setdefault((si, t), []).append(e)
        per = {k: np.array(v) for k, v in per.items()}
        if len(segs) < 3 or (0, 0) not in per or (0, 1) not in per:
            continue
        isA0 = {0: True}; Xs, ys = [per[(0, 0)], per[(0, 1)]], [np.ones(len(per[(0, 0)])), np.zeros(len(per[(0, 1)]))]; clf = None
        gA, gB = per[(0, 0)].mean(0), per[(0, 1)].mean(0)
        for si in range(1, len(segs)):
            if (si, 0) not in per or (si, 1) not in per:
                isA0[si] = isA0[si - 1]; continue
            a, b = per[(si, 0)], per[(si, 1)]
            d = ((a.mean(0) @ gA + b.mean(0) @ gB) - (a.mean(0) @ gB + b.mean(0) @ gA)) if clf is None else (clf.decision_function(a).mean() - clf.decision_function(b).mean())
            isA0[si] = bool(d > 0); Xs += [a, b]; ys += [np.full(len(a), float(isA0[si])), np.full(len(b), float(not isA0[si]))]
            clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack(Xs), np.concatenate(ys))
        lab = np.full((F, kp.shape[1]), -1, np.int8); in_seg = np.zeros(F, bool)
        for si, s in enumerate(segs):
            for i, p0, p1 in zip(s["frames"], s["tr"][0], s["tr"][1]):
                lab[i, p0] = int(isA0[si]); lab[i, p1] = int(not isA0[si]); in_seg[i] = True
        Fw = carry(kp, sc, pick, lab, in_seg, pr["cut"]); Bw = carry(kp, sc, pick, lab, in_seg, pr["cut"], reverse=True)
        TS, nd = two_sided(Fw, Bw, in_seg); dis += nd; runs += sum(1 for k in range(1, F) if not in_seg[k] and in_seg[k - 1])
        v = (lab >= 0) & (role > 0); a_is_red = ((lab[v] == 1) == (role[v] == 1)).mean() >= 0.5; fps = float(R["fps"]); mins += F / fps / 60
        for name, C in (("forward carry", Fw), ("two-sided carry", TS)):
            ok = ((C == 1) == ((role == 1) == a_is_red)); val = (C >= 0) & (role > 0); t = T[name]
            t["seg_ok"] += (ok & val)[in_seg].sum(); t["seg_n"] += val[in_seg].sum(); t["ov_ok"] += (ok & val)[~in_seg].sum(); t["ov_n"] += val[~in_seg].sum()
            t["dur"] += episodes(((~ok) & val).any(1) & ~in_seg, fps)
            for e in pr["ev"]:
                i, p = e["i"], e["p"]
                if p >= 0 and C[i, p] >= 0:
                    t["pun_n"] += 1; t["pun_ok"] += bool(ok[i, p])
    print(f"BoxingWeb test rounds, {mins:.0f} min; overlaps: {runs} ({runs / mins:.1f}/min); forward and backward carry disagree in {dis} of them ({dis / mins:.2f}/min)")
    for name, t in T.items():
        d = np.array(t["dur"])
        print(f"  {name:16s} right boxer: apart frames {t['seg_ok'] / t['seg_n']:.2%} | overlap frames {t['ov_ok'] / t['ov_n']:.2%} | "
              f"wrong episodes inside overlaps {len(d) / mins:.2f}/min, median {np.median(d):.2f} s, total {d.sum() / (mins * 60):.2%} of the time | punch credit {t['pun_ok'] / t['pun_n']:.1%} of {t['pun_n']}")
    json.dump({k: {kk: (float(np.sum(vv)) if kk == "dur" else float(vv)) for kk, vv in t.items()} for k, t in T.items()}, open(os.path.join(ROOT, "runs", "identity_swaps_bw.json"), "w"), indent=1)

def head_feats(im, kp, sc):
    ok = sc[:5] > 0.3
    if ok.sum() < 2:
        return None
    c = kp[:5][ok].mean(0); r = max(0.45 * np.linalg.norm(kp[[5, 6]].mean(0) - kp[[11, 12]].mean(0)), 8)
    x1, y1, x2, y2 = int(c[0] - r), int(c[1] - 1.3 * r), int(c[0] + r), int(c[1] + 0.7 * r)
    patch = im[max(y1, 0):y2, max(x1, 0):x2]
    if patch.size < 60:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.r_[lab.mean(0), lab.std(0), np.percentile(lab, [10, 90], axis=0).ravel()]

def clip(cid):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); kp, sc = R["kp"].astype(np.float64), R["score"]; F = len(R["frame"]); fps = float(R["fps"])
    z = np.load(os.path.join(ROOT, "runs", "identity_sparring", cid + ".npz")); pick, in_seg = z["pick"], z["in_seg"]
    lab = np.where(in_seg[:, None], z["ident"], -1).astype(np.int8)
    Fw = carry(kp, sc, pick, lab, in_seg, None); Bw = carry(kp, sc, pick, lab, in_seg, None, reverse=True); TS, nd = two_sided(Fw, Bw, in_seg)
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); pos = 0; H = {}
    for i in range(F):
        f = int(R["frame"][i])
        while pos < f:
            cap.grab(); pos += 1
        ok, im = cap.read(); pos += 1
        if not ok:
            break
        for p in np.flatnonzero((Fw[i] >= 0) | (TS[i] >= 0)):
            h = head_feats(im, kp[i, p], sc[i, p])
            if h is not None:
                H[(i, p)] = h
    # head-colour A-vs-B classifier: train on stretches in odd 10-s blocks, test on even ones, then train on all stretch frames
    keys = [k for k in H if in_seg[k[0]]]; X = np.array([H[k] for k in keys]); y = np.array([lab[k] for k in keys]); blk = np.array([(k[0] // int(10 * fps)) % 2 for k in keys])
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    cv_acc = np.mean([mk().fit(X[blk == a], y[blk == a]).score(X[blk != a], y[blk != a]) for a in (0, 1)])
    clf = mk().fit(X, y); mins = F / fps / 60
    # frame-level check: with both boxers visible, does the head colour prefer the SWAPPED assignment?
    def swapped(C, held=None):
        flag = np.zeros(F, bool); known = np.zeros(F, bool)
        for i in range(F):
            ps = [p for p in np.flatnonzero(C[i] >= 0) if (i, p) in H]
            if len(ps) != 2 or C[i, ps[0]] == C[i, ps[1]]:
                continue
            d = clf.decision_function(np.array([H[(i, p)] for p in ps])); s = d[0] - d[1] if C[i, ps[0]] == 1 else d[1] - d[0]
            known[i] = True; flag[i] = s < -2.0                      # clear preference for the opposite labelling
        return flag, known
    print(f"{cid}: {mins:.1f} min, {int((~in_seg[1:] & in_seg[:-1]).sum())} overlaps ({(~in_seg[1:] & in_seg[:-1]).sum() / mins:.1f}/min); forward and backward carry disagree in {nd} ({nd / mins:.2f}/min)")
    print(f"  head-colour proxy, held-out stretches: agrees with the recognition labels {cv_acc:.1%}")
    res = {}
    for name, C in (("forward carry", Fw), ("two-sided carry", TS)):
        fl, kn = swapped(C); ov = ~in_seg
        runs_ov = [d for d in episodes(fl & ov, fps) if d >= 3 / fps]; runs_seg = [d for d in episodes(fl & in_seg, fps) if d >= 3 / fps]
        print(f"  {name:16s} overlap frames flagged as swapped: {(fl & ov).sum() / max((kn & ov).sum(), 1):.2%} | swap episodes (>= 3 frames) in overlaps {len(runs_ov) / mins:.2f}/min, "
              f"median {np.median(runs_ov) if runs_ov else 0:.2f} s, total {sum(runs_ov) / (mins * 60):.2%} of the clip | false-alarm level in apart frames {(fl & in_seg).sum() / max((kn & in_seg).sum(), 1):.2%} ({len(runs_seg) / mins:.2f} episodes/min)")
        res[name] = dict(flag_overlap=float((fl & ov).sum() / max((kn & ov).sum(), 1)), episodes_per_min=len(runs_ov) / mins, share=sum(runs_ov) / (mins * 60), false_alarm=float((fl & in_seg).sum() / max((kn & in_seg).sum(), 1)))
    np.savez_compressed(os.path.join(ROOT, "runs", "identity_sparring", cid + "_twosided.npz"), ident=TS, pick=pick, in_seg=in_seg)
    json.dump(dict(proxy_cv=float(cv_acc), **res), open(os.path.join(ROOT, "runs", "identity_sparring", cid + "_swaps.json"), "w"), indent=1)

if __name__ == "__main__":
    bw() if sys.argv[1] == "bw" else clip(sys.argv[2])
