"""Identity, the real pipeline (no truth handed over), scored AT THE PUNCH.

Per BoxingWeb test round:
  1. boxers  : the learned skeleton-only selector (top 2 per frame), runs/selector_learned/gbm_boxingweb_train.joblib
  2. stretches where the two picks are clearly apart; pixel camera cuts split them (identity_episodes.segments)
  3. identity: OSNet-AIN re-identification embeddings (colour). Stretch 0 defines boxer A and boxer B; every later
     stretch is labelled by an A-vs-B classifier that keeps learning from its own answers. No correction, ever.
  4. at each labelled punch (middle frame), the puncher's skeleton gets an identity by
       carry  : the stretch it is in, else followed by motion from the last stretch
       direct : crop the puncher at the punch frame and ask the classifier (user's idea)
       both   : stretch identity when in a stretch, else direct if the classifier is confident, else carry
Truth is used ONLY to score: which RTMPose person is the labelled puncher (role == punch side), and whether
A/B = red/blue or blue/red (one global choice per round, so 50% = chance).
    python boxingweb/identity_pipeline.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "olympic"))
import mem_guard  # noqa: F401
import cv2, joblib, numpy as np
import feats, identity_episodes as ie, identity_reid2 as ir2, train_selector as ts
CONF = 1.0                                             # |classifier decision| above which "direct" overrides "carry"

def crops_at(R, cid, items):
    """items [(frame index, person)] -> tight 256x128 RGB crops (None where it fails)."""
    W, H = int(R["width"]), int(R["height"]); cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4")); out = {}; pos = -1
    for i in sorted({i for i, _ in items}):
        f = int(R["frame"][i])
        if f != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read(); pos = f
        for j, p in [(j, p) for j, p in items if j == i]:
            if not ok:
                continue
            x1, y1, x2, y2 = R["box"][i, p]; w, h = x2 - x1, y2 - y1
            x1, y1, x2, y2 = int(max(x1 - .05 * w, 0)), int(max(y1 - .05 * h, 0)), int(min(x2 + .05 * w, W)), int(min(y2 + .05 * h, H))
            c = im[y1:y2, x1:x2]
            if c.size:
                out[(j, p)] = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (128, 256), interpolation=cv2.INTER_AREA)
    return out

def prepare(cid):
    """Everything that needs no neural network: picks, stretches, stretch crops, punch crops."""
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); D = feats.load(cid)
    gbm = joblib.load(os.path.join(ROOT, "runs", "selector_learned", "gbm_boxingweb_train.joblib"))
    m = D["tid"] >= 0; P = np.full(m.shape, np.nan); P[m] = gbm.predict_proba(D["X2"][m])[:, 1]
    pick = ts.pick_pair(D, ts.smooth(D, np.nan_to_num(P)), pair=False)
    pseudo = np.zeros(m.shape, np.int8)
    for i, pk in enumerate(pick):
        for p in pk:
            if p >= 0:
                pseudo[i, p] = 1
    cut = np.load(os.path.join(ROOT, "external", "cuts", cid + ".npz"))["cut"][R["frame"]]
    rd = dict(id=cid, segs=ie.segments(R, pseudo, cut)); scrops = ir2.crops_for_round(rd)
    role = np.load(os.path.join(ROOT, "external", "boxers_gt", cid + ".npz"))["role"]
    ev = [e for e in json.load(open(os.path.join(HERE, "events.json"))) if e["video"] == cid]
    items = []
    for e in ev:
        i = min((e["start"] + e["end"]) // 2, len(R["frame"]) - 1); want = 1 if e["side"] == "red" else 2
        p = np.flatnonzero(role[i] == want); e["i"] = int(i); e["p"] = int(p[0]) if len(p) else -1
        if len(p):
            items.append((int(i), int(p[0])))
    return dict(id=cid, segs=rd["segs"], scrops=scrops, pcrops=crops_at(R, cid, items), pick=pick, role=role, ev=ev, cut=cut)

def run_round(pr, emb):
    from sklearn.linear_model import LogisticRegression
    cid = pr["id"]; R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); kp, sc = R["kp"].astype(np.float64), R["score"]
    segs, pick, role = pr["segs"], pr["pick"], pr["role"]; F = len(R["frame"])
    E = emb(pr["scrops"], False); per = {}
    for (si, t, *_), e in zip(pr["scrops"], E):
        per.setdefault((si, t), []).append(e)
    per = {k: np.array(v) for k, v in per.items()}
    if len(segs) < 3 or (0, 0) not in per or (0, 1) not in per:
        return None
    # --- identity of every stretch: is track 0 boxer A?
    isA0 = {0: True}; Xs, ys = [per[(0, 0)], per[(0, 1)]], [np.ones(len(per[(0, 0)])), np.zeros(len(per[(0, 1)]))]; clf = None
    gA, gB = per[(0, 0)].mean(0), per[(0, 1)].mean(0)
    for si in range(1, len(segs)):
        if (si, 0) not in per or (si, 1) not in per:
            isA0[si] = isA0[si - 1]; continue
        a, b = per[(si, 0)], per[(si, 1)]
        if clf is None:
            d = (a.mean(0) @ gA + b.mean(0) @ gB) - (a.mean(0) @ gB + b.mean(0) @ gA)
        else:
            d = clf.decision_function(a).mean() - clf.decision_function(b).mean()
        isA0[si] = bool(d > 0)
        Xs += [a, b]; ys += [np.full(len(a), float(isA0[si])), np.full(len(b), float(not isA0[si]))]
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack(Xs), np.concatenate(ys))
    # --- per-frame identity of the picked people: 1 = A, 0 = B, -1 = unknown
    ident = np.full(pick.shape[:1] + (kp.shape[1],), -1, np.int8); in_seg = np.zeros(F, bool)
    for si, s in enumerate(segs):
        for i, p0, p1 in zip(s["frames"], s["tr"][0], s["tr"][1]):
            ident[i, p0] = int(isA0[si]); ident[i, p1] = int(not isA0[si]); in_seg[i] = True
    carry = ident.copy(); st = None
    for i in range(F):
        if pr["cut"][i]:
            st = None
        ps = [p for p in pick[i] if p >= 0]
        if in_seg[i]:
            st = {int(ident[i, p]): (kp[i, p].copy(), sc[i, p].copy()) for p in ps}; continue
        if st is None or len(st) < 2 or not ps:
            continue
        C = {(a, p): ie.pdist(st[a][0], st[a][1], kp[i, p], sc[i, p]) for a in (0, 1) for p in ps}
        if len(ps) == 2:
            asg = {1: ps[0], 0: ps[1]} if C[(1, ps[0])] + C[(0, ps[1])] <= C[(1, ps[1])] + C[(0, ps[0])] else {1: ps[1], 0: ps[0]}
        else:
            asg = {1 if C[(1, ps[0])] <= C[(0, ps[0])] else 0: ps[0]}
        for a, p in asg.items():
            carry[i, p] = a; st[a] = (kp[i, p].copy(), sc[i, p].copy())
    # --- A/B <-> red/blue: one choice per round, from the stretch frames (scoring only)
    v = (ident >= 0) & (role > 0); agree = ((ident[v] == 1) == (role[v] == 1)).mean() if v.any() else 0.5
    a_is_red = agree >= 0.5; seg_right = max(agree, 1 - agree); nseg = int(v.sum())
    # --- punches
    keys = [(e["i"], e["p"]) for e in pr["ev"] if e["p"] >= 0 and (e["i"], e["p"]) in pr["pcrops"]]
    dec = {}
    if keys and clf is not None:
        Ep = emb([(None, None, None, pr["pcrops"][k]) for k in keys], False)
        dec = dict(zip(keys, clf.decision_function(Ep)))
    out = dict(punches=0, no_skel=0, not_picked=0, carry=0, carry_n=0, direct=0, direct_n=0, both=0, both_n=0, in_seg=0, seg_right=seg_right * nseg, seg_n=nseg,
               stretches=len(segs), a_is_red=bool(a_is_red))
    for e in pr["ev"]:
        out["punches"] += 1; i, p = e["i"], e["p"]; red = e["side"] == "red"
        if p < 0:
            out["no_skel"] += 1; continue
        if p not in pick[i]:
            out["not_picked"] += 1; continue
        out["in_seg"] += bool(in_seg[i])
        right = lambda isA: (isA == a_is_red) == red
        c = carry[i, p]; d = dec.get((i, p))
        if c >= 0:
            out["carry_n"] += 1; out["carry"] += right(bool(c))
        if d is not None:
            out["direct_n"] += 1; out["direct"] += right(d > 0)
        b = bool(ident[i, p]) if ident[i, p] >= 0 else (d > 0 if d is not None and (abs(d) >= CONF or c < 0) else (bool(c) if c >= 0 else None))
        if b is not None:
            out["both_n"] += 1; out["both"] += right(b)
    return out

def main():
    from multiprocessing import Pool
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_test*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(5) as pool:
        prs = pool.map(prepare, ids)
    emb = ir2.make_models()["OSNet-AIN (re-identification)"]; T = {}
    for pr in prs:
        o = run_round(pr, emb)
        if o is None:
            print(pr["id"][7:45], "skipped (too few stretches)"); continue
        print(f"{pr['id'][7:45]:38s} stretches {o['stretches']:3d} | stretch frames right {o['seg_right'] / max(o['seg_n'], 1):5.1%} | punches {o['punches']:3d}: "
              f"credit right carry {o['carry'] / max(o['carry_n'], 1):5.1%}, direct {o['direct'] / max(o['direct_n'], 1):5.1%}, both {o['both'] / max(o['both_n'], 1):5.1%}", flush=True)
        for k, v in o.items():
            if k != "a_is_red":
                T[k] = T.get(k, 0) + v
    print(f"\nALL: {T['punches']} labelled punches in {len(prs)} test rounds")
    print(f"  puncher has no RTMPose skeleton matched to the truth: {T['no_skel'] / T['punches']:.1%}; has one but the selector did not pick it: {T['not_picked'] / T['punches']:.1%}")
    print(f"  punches inside an 'apart' stretch: {T['in_seg'] / T['punches']:.1%} (the rest happen while the boxers overlap)")
    print(f"  identity in the stretch frames: {T['seg_right'] / T['seg_n']:.1%} right")
    for k, name in (("carry", "carry (stretch, else follow by motion)"), ("direct", "direct (crop the puncher at the punch, ask the classifier)"), ("both", "both")):
        print(f"  credit to the right boxer, {name}: {T[k] / T[k + '_n']:.1%} of {T[k + '_n']} punches ({T[k + '_n'] / T['punches']:.0%} of all)")
    json.dump({k: float(v) for k, v in T.items()}, open(os.path.join(ROOT, "runs", "identity_pipeline.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
