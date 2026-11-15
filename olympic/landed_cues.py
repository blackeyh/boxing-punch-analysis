"""LANDED = made contact with the head or body (not blocked, not missed)  [definition agreed with the user 2026-11-14].
Labels: Olympic set, referee: head / body = landed (63%), block (12%), miss (26%), with a box at the contact point.

Step 2 of the landed plan: do cheap PHYSICAL cues from the two skeletons separate the outcomes?
For each labelled punch: the two boxers = the learned selector's picks at the contact frame; the puncher = the pick
whose labelled-hand glove is nearer the referee's box (label information; at run time the punch detector gives the
puncher and the hand). Glove = wrist pushed 40% of the forearm beyond the wrist. Window: 3 frames before to 5 after
contact at 25 fps. All distances in units of the puncher's body scale.
  reach    : closest the glove gets to the opponent's head / torso
  guard    : closest the glove gets to the opponent's wrists and forearms; guard minus reach; how high the guard is
  reaction : how far the opponent's head / torso moves after contact, and how much of it is along the punch
  arm      : extension at contact, glove speed before and after, slow-down
  spacing  : distance between the boxers, overlap of their boxes
Reported: each cue's AUC for landed-vs-missed, landed-vs-blocked, blocked-vs-missed (0.5 = no information), then a
boosted-tree model on all cues, trained on one gym and tested on the other.
    python olympic/landed_cues.py
"""
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
NAMES = ["reach_head", "reach_torso", "reach_any", "head_at_contact", "guard_min", "guard_minus_reach", "guard_height", "head_move", "head_move_along",
         "torso_move", "torso_move_along", "extension", "extension_max", "speed_before", "speed_after", "slow_down", "separation", "box_overlap", "right_hand"]
PRE, POST = 3, 5

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)

def track(kp, sc, i0, p0, lo, hi):
    """Follow person p0 from frame i0 over [lo, hi] by nearest pose -> (frames, kp, sc)."""
    out = {i0: p0}
    for rng in (range(i0 + 1, hi + 1), range(i0 - 1, lo - 1, -1)):
        prev = kp[i0, p0]
        for i in rng:
            ok = ~np.isnan(kp[i, :, 0, 0])
            if not ok.any():
                break
            d = np.linalg.norm(kp[i] - prev[None], axis=-1).mean(-1); d[~ok] = np.inf; p = int(d.argmin())
            if d[p] > 0.5 * max(np.ptp(prev, 0).max(), 20):
                break
            out[i] = p; prev = kp[i, p]
    f = sorted(out); return np.array(f), np.stack([kp[i, out[i]] for i in f]), np.stack([sc[i, out[i]] for i in f])

def cues(R, pick, e):
    kp, sc, box, fr = R["kp"].astype(np.float64), R["score"], R["box"], R["frame"]
    i = int(np.clip(np.searchsorted(fr, (e["start"] + e["end"]) // 2), 0, len(fr) - 1)); ps = [p for p in pick[i] if p >= 0]
    if len(ps) != 2:
        return None
    w = 9 if e["hand"] == "left" else 10; el, shd = w - 2, w - 4
    glove = lambda k: k[..., w, :] + 0.4 * (k[..., w, :] - k[..., el, :])
    b = np.array(e["boxes"]); c = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
    d = [np.linalg.norm(glove(kp[i, p]) - c) for p in ps]; a, o = (ps[0], ps[1]) if d[0] <= d[1] else (ps[1], ps[0])
    lo, hi = max(i - PRE, 0), min(i + POST, len(fr) - 1)
    fa, ka, _ = track(kp, sc, i, a, lo, hi); fo, ko, so = track(kp, sc, i, o, lo, hi)
    com = np.intersect1d(fa, fo)
    if len(com) < 4 or i not in com:
        return None
    ka = ka[np.isin(fa, com)]; ko = ko[np.isin(fo, com)]; so = so[np.isin(fo, com)]; t = int(np.flatnonzero(com == i)[0])
    S = max(np.median(np.linalg.norm(ka[:, [5, 6]].mean(1) - ka[:, [11, 12]].mean(1), axis=1)), 8.0)
    g = glove(ka); head = ko[:, :5].mean(1); torso = ko[:, [5, 6, 11, 12]].mean(1)
    guard_pts = np.concatenate([ko[:, [9, 10]], (ko[:, [9, 10]] + ko[:, [7, 8]]) / 2, ko[:, [9, 10]] + 0.4 * (ko[:, [9, 10]] - ko[:, [7, 8]])], 1)
    dh = np.linalg.norm(g - head, axis=1) / S; dt = np.linalg.norm(g - torso, axis=1) / S; dg = np.linalg.norm(g[:, None] - guard_pts, axis=-1).min(1) / S
    direc = g[t] - ka[t, shd]; direc = direc / max(np.linalg.norm(direc), 1e-6)
    pre, post = slice(0, t), slice(min(t + 2, len(com) - 1), len(com))
    def move(x):
        if t == 0:
            return np.nan, np.nan
        v = (x[post].mean(0) - x[pre].mean(0)) - (ka[post][:, [11, 12]].mean((0, 1)) - ka[pre][:, [11, 12]].mean((0, 1))) * 0      # image motion, camera is fixed
        return np.linalg.norm(v) / S, float(v @ direc) / S
    hm, hma = move(head); tm, tma = move(torso)
    arm = np.linalg.norm(ka[:, shd] - ka[:, el], axis=1) + np.linalg.norm(ka[:, el] - ka[:, w], axis=1)
    ext = np.linalg.norm(ka[:, shd] - ka[:, w], axis=1) / np.maximum(arm, 1e-6)
    sp = np.linalg.norm(np.diff(g, axis=0), axis=1) / S
    sb = sp[:t].mean() if t > 0 else np.nan; sa = sp[t:t + 3].mean() if t < len(sp) else np.nan
    Sopp = max(np.median(np.linalg.norm(ko[:, [5, 6]].mean(1) - ko[:, [11, 12]].mean(1), axis=1)), 8.0)
    return [dh.min(), dt.min(), min(dh.min(), dt.min()), dh[t], dg.min(), dg.min() - min(dh.min(), dt.min()), float(((head[t, 1] - ko[t, [9, 10], 1]) / Sopp).mean()),
            hm, hma, tm, tma, ext[t], ext.max(), sb, sa, sb - sa, np.linalg.norm(ka[t, [11, 12]].mean(0) - ko[t, [11, 12]].mean(0)) / S, iou(box[i, a], box[i, o]), float(w == 10)]

def auc(y, s):
    m = ~np.isnan(s); y, s = y[m], s[m]; o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)

def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    ev = json.load(open(os.path.join(HERE, "events.json"))); X, y, gym = [], [], []; n = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "pose_rtm", "O_*.npz"))):
        if os.path.basename(f).count(".") != 1:
            continue
        cid = os.path.basename(f)[:-4]; R = np.load(f); pick = np.load(os.path.join(ROOT, "external", "boxers_rtm", cid + ".npz"))["pick"]
        for e in [e for e in ev if "O_" + e["video"] == cid]:
            n += 1; c = cues(R, pick, e)
            if c is not None:
                X.append(c); y.append({"head": 0, "body": 0, "block": 1, "miss": 2}[e["outcome"]]); gym.append(cid[2:6])
    X, y, gym = np.array(X, np.float64), np.array(y), np.array(gym)
    print(f"{n} labelled punches, {len(y)} usable (both boxers picked and followed around contact: {len(y) / n:.0%}): landed {np.mean(y == 0):.0%}, blocked {np.mean(y == 1):.0%}, missed {np.mean(y == 2):.0%}")
    print(f"\n{'cue':20s} landed-vs-missed  landed-vs-blocked  blocked-vs-missed   (AUC; 0.5 = nothing, further from 0.5 = more)")
    for k, name in enumerate(NAMES):
        r = []
        for a, b in ((0, 2), (0, 1), (1, 2)):
            m = (y == a) | (y == b); r.append(auc((y[m] == a).astype(int), X[m, k]))
        print(f"{name:20s} {r[0]:14.2f} {r[1]:18.2f} {r[2]:18.2f}")
    res = {}; cm = np.zeros((3, 3), int); pb, yb = [], []
    for held in ("kam2", "kam4"):
        tr, te = gym != held, gym == held
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=0).fit(X[tr], y[tr])
        P = m.predict_proba(X[te]); yh = P.argmax(1); np.add.at(cm, (y[te], yh), 1); pb.append(1 - P[:, 0]); yb.append((y[te] != 0).astype(int))
    acc = np.trace(cm) / cm.sum(); f1 = []
    for k in range(3):
        pr = cm[k, k] / max(cm[:, k].sum(), 1); rc = cm[k, k] / max(cm[k].sum(), 1); f1.append(2 * pr * rc / max(pr + rc, 1e-9))
    print(f"\nAll cues together, trained on one gym and tested on the other ({cm.sum()} punches):")
    print(f"  3-way accuracy {acc:.1%} (always saying 'landed' = {np.mean(y == 0):.1%}), macro-F1 {np.mean(f1):.3f}; F1 landed {f1[0]:.2f}, blocked {f1[1]:.2f}, missed {f1[2]:.2f}")
    print("  rows = truth, columns = predicted (landed, blocked, missed):"); print("   " + str(cm.tolist()))
    print(f"  landed vs not landed: AUC {auc(1 - np.concatenate(yb), -np.concatenate(pb)):.3f}")
    json.dump(dict(n=int(len(y)), accuracy=float(acc), macro_f1=float(np.mean(f1)), f1=[float(v) for v in f1], confusion=cm.tolist()), open(os.path.join(ROOT, "runs", "olympic_landed_cues.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(ROOT, "external", "olympic_landed_cues.npz"), X=X, y=y, gym=gym)

if __name__ == "__main__":
    main()
