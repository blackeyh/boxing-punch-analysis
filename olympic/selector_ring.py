"""Selector experiment: add a DEPTH cue the flat image lacks -- are the person's feet on the ring floor?

(User's objection, 2026-11-14: "near each other on screen" is meaningless in 2D; a spectator metres behind the
ring can sit next to a boxer in the image.)  The ring floor is a plane seen by a fixed camera, so where the feet
touch it in the image says where the person stands in the room.
  Stage 1: the pose/motion selector (select_boxers) scores everyone.
  Ring   : per bout, with NO labels: convex hull of the ankle positions of people stage 1 is very sure about
           (P > 0.9), outliers trimmed.
  Stage 2: stage-1 features + [a foot inside the ring, no visible foot] -> logistic regression.
Leave-one-gym-out. Reports, at the referee-labelled contact punches of the held-out gym:
  puncher kept   : a selected person has a wrist in the referee's box
  wrong pick     : share of selected people who are nowhere near the box (referee / crowd)
    python olympic/selector_ring.py
"""
import glob, json, os, sys, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_boxers as sb

def ring_features(D, P):
    """-> (F,K,2): [a visible foot inside the ring hull, no visible foot]; hull from confident stage-1 boxers."""
    F, K = D["tid"].shape; out = np.zeros((F, K, 2), np.float32)
    ank, asc = D["kp"][:, :, [15, 16]], D["sc"][:, :, [15, 16]] > 0.3
    fs, ks = np.nonzero(np.nan_to_num(P) > 0.9)
    pts = np.vstack([ank[i, p][asc[i, p]] for i, p in zip(fs[::3], ks[::3])] or [np.zeros((0, 2))])
    if len(pts) < 50:
        return out, None
    ctr = np.median(pts, 0); d = np.linalg.norm((pts - ctr) / (pts.std(0) + 1e-6), axis=1); pts = pts[d < np.percentile(d, 97)]
    hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.float32)
    for f in range(F):
        for p in range(int(D["n"][f])):
            v = ank[f, p][asc[f, p]]
            if len(v) == 0:
                out[f, p, 1] = 1
            else:
                out[f, p, 0] = float(any(cv2.pointPolygonTest(hull, (float(q[0]), float(q[1])), False) >= 0 for q in v))
    return out, hull

def evaluate(D, es, pick):
    kept = n = wrong = sel_n = 0
    for e in es:
        if e["outcome"] == "miss":
            continue
        i = int(np.clip(np.searchsorted(D["frame"], (e["start"] + e["end"]) // 2), 0, len(D["frame"]) - 1))
        m = D["n"][i]; box = sb.event_box(e); ok = D["sc"][i, :m] > 0.3
        w = (sb.inside(D["kp"][i, :m][:, sb.WR], box) & ok[:, sb.WR]).any(1)
        b = (sb.inside(D["kp"][i, :m][:, sb.HEADTORSO], box) & ok[:, sb.HEADTORSO]).any(1)
        touching = (sb.inside(D["kp"][i, :m], box, 60) & ok).any(1)
        if w.sum() < 1 or (w | b).sum() != 2:
            continue
        sel = [p for p in pick[i] if p >= 0]
        n += 1; kept += any(w[p] for p in sel); sel_n += len(sel); wrong += sum(not touching[p] for p in sel)
    return kept, n, wrong, sel_n

def main():
    ev = json.load(open(sb.ROOT / "olympic" / "events.json"))
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(sb.ROOT / "external" / "pose_rtm" / "O_*.npz")) if os.path.basename(f).count(".") == 1)
    data = {}
    for c in ids:
        D = sb.load(c); es = [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]]
        data[c] = (D, es, sb.weak_labels(D, es))
    res = {}
    for held in ("kam2", "kam4"):
        tr = [c for c in ids if held not in c]; te = [c for c in ids if held in c]
        X1 = np.vstack([[data[c][0]["X"][i, p] for i, p, _ in data[c][2]] for c in tr]); y = np.concatenate([[l for *_, l in data[c][2]] for c in tr])
        m1 = sb.fit_lr(X1, y)
        ring = {}
        for c in ids:                                         # ring per bout from stage 1 (no labels used)
            D = data[c][0]; P = np.where(D["tid"] >= 0, sb.prob(m1, np.nan_to_num(D["X"])), np.nan)
            ring[c] = ring_features(D, P)[0]
        X2 = np.vstack([[np.r_[data[c][0]["X"][i, p], ring[c][i, p]] for i, p, _ in data[c][2]] for c in tr])
        m2 = sb.fit_lr(X2, y); m2["names"] = sb.NAMES + ["foot_in_ring", "no_foot"]
        print(f"\nheld-out {held}: stage-2 weights: " + ", ".join(f"{n} {w:+.2f}" for n, w in zip(m2["names"], m2["w"])))
        tot = {"pose/motion only": np.zeros(6), "+ feet on the ring floor": np.zeros(6)}
        for c in te:
            D, es, _ = data[c]
            for name, model, X in (("pose/motion only", m1, D["X"]), ("+ feet on the ring floor", m2, np.concatenate([D["X"], ring[c]], -1))):
                pick, S = sb.select(dict(D, X=X), model)
                k, n, w, s = evaluate(D, es, pick); two = (pick >= 0).sum(1)
                tot[name] += [k, n, w, s, (two == 2).sum(), len(two)]
        for name, t in tot.items():
            print(f"  {name:26s} puncher kept {t[0] / t[1]:.1%} of {int(t[1])} | wrong pick {t[2] / t[3]:.1%} of {int(t[3])} selected | frames with 2 selected {t[4] / t[5]:.1%}")
        res[held] = {k: v.tolist() for k, v in tot.items()}
    json.dump(res, open(sb.ROOT / "runs" / "olympic_selector_ring.json", "w"), indent=1)

if __name__ == "__main__":
    main()
