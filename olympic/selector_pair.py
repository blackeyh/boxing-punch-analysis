"""Selector experiment: choose the two boxers JOINTLY, with a depth-consistency rule.

Two boxers exchanging punches stand at the same distance from the camera, so they have the same APPARENT SIZE.
A spectator metres behind the ring can overlap a boxer on screen (user's objection) but is visibly smaller.
  size(person)   = median over +-1 s of the tracklet of a robust body scale (max of torso, 0.9 * thigh+shin)
  pair (i, j) OK = size ratio within [1/R, R] and hips within D body-scales of each other
  pick           = the OK pair with the highest P_i + P_j (both >= THR_LOW, at least one >= THR); else the single best.
    python olympic/selector_pair.py            # leave-one-gym-out numbers
"""
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_boxers as sb
import selector_ring as sr

R, DMAX, THR, THR_LOW = 1.5, 4.0, 0.5, 0.25

def body_scale(D):
    kp = D["kp"]; n = lambda a, b: np.linalg.norm(kp[:, :, a] - kp[:, :, b], axis=-1)
    sh = (kp[:, :, 5] + kp[:, :, 6]) / 2; hip = (kp[:, :, 11] + kp[:, :, 12]) / 2
    torso = np.linalg.norm(sh - hip, axis=-1)
    leg = np.maximum(n(11, 13) + n(13, 15), n(12, 14) + n(14, 16))
    legok = (D["sc"][:, :, [13, 14, 15, 16]] > 0.3).all(-1)
    raw = np.where(legok, np.maximum(torso, 0.55 * leg), torso)      # leg ~ 1.8 x torso when standing
    S = np.full(raw.shape, np.nan); fs, ks = np.nonzero(D["tid"] >= 0); t = D["tid"][fs, ks]
    order = np.argsort(t, kind="stable"); fs, ks, t = fs[order], ks[order], t[order]
    bounds = np.flatnonzero(np.diff(t)) + 1
    for f, k in zip(np.split(fs, bounds), np.split(ks, bounds)):
        v = raw[f, k]
        for j in range(len(f)):
            a, b = np.searchsorted(f, f[j] - sb.WIN), np.searchsorted(f, f[j] + sb.WIN, "right")
            S[f[j], k[j]] = np.median(v[a:b])
    return S, hip

def select_pair(D, model, R=R, DMAX=DMAX):
    _, P = sb.select(D, model)                                   # smoothed P(boxer)
    scale, hip = body_scale(D); F = len(D["frame"]); pick = np.full((F, 2), -1, np.int32)
    for f in range(F):
        c = [k for k in range(int(D["n"][f])) if P[f, k] >= THR_LOW]
        best, bs = None, -1
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                i, j = c[a], c[b]
                if max(P[f, i], P[f, j]) < THR:
                    continue
                r = scale[f, i] / scale[f, j]
                if not (1 / R <= r <= R):
                    continue
                if np.linalg.norm(hip[f, i] - hip[f, j]) > DMAX * max(scale[f, i], scale[f, j]):
                    continue
                if P[f, i] + P[f, j] > bs:
                    best, bs = (i, j), P[f, i] + P[f, j]
        if best:
            pick[f] = best
        else:
            c = [k for k in c if P[f, k] >= THR]
            if c:
                pick[f, 0] = max(c, key=lambda k: P[f, k])
    return pick, P

def main():
    ev = json.load(open(sb.ROOT / "olympic" / "events.json"))
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(sb.ROOT / "external" / "pose_rtm" / "O_*.npz")) if os.path.basename(f).count(".") == 1)
    data = {}
    for c in ids:
        D = sb.load(c); es = [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]]; data[c] = (D, es, sb.weak_labels(D, es))
    for held in ("kam2", "kam4"):
        tr = [c for c in ids if held not in c]; te = [c for c in ids if held in c]
        m1 = sb.fit_lr(np.vstack([[data[c][0]["X"][i, p] for i, p, _ in data[c][2]] for c in tr]), np.concatenate([[l for *_, l in data[c][2]] for c in tr]))
        tot = {"top 2 independently": np.zeros(6), "pair with same apparent size": np.zeros(6)}
        for c in te:
            D, es, _ = data[c]
            for name, pick in (("top 2 independently", sb.select(D, m1)[0]), ("pair with same apparent size", select_pair(D, m1)[0])):
                k, n, w, s = sr.evaluate(D, es, pick); two = (pick >= 0).sum(1); tot[name] += [k, n, w, s, (two == 2).sum(), len(two)]
        print(f"\nheld-out {held}")
        for name, t in tot.items():
            print(f"  {name:30s} puncher kept {t[0] / t[1]:.1%} of {int(t[1])} | picks nowhere near the punch {t[2] / t[3]:.1%} | frames with 2 selected {t[4] / t[5]:.1%}")

if __name__ == "__main__":
    main()
