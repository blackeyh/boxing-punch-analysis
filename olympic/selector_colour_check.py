"""Independent check of the boxer selector by KIT COLOUR (colour is not one of its cues).

Leave-one-venue-out, like `select_boxers.py fit`. On N random frames per held-out bout, classify the torso of
each selected person: red/orange vest, blue vest, white shirt (referee), other. A frame is RIGHT when exactly
one red and one blue person are selected (the two boxers always wear red/orange vs blue here).
    python olympic/selector_colour_check.py [N]
"""
import collections, glob, json, os, sys, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_boxers as sb

def torso_colour(hsv, kp):
    q = kp[[5, 6, 12, 11]]; c = q.mean(0); q = (c + 0.6 * (q - c)).astype(np.int32)
    mask = np.zeros(hsv.shape[:2], np.uint8); cv2.fillConvexPoly(mask, q, 1); px = hsv[mask > 0]
    if len(px) < 30:
        return "tiny"
    h, s, v = px[:, 0].astype(int), px[:, 1], px[:, 2]
    red = np.mean(((h < 18) | (h > 165)) & (s > 90) & (v > 60)); blue = np.mean((h > 95) & (h < 135) & (s > 80) & (v > 40))
    white = np.mean((s < 60) & (v > 140))
    best = max((red, "red"), (blue, "blue"), (white, "white"))
    return best[1] if best[0] > 0.4 else "other"

def main(N=60):
    ev = json.load(open(sb.ROOT / "olympic" / "events.json"))
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(sb.ROOT / "external" / "pose_rtm" / "O_*.npz")) if os.path.basename(f).count(".") == 1)
    data = {}
    for c in ids:
        D = sb.load(c); es = [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]]
        rows = sb.weak_labels(D, es); data[c] = (D, np.array([D["X"][i, p] for i, p, _ in rows]), np.array([y for *_, y in rows]))
    out = {}
    for held in ("kam2", "kam4"):
        tr = [c for c in ids if held not in c]; te = [c for c in ids if held in c]
        model = sb.fit_lr(np.vstack([data[c][1] for c in tr]), np.concatenate([data[c][2] for c in tr]))
        frames = collections.Counter(); people = collections.Counter()
        for c in te:
            D = data[c][0]; pick, S = sb.select(D, model)
            cap = cv2.VideoCapture(str(sb.ROOT / "external" / "raw" / f"{c}.mp4")); rng = np.random.default_rng(0)
            for i in sorted(rng.choice(len(D["frame"]), N, replace=False)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(D["frame"][i])); ok, im = cap.read()
                if not ok:
                    continue
                hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                cols = sorted(torso_colour(hsv, D["kp"][i, p]) for p in pick[i] if p >= 0)
                for x in cols: people[x] += 1
                frames["one red + one blue (both boxers)" if cols == ["blue", "red"] else
                       "only one boxer selected (red or blue alone)" if cols in (["red"], ["blue"]) else
                       "referee (white) selected" if "white" in cols else
                       "nobody selected" if not cols else "other mix: " + "+".join(cols)] += 1
        n = sum(frames.values()); print(f"\nheld-out {held}: {n} random frames from {len(te)} bouts")
        for k, v in frames.most_common(8): print(f"  {v / n:5.1%}  {k}")
        m = sum(people.values()); print("  selected people by torso colour: " + ", ".join(f"{k} {v / m:.1%}" for k, v in people.most_common()))
        out[held] = dict(frames=dict(frames), people=dict(people))
    json.dump(out, open(sb.ROOT / "runs" / "olympic_selector_colour_check.json", "w"), indent=1)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
