"""Sheets of random HELD-OUT frames showing only the selector's picks, for judging by eye.
Same seed -> same frames for every method, so methods can be compared frame by frame.
    python olympic/render_random_frames.py OUT_DIR {top2|pair|learned}
"""
import glob, json, os, sys, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_boxers as sb, selector_pair as sp
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
out, method = sys.argv[1], sys.argv[2]
ev = json.load(open(sb.ROOT / "olympic" / "events.json"))
ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(sb.ROOT / "external" / "pose_rtm" / "O_*.npz")) if os.path.basename(f).count(".") == 1)
data = {}
for c in ids:
    D = sb.load(c); data[c] = (D, sb.weak_labels(D, [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]]))
rng = np.random.default_rng(7)
for held in ("kam2", "kam4"):
    tr = [c for c in ids if held not in c]; te = [c for c in ids if held in c]
    m1 = sb.fit_lr(np.vstack([[data[c][0]["X"][i, p] for i, p, _ in data[c][1]] for c in tr]), np.concatenate([[l for *_, l in data[c][1]] for c in tr]))
    tiles = []
    for c in rng.choice(te, 12, replace=False):
        D = data[c][0]; i = int(rng.integers(100, len(D["frame"]) - 100))
        if method == "learned":        # picks saved by boxingweb/train_selector.py (trained on BoxingWeb only)
            pick = np.load(str(sb.ROOT / "external" / "boxers_rtm" / f"{c}.npz"))["pick"]
        else:
            pick = (sp.select_pair(D, m1) if method == "pair" else sb.select(D, m1))[0]
        cap = cv2.VideoCapture(str(sb.ROOT / "external" / "raw" / f"{c}.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, int(D["frame"][i])); ok, im = cap.read()
        for p in pick[i]:
            if p < 0:
                continue
            xy = D["kp"][i, p]; sc = D["sc"][i, p]
            for a, b in EDGES:
                if sc[a] > .3 and sc[b] > .3:
                    cv2.line(im, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), (0, 255, 0), 6)
        cv2.putText(im, f"{len(tiles) + 1}: {c[2:]} f{D['frame'][i]}", (20, 60), 0, 1.8, (0, 255, 255), 4)
        tiles.append(cv2.resize(im[100:980], (760, 348)))
    cv2.imwrite(f"{out}/rand_{held}_{method}.jpg", np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 12, 3)]), [cv2.IMWRITE_JPEG_QUALITY, 78])
