"""Render a clip showing the boxer selector on a HELD-OUT bout (model trained on the other gym only).
Green thick = selected as boxer; red thin = everyone else; number = P(boxer). Yellow box = referee's punch label.
    python olympic/render_selector.py ID START_SECONDS SECONDS
"""
import glob, json, os, sys, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_boxers as sb
EDGES = [(0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

def main(cid, t0, secs):
    ev = json.load(open(sb.ROOT / "olympic" / "events.json"))
    held = cid[2:6]
    ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(str(sb.ROOT / "external" / "pose_rtm" / "O_*.npz")) if os.path.basename(f).count(".") == 1 and held not in f)
    X, y = [], []
    for c in ids:
        D = sb.load(c); rows = sb.weak_labels(D, [e for e in ev if "O_" + e["video"] == c and e["end"] <= D["frame"][-1]])
        X += [D["X"][i, p] for i, p, _ in rows]; y += [l for *_, l in rows]
    model = sb.fit_lr(np.array(X), np.array(y))
    D = sb.load(cid); pick, S = sb.select(D, model); fr = D["frame"]; es = [e for e in ev if "O_" + e["video"] == cid]
    fps = 25.0; i0 = int(t0 * fps); n = int(secs * fps)
    cap = cv2.VideoCapture(str(sb.ROOT / "external" / "raw" / f"{cid}.mp4")); cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[i0])); pos = int(fr[i0])
    out_dir = sb.ROOT / "runs" / "rtm_samples"; out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{cid}_selector_{int(t0)}s.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, (1280, 720))
    for i in range(i0, min(i0 + n, len(fr))):
        while pos < int(fr[i]):
            cap.grab(); pos += 1
        ok, im = cap.read(); pos += 1
        for p in range(int(D["n"][i])):
            ch = p in pick[i]; col = (0, 255, 0) if ch else (0, 0, 255); xy = D["kp"][i, p]; sc = D["sc"][i, p]
            for a, b in EDGES:
                if sc[a] > .3 and sc[b] > .3:
                    cv2.line(im, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), col, 5 if ch else 2)
            x, yy = np.nanmin(xy, 0).astype(int)
            cv2.putText(im, f"{'BOXER ' if ch else ''}{S[i, p]:.2f}", (x, yy - 8), 0, 0.9 if ch else 0.7, col, 2)
        for e in es:
            if e["start"] - 10 <= fr[i] <= e["end"] + 10:
                b = e["boxes"][0]; cv2.rectangle(im, (int(b[1]), int(b[2])), (int(b[3]), int(b[4])), (0, 255, 255), 3)
        cv2.putText(im, "Boxer selector (trained on the OTHER gym). Green = picked as boxer, red = rejected, yellow = referee's punch label", (20, 40), 0, 0.9, (255, 255, 255), 2)
        out.write(cv2.resize(im, (1280, 720)))
    out.release(); print(path)

if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
