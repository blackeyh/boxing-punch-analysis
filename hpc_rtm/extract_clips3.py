"""HPC job (CPU): cut LONG clips to disk, with the puncher's UPPER ARM drawn in (magenta line, about 3 px at 224):
clip["arm"][j] = [sx, sy, ex, ey] in video pixels or null (olympic/landed_clips_arm.py). Tells the model WHICH punch.
Otherwise identical to extract_clips2.py: Per clip: video frames [f-PRE, f-PRE+T-1], one fixed square crop centred on
(cx, cy), side `side`, resized to 224 -> uint8 (N, T, 224, 224, 3) RGB, clip order = list order.
Written on node-local disk, then copied once.   python extract_clips3.py CLIPS.json VIDEOS_DIR OUT.npy T PRE
"""
import json, os, shutil, sys, time
import cv2, numpy as np

def crop(im, cx, cy, side, arm=None):
    h, w = im.shape[:2]; s = int(max(side, 64)); x1, y1 = int(cx - s / 2), int(cy - s / 2); out = np.zeros((s, s, 3), np.uint8)
    xa, ya, xb, yb = max(x1, 0), max(y1, 0), min(x1 + s, w), min(y1 + s, h)
    if xb > xa and yb > ya:
        out[ya - y1:yb - y1, xa - x1:xb - x1] = im[ya:yb, xa:xb]
    if arm is not None:
        cv2.line(out, (int(arm[0] - x1), int(arm[1] - y1)), (int(arm[2] - x1), int(arm[3] - y1)), (255, 0, 255), max(2, int(round(3 * s / 224))), cv2.LINE_AA)
    return cv2.cvtColor(cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

def main():
    clips = json.load(open(sys.argv[1])); vids, out, T, PRE = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]); N = len(clips)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "clips2.npy")
    A = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(N, T, 224, 224, 3)); t0 = time.time(); filled = np.zeros((N, T), bool)
    for cid in sorted({c["id"] for c in clips}):
        mine = [(n, c) for n, c in enumerate(clips) if c["id"] == cid]; need = {}
        for n, c in mine:
            for j in range(T):
                need.setdefault(c["frame"] - PRE + j, []).append((n, j))
        cap = cv2.VideoCapture(os.path.join(vids, cid + ".mp4")); k = 0; last = max(need)
        while k <= last:
            if not cap.grab():
                break
            if k in need:
                ok, im = cap.retrieve()
                if ok:
                    for n, j in need[k]:
                        c = clips[n]; A[n, j] = crop(im, c["cx"], c["cy"], c["side"], c["arm"][j] if j < len(c["arm"]) else None); filled[n, j] = True
            k += 1
        print(f"[{cid}] {len(mine)} clips, {time.time() - t0:.0f}s", flush=True)
    for n in range(N):
        if not filled[n].all() and filled[n].any():
            good = np.flatnonzero(filled[n])
            for j in np.flatnonzero(~filled[n]):
                A[n, j] = A[n, good[np.abs(good - j).argmin()]]
    A.flush(); del A; print(f"missing frames filled in {int((~filled).any(1).sum())} clips; copying", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True); shutil.copy(tmp, out + ".part"); os.replace(out + ".part", out); os.remove(tmp); print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
