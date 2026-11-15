"""HPC job (CPU): cut the contact clips to disk for fine-tuning.
Per clip (olympic/landed_clips.py): video frames [f-10, f+9] (20 frames at 50 fps), fixed square crop centred on the
puncher's glove, side 2.4 body scales, resized to 224 -> one uint8 array (N, 20, 224, 224, 3), RGB, clip order = list order.
Written on node-local disk, then copied to OUT once.     python landed_extract_clips.py CLIPS.json VIDEOS_DIR OUT.npy
"""
import json, os, shutil, sys, time
import cv2, numpy as np
T, PRE = 20, 10

def crop(im, cx, cy, side):
    h, w = im.shape[:2]; s = int(max(side, 64)); x1, y1 = int(cx - s / 2), int(cy - s / 2); out = np.zeros((s, s, 3), np.uint8)
    xa, ya, xb, yb = max(x1, 0), max(y1, 0), min(x1 + s, w), min(y1 + s, h)
    if xb > xa and yb > ya:
        out[ya - y1:yb - y1, xa - x1:xb - x1] = im[ya:yb, xa:xb]
    return cv2.cvtColor(cv2.resize(out, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

def main():
    clips = json.load(open(sys.argv[1])); vids, out = sys.argv[2], sys.argv[3]; N = len(clips)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "landed_clips.npy")
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
                        c = clips[n]; A[n, j] = crop(im, c["cx"], c["cy"], c["side"]); filled[n, j] = True
            k += 1
        print(f"[{cid}] {len(mine)} clips, {time.time() - t0:.0f}s", flush=True)
    for n in range(N):                                        # frames past the ends of a video: repeat the nearest real frame
        if not filled[n].all() and filled[n].any():
            good = np.flatnonzero(filled[n])
            for j in np.flatnonzero(~filled[n]):
                A[n, j] = A[n, good[np.abs(good - j).argmin()]]
    A.flush(); del A
    print(f"missing frames filled in {int((~filled).any(1).sum())} clips; copying", flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True); shutil.copy(tmp, out + ".part"); os.replace(out + ".part", out); os.remove(tmp); print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
