"""Which boxer is which, SKELETON-ONLY: baseline measurements on BoxingWeb data_test (truth known in every frame).

To isolate identity from selection, the two boxers are given (the RTMPose people matched to the red / blue truth).
A. MOTION CONTINUITY tracker: each fighter's next pose is predicted from the last pose + velocity; detections are
   assigned to fighters by pose distance (2x2). No appearance. A swap, once made, persists.
   Reported: share of boxer-frames with the right identity; swaps per minute; where swaps happen.
   Broadcast footage has camera CUTS (both fighters jump at once), where continuity carries nothing; a second score
   re-initialises identity from the truth at every detected cut = what a single continuous camera would face.
B. BODY SIGNATURE: can a fighter be recognised from skeleton proportions alone? Per round: learn red vs blue on the
   first half from limb-length ratios (1 s averages), test on the second half. 50% = no information.
    python boxingweb/identity_baseline.py
"""
import glob, json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
LIMBS = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / max(u, 1e-9)

def track(R, role, reinit_at_cuts):
    kp, sc, box, fr = R["kp"].astype(np.float64), R["score"], R["box"], R["frame"]
    st = None; res = []; swaps = []; cuts = 0; state_wrong = False
    for i in range(len(fr)):
        idx = {int(role[i, p]): p for p in np.flatnonzero(role[i] > 0)}              # truth role -> person
        if not idx:
            continue
        dets = list(idx.items())                                                     # [(role, person)]
        if st is None:
            if len(dets) < 2:
                continue
            st = {r: dict(pose=kp[i, p].copy(), vel=np.zeros(2), t=i) for r, p in dets}; continue
        def cost(tid, p):
            s = st[tid]; dt = min(i - s["t"], 5); pred = s["pose"] + s["vel"] * dt
            ok = sc[i, p] > 0.3; scale = max(np.ptp(kp[i, p][ok], 0).max() if ok.sum() > 2 else 50.0, 20.0)
            return np.linalg.norm(pred[ok] - kp[i, p][ok], axis=1).mean() / scale if ok.any() else 9.0
        C = {(tid, p): cost(tid, p) for tid in (1, 2) for _, p in dets}
        is_cut = all(min(C[(tid, p)] for _, p in dets) > 0.8 for tid in (1, 2))
        if is_cut:
            cuts += 1
            if reinit_at_cuts and len(dets) == 2:
                st = {r: dict(pose=kp[i, p].copy(), vel=np.zeros(2), t=i) for r, p in dets}; state_wrong = False; continue
        if len(dets) == 2:
            (r1, p1), (r2, p2) = dets
            assign = {1: p1, 2: p2} if C[(1, p1)] + C[(2, p2)] <= C[(1, p2)] + C[(2, p1)] else {1: p2, 2: p1}
        else:
            r1, p1 = dets[0]; assign = {1: p1} if C[(1, p1)] <= C[(2, p1)] else {2: p1}
        ov = len(dets) == 2 and iou(box[i, dets[0][1]], box[i, dets[1][1]]) > 0.3
        for tid, p in assign.items():
            true = int(role[i, p]); res.append(tid == true)
            s = st[tid]; dt = max(i - s["t"], 1); c0 = s["pose"][[11, 12]].mean(0); c1 = kp[i, p][[11, 12]].mean(0)
            s["vel"] = 0.5 * s["vel"] + 0.5 * (c1 - c0) / dt; s["pose"] = kp[i, p].copy(); s["t"] = i
        wrong_now = any(tid != int(role[i, p]) for tid, p in assign.items())
        if len(dets) == 2 and wrong_now != state_wrong:
            swaps.append(dict(i=i, overlap=bool(ov), cut=bool(is_cut), gap=False)); state_wrong = wrong_now
    return np.array(res), swaps, cuts

def signature(R, role):
    kp, sc = R["kp"].astype(np.float64), R["score"]; X, y, f = [], [], []
    L = np.stack([np.linalg.norm(kp[:, :, a] - kp[:, :, b], axis=-1) for a, b in LIMBS], -1)        # (F,K,12)
    for r in (1, 2):
        fs, ks = np.nonzero(role == r); l = L[fs, ks]; ok = (sc[fs, ks] > 0.3).mean(1) > 0.8
        l = l / np.maximum(l.sum(1, keepdims=True), 1e-6)
        for a in range(0, len(fs) - 25, 25):                                                        # 1 s averages
            if ok[a:a + 25].mean() > 0.6:
                X.append(np.median(l[a:a + 25][ok[a:a + 25]], 0)); y.append(r); f.append(fs[a])
    return np.array(X), np.array(y), np.array(f)

def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    te = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_test*.npz")) if os.path.basename(p).count(".") == 1)
    out = {}; A = dict(n=0, ok=0, okc=0, nc=0, swaps=0, swaps_c=0, ov=0, cuts=0, sec=0.0); sig = []
    for c in te:
        R = np.load(os.path.join(ROOT, "external", "pose_rtm", c + ".npz")); role = np.load(os.path.join(ROOT, "external", "boxers_gt", c + ".npz"))["role"]
        r0, s0, cuts = track(R, role, False); r1, s1, _ = track(R, role, True)
        X, y, f = signature(R, role); half = np.median(f); tr, tt = f < half, f >= half
        acc = np.nan
        if tr.sum() > 20 and tt.sum() > 20 and len(set(y[tr])) == 2:
            acc = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=1000)).fit(X[tr], y[tr]).score(X[tt], y[tt]); sig.append(acc)
        sec = len(R["frame"]) / float(R["fps"])
        print(f"{c[7:47]:40s} right identity {r0.mean():5.1%} | cuts reset: {r1.mean():5.1%}, {len(s1)} swaps ({sum(s['overlap'] for s in s1)} in overlap) | cuts {cuts:3d} | body signature {acc:.0%}", flush=True)
        A["n"] += len(r0); A["ok"] += r0.sum(); A["nc"] += len(r1); A["okc"] += r1.sum(); A["swaps"] += len(s0); A["swaps_c"] += len(s1)
        A["ov"] += sum(s["overlap"] for s in s1); A["cuts"] += cuts; A["sec"] += sec
        out[c] = dict(acc=float(r0.mean()), acc_cut_reset=float(r1.mean()), swaps=len(s1), cuts=cuts, signature=float(acc))
    print(f"\nALL {len(te)} test rounds, {A['sec'] / 60:.0f} min")
    print(f"  motion continuity only, never corrected: right identity in {A['ok'] / A['n']:.1%} of boxer-frames; {A['cuts']} camera cuts detected ({A['cuts'] / A['sec'] * 60:.1f}/min)")
    print(f"  identity reset at camera cuts (= one continuous camera): right identity {A['okc'] / A['nc']:.1%}; {A['swaps_c']} swaps = {A['swaps_c'] / A['sec'] * 60:.2f} per minute; {A['ov']} of them while the boxers overlap")
    print(f"  body signature (limb proportions, first half -> second half): mean {np.mean(sig):.1%}, min {np.min(sig):.0%}, max {np.max(sig):.0%}  (50% = nothing)")
    json.dump(dict(total={k: float(v) for k, v in A.items()}, rounds=out), open(os.path.join(ROOT, "runs", "identity_baseline.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
