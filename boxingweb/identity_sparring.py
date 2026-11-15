"""Run the identity pipeline on a clip WITHOUT truth (A1 / A2 sparring) and render it for judging by eye.

selector (skeleton-only, same-size pair rule) -> apart stretches -> OSNet-AIN embeddings -> stretch 0 defines A and B,
later stretches labelled by a self-updating A-vs-B classifier -> identity carried by motion through overlaps.
Outputs runs/identity_sparring/{id}_sheet.jpg (24 evenly spaced stretch frames), {id}_clip.mp4, {id}.json.
A = orange, B = cyan. Everything else is left undrawn.
    python boxingweb/identity_sparring.py ID [clip_start_s clip_seconds]
"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "olympic"))
import mem_guard  # noqa: F401
import cv2, joblib, numpy as np
import feats, identity_episodes as ie, identity_reid2 as ir2, train_selector as ts
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
COL = {1: (0, 140, 255), 0: (255, 220, 0)}

def main(cid, t0=None, secs=30):
    from sklearn.linear_model import LogisticRegression
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); kp, sc = R["kp"].astype(np.float64), R["score"]; F = len(R["frame"]); fps = float(R["fps"])
    D = feats.load(cid); gbm = joblib.load(os.path.join(ROOT, "runs", "selector_learned", "gbm_boxingweb_train.joblib"))
    m = D["tid"] >= 0; P = np.full(m.shape, np.nan); P[m] = gbm.predict_proba(D["X2"][m])[:, 1]
    pick = ts.pick_pair(D, ts.smooth(D, np.nan_to_num(P)))
    pseudo = np.zeros(m.shape, np.int8)
    for i, pk in enumerate(pick):
        for p in pk:
            if p >= 0:
                pseudo[i, p] = 1
    rd = dict(id=cid, segs=ie.segments(R, pseudo, None)); segs = rd["segs"]; crops = ir2.crops_for_round(rd)
    emb = ir2.make_models()["OSNet-AIN (re-identification)"]; E = emb(crops, False); per = {}
    for (si, t, *_), e in zip(crops, E):
        per.setdefault((si, t), []).append(e)
    per = {k: np.array(v) for k, v in per.items()}
    isA0 = {0: True}; Xs, ys = [per[(0, 0)], per[(0, 1)]], [np.ones(len(per[(0, 0)])), np.zeros(len(per[(0, 1)]))]; clf = None; margins = []
    gA, gB = per[(0, 0)].mean(0), per[(0, 1)].mean(0)
    for si in range(1, len(segs)):
        if (si, 0) not in per or (si, 1) not in per:
            isA0[si] = isA0[si - 1]; continue
        a, b = per[(si, 0)], per[(si, 1)]
        d = ((a.mean(0) @ gA + b.mean(0) @ gB) - (a.mean(0) @ gB + b.mean(0) @ gA)) if clf is None else (clf.decision_function(a).mean() - clf.decision_function(b).mean())
        if clf is not None:
            margins.append(abs(d))
        isA0[si] = bool(d > 0)
        Xs += [a, b]; ys += [np.full(len(a), float(isA0[si])), np.full(len(b), float(not isA0[si]))]
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack(Xs), np.concatenate(ys))
    ident = np.full((F, kp.shape[1]), -1, np.int8); in_seg = np.zeros(F, bool)
    for si, s in enumerate(segs):
        for i, p0, p1 in zip(s["frames"], s["tr"][0], s["tr"][1]):
            ident[i, p0] = int(isA0[si]); ident[i, p1] = int(not isA0[si]); in_seg[i] = True
    st = None
    for i in range(F):
        ps = [p for p in pick[i] if p >= 0]
        if in_seg[i]:
            st = {int(ident[i, p]): (kp[i, p].copy(), sc[i, p].copy()) for p in ps}; continue
        if st is None or len(st) < 2 or not ps:
            continue
        C = {(a, p): ie.pdist(st[a][0], st[a][1], kp[i, p], sc[i, p]) for a in (0, 1) for p in ps}
        asg = ({1: ps[0], 0: ps[1]} if C[(1, ps[0])] + C[(0, ps[1])] <= C[(1, ps[1])] + C[(0, ps[0])] else {1: ps[1], 0: ps[0]}) if len(ps) == 2 else {1 if C[(1, ps[0])] <= C[(0, ps[0])] else 0: ps[0]}
        for a, p in asg.items():
            ident[i, p] = a; st[a] = (kp[i, p].copy(), sc[i, p].copy())
    out = os.path.join(ROOT, "runs", "identity_sparring"); os.makedirs(out, exist_ok=True)
    two = (pick >= 0).sum(1)
    info = dict(frames=F, seconds=F / fps, two_selected=float((two == 2).mean()), stretches=len(segs), stretch_frames=float(in_seg.mean()),
                frames_with_identity=float((ident >= 0).any(1).mean()), swaps_between_stretches=int(sum(isA0[k] != isA0[k - 1] for k in range(1, len(segs)))),
                margin_median=float(np.median(margins)) if margins else None, low_margin_share=float(np.mean(np.array(margins) < 1.0)) if margins else None)
    json.dump(info, open(os.path.join(out, cid + ".json"), "w"), indent=1); print(cid, info)
    np.savez_compressed(os.path.join(out, cid + ".npz"), ident=ident, pick=pick, in_seg=in_seg)
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4"))
    def draw(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(R["frame"][i])); ok, im = cap.read()
        for p in np.flatnonzero(ident[i] >= 0):
            col = COL[int(ident[i, p])]; xy = kp[i, p]
            for a, b in EDGES:
                if sc[i, p, a] > .3 and sc[i, p, b] > .3:
                    cv2.line(im, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), col, 4)
            x, y = np.nanmin(xy, 0).astype(int); cv2.putText(im, "A" if ident[i, p] else "B", (x, y - 8), 0, 1.6, col, 4)
        cv2.putText(im, f"{R['frame'][i] / float(R['fps_source']):.1f}s {'apart' if in_seg[i] else 'overlap'}", (15, 40), 0, 1.1, (255, 255, 255), 2)
        return im
    idx = np.flatnonzero(in_seg); sel = idx[np.linspace(0, len(idx) - 1, 24).astype(int)]
    tiles = [cv2.resize(draw(i), (640, 360)) for i in sel]
    cv2.imwrite(os.path.join(out, cid + "_sheet.jpg"), np.vstack([np.hstack(tiles[k:k + 4]) for k in range(0, 24, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 78])
    i0 = int((t0 if t0 is not None else 60) * fps); vw = cv2.VideoWriter(os.path.join(out, cid + "_clip.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, (1280, 720))
    for i in range(i0, min(i0 + int(secs * fps), F)):
        vw.write(cv2.resize(draw(i), (1280, 720)))
    vw.release()

if __name__ == "__main__":
    a = sys.argv[1:]; main(a[0], float(a[1]) if len(a) > 1 else None, float(a[2]) if len(a) > 2 else 30)
