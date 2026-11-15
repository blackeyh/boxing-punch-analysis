""""WHICH PUNCH" lists: the same clips as the glove-centred lists, plus the puncher's UPPER ARM (shoulder -> elbow of
the punching hand) for each of the 44 clip frames, to be drawn into the clip as a thin line (hpc_rtm/extract_clips3.py).
Why: in exchanges the model cannot know which of two near-simultaneous punches is meant (right on 59% when another
punch with a different outcome is within 0.16 s, 74% overall). The upper arm marks the puncher and the hand without
covering the glove or the contact area.
  hpc_rtm/lists/landed_clips_arm.json  referee punches: puncher = picked boxer whose labelled-hand glove is nearest
                                       the referee's box; followed frame to frame by the torso centre
  hpc_rtm/lists/chain_clips_v5.json    the chain's punches: the counted boxer's stream and the detector's hand
Each clip gets "arm": 44 x [sx, sy, ex, ey] in video pixels (null where the arm is not seen).
    python olympic/landed_clips_arm.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import landed_cues as lc, landed_clips_p as lcp
T, PRE = 44, 22; TORSO = [5, 6, 11, 12]

def interp(frames, pts, f0):
    """frames (n,) video frame numbers with pts (n,4) (nan = unseen) -> list of T entries for video frames f0..f0+T-1."""
    out = []
    for f in range(f0, f0 + T):
        j = int(np.searchsorted(frames, f))
        if j < len(frames) and frames[j] == f:
            v = pts[j]
        elif 0 < j < len(frames):
            w = (f - frames[j - 1]) / (frames[j] - frames[j - 1]); v = pts[j - 1] * (1 - w) + pts[j] * w
        else:
            v = np.full(4, np.nan)
        out.append([round(float(x), 1) for x in v] if np.isfinite(v).all() else None)
    return out

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); P = "--p" in sys.argv; old = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips_p.json" if P else "landed_clips.json"))); out = []; n = 0        # --p: corrected puncher rule (landed_clips_p.py)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "pose_rtm", "O_*.npz"))):
        if os.path.basename(f).count(".") != 1:
            continue
        cid = os.path.basename(f)[:-4]; R = np.load(f); pick = np.load(os.path.join(ROOT, "external", "boxers_rtm", cid + ".npz"))["pick"]; kp, sc, fr = R["kp"].astype(np.float64), R["score"], R["frame"]; F = len(fr)
        for e in [e for e in ev if "O_" + e["video"] == cid]:
            if lc.cues(R, pick, e) is None:
                continue
            o = old[n]; n += 1; assert o["id"] == cid and o["frame"] == (e["start"] + e["end"]) // 2
            i = int(np.clip(np.searchsorted(fr, o["frame"]), 0, F - 1)); ps = [p for p in pick[i] if p >= 0]; w = 9 if e["hand"] == "left" else 10
            g = [kp[i, p, w] + 0.4 * (kp[i, p, w] - kp[i, p, w - 2]) for p in ps]; a = ps[lcp.puncher([kp[i, q] for q in ps], np.array([o["bx"], o["by"]]), e["outcome"])] if P else ps[int(np.argmin([np.linalg.norm(x - np.array([o["bx"], o["by"]])) for x in g]))]
            idx = list(range(max(i - 13, 0), min(i + 14, F))); pts = np.full((len(idx), 4), np.nan); c0 = idx.index(i)
            def put(k, p):
                if sc[idx[k], p, w - 4] > 0.3 and sc[idx[k], p, w - 2] > 0.3:
                    pts[k] = np.r_[kp[idx[k], p, w - 4], kp[idx[k], p, w - 2]]
            put(c0, a)
            for rng in (range(c0 + 1, len(idx)), range(c0 - 1, -1, -1)):
                last = kp[i, a][TORSO].mean(0)                                             # follow the PERSON (torso centre), not the wrist: wrists of two boxers cross in exchanges
                for k in rng:
                    cand = [p for p in pick[idx[k]] if p >= 0]
                    if not cand:
                        break
                    p = min(cand, key=lambda p: np.linalg.norm(kp[idx[k], p][TORSO].mean(0) - last))
                    if np.linalg.norm(kp[idx[k], p][TORSO].mean(0) - last) > 0.25 * o["side"]:
                        break
                    put(k, p); last = kp[idx[k], p][TORSO].mean(0)
            d = dict(o); d["arm"] = interp(fr[idx], pts, o["frame"] - PRE); out.append(d)
    assert len(out) == len(old); json.dump(out, open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips_armp.json" if P else "landed_clips_arm.json"), "w"))
    cov = np.mean([np.mean([x is not None for x in d["arm"]]) for d in out]); ctr = np.mean([d["arm"][PRE] is not None for d in out])
    print(f"{len(out)} referee clips: upper arm known in {cov:.1%} of clip frames, at the contact frame in {ctr:.1%} of clips")
    if P:
        return
    v3 = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); cache = {}; v5 = []
    for c in v3:
        if c["id"] not in cache:
            z = np.load(os.path.join(ROOT, "external", "chain_streams", c["id"] + ".npz")); cache[c["id"]] = (z["K"].astype(np.float64), z["frame"])
        K, fr = cache[c["id"]]; step = int(fr[1] - fr[0]); t = int(np.clip(round(c["frame"] / step), 0, len(fr) - 1)); s = "AB".index(c["boxer"]); w = 9 if c["hand"] == "left" else 10
        a, b = max(t - 13, 0), min(t + 14, len(fr)); pts = np.concatenate([K[s, a:b, w - 4], K[s, a:b, w - 2]], 1); d = dict(c); d["arm"] = interp(fr[a:b], pts, c["frame"] - PRE); v5.append(d)
    json.dump(v5, open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v5.json"), "w"))
    print(f"{len(v5)} chain clips: upper arm known in {np.mean([np.mean([x is not None for x in d['arm']]) for d in v5]):.1%} of clip frames")

if __name__ == "__main__":
    main()
