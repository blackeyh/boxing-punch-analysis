"""Referee contact clips with the puncher chosen CORRECTLY -> hpc_rtm/lists/landed_clips_p.json (same punches, same
order as landed_clips.json).
The referee's box sits on the TARGET (the head or body of the boxer being hit). landed_clips.py took "the boxer whose
labelled-hand glove is nearest the box" as the puncher; a defender's glove in guard is next to his own head, so that
picked the defender for about a third of the punches (found 2026-11-15 on contact sheets; with the rule below each
fighter's share of the punches per bout correlates 0.89 with the detector's counts, with the old rule 0.23).
Rule: target = the boxer whose head (torso for body shots) is nearest the box; puncher = the other boxer; crop centre =
the puncher's labelled-hand glove at the contact.
    python olympic/landed_clips_p.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import landed_cues as lc

def puncher(kps, box_c, outcome):
    """kps: list of (17, 2) skeletons (the two boxers); returns the index of the puncher."""
    if len(kps) < 2:
        return 0
    part = [5, 6, 11, 12] if outcome == "body" else [0, 1, 2, 3, 4]
    return int(np.argmax([np.linalg.norm(np.nanmean(k[part], 0) - box_c) for k in kps]))

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); out = []; old = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips.json")))
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "pose_rtm", "O_*.npz"))):
        if os.path.basename(f).count(".") != 1:
            continue
        cid = os.path.basename(f)[:-4]; R = np.load(f); pick = np.load(os.path.join(ROOT, "external", "boxers_rtm", cid + ".npz"))["pick"]; kp, fr = R["kp"].astype(np.float64), R["frame"]
        for e in [e for e in ev if "O_" + e["video"] == cid]:
            if lc.cues(R, pick, e) is None:
                continue
            i = int(np.clip(np.searchsorted(fr, (e["start"] + e["end"]) // 2), 0, len(fr) - 1)); ps = [p for p in pick[i] if p >= 0]; w = 9 if e["hand"] == "left" else 10
            b = np.array(e["boxes"]); c = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2])
            p = ps[puncher([kp[i, q] for q in ps], c, e["outcome"])]; g = kp[i, p, w] + 0.4 * (kp[i, p, w] - kp[i, p, w - 2])
            S = max(np.linalg.norm(kp[i, p, [5, 6]].mean(0) - kp[i, p, [11, 12]].mean(0)), 20.0)
            out.append(dict(id=cid, frame=int((e["start"] + e["end"]) // 2), cx=float(g[0]), cy=float(g[1]), side=float(2.4 * S), bx=float(c[0]), by=float(c[1]),
                            y={"head": 0, "body": 0, "block": 1, "miss": 2}[e["outcome"]], outcome=e["outcome"], hand=e["hand"], gym=cid[2:6]))
    assert len(out) == len(old) and all(a["id"] == b["id"] and a["frame"] == b["frame"] for a, b in zip(out, old))
    json.dump(out, open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips_p.json"), "w"))
    mv = np.array([np.hypot(a["cx"] - b["cx"], a["cy"] - b["cy"]) / b["side"] for a, b in zip(out, old)]); d = np.array([np.hypot(o["cx"] - o["bx"], o["cy"] - o["by"]) / o["side"] for o in out])
    print(f"{len(out)} clips; crop centre changed (another boxer's glove) for {np.mean(mv > 0.02):.1%}, by outcome: " + ", ".join(f"{k} {np.mean(mv[[o['outcome'] == k for o in out]] > 0.02):.0%}" for k in ("head", "body", "block", "miss")))
    print(f"glove-to-referee-box distance / crop side: median {np.median(d):.2f}, 90th pct {np.percentile(d, 90):.2f}; box centre inside the crop for {np.mean(d < 0.5):.1%}")

if __name__ == "__main__":
    main()
