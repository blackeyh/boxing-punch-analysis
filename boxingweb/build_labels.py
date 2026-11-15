"""BoxingWeb -> training labels for the boxer selector and for identity.

Each round ships {round}_pose_gt.pkl: for EVERY video frame, the red boxer's and the blue boxer's pose
(45 points, 2D normalised by width/height, plus 3D). Loaded with a restricted unpickler (boxingweb/safeload.py:
only numpy arrays allowed; the files reference nothing else).
Here every RTMPose person in every frame gets a role: 1 = red boxer, 2 = blue boxer, 0 = someone else
(referee, crowd, cornermen), by matching RTMPose torsos to the two given poses (Hungarian, gated).
Output: external/boxers_gt/{id}.npz  role (F,K) int8 (-1 = no person), cost (F,2); boxingweb/events.json; summary.
    python boxingweb/build_labels.py "<boxingweb root>"
"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); import safeload
GT_TORSO, RTM_TORSO = [34, 33, 12, 9], [5, 6, 11, 12]        # same body points in the two layouts (measured)
GATE = 0.9                                                    # mean torso-point distance / RTMPose torso length

def main(root):
    out = os.path.join(ROOT, "external", "boxers_gt"); os.makedirs(out, exist_ok=True)
    events, summ = [], {}
    for d in sorted(glob.glob(os.path.join(root, "data_*", "*"))):
        split = os.path.basename(os.path.dirname(d)).replace("data_", ""); cid = f"W_{split}_{os.path.basename(d)}"
        R = np.load(os.path.join(ROOT, "external", "pose_rtm", cid + ".npz")); W, H = float(R["width"]), float(R["height"])
        G = safeload.load(glob.glob(os.path.join(d, "*.pkl"))[0])
        fr, n, kp = R["frame"], R["n"].astype(int), R["kp"]
        gt = np.stack([G["pose_red_2d"], G["pose_blue_2d"]], 1).astype(np.float64) * [W, H]      # (T,2,45,2)
        role = np.full(n.shape + (kp.shape[1],), -1, np.int8); cost = np.full((len(fr), 2), np.nan, np.float32)
        for i, t in enumerate(fr):
            m = n[i]
            if m == 0 or t >= len(gt):
                continue
            role[i, :m] = 0
            p = kp[i, :m][:, RTM_TORSO]; g = gt[t][:, GT_TORSO]
            scale = np.maximum(np.linalg.norm(p[:, :2].mean(1) - p[:, 2:].mean(1), axis=1), 10.0)
            C = np.linalg.norm(p[None] - g[:, None], axis=-1).mean(-1) / scale[None]              # (2, m)
            r, c = linear_sum_assignment(C)
            for a, b in zip(r, c):
                cost[i, a] = C[a, b]
                if C[a, b] < GATE:
                    role[i, b] = a + 1
        np.savez_compressed(os.path.join(out, cid + ".npz"), frame=fr, role=role, cost=cost)
        ev = json.load(open(os.path.join(d, "video_event.json"), encoding="utf-8"))
        side = next((e for e in ev if e["name"] == "set_side"), None)
        k = 0
        for e in ev:
            if e["name"] != "punching":
                continue
            events.append(dict(video=cid, split=split, start=int(e["frame_begin"]), end=int(e["frame_end"]), side=e["side"],
                               technique=e["technique"], hand="left" if e["technique"][0] == "l" else "right",
                               target=e["target"], landed=e["effect"] == "effective", distance=e["distance"], foul=bool(e["foul"])))
            k += 1
        has = role >= 0; both = ((role == 1).any(1) & (role == 2).any(1)); one = ((role == 1).any(1) ^ (role == 2).any(1))
        summ[cid] = dict(frames=int(len(fr)), fps=float(R["fps"]), gt_frames=int(len(gt)), punches=k, both=float(both.mean()), one=float(one.mean()),
                         others_per_frame=float((role == 0).sum() / len(fr)), left=side["left"] if side else None)
    json.dump(events, open(os.path.join(HERE, "events.json"), "w"))
    json.dump(summ, open(os.path.join(HERE, "labels_summary.json"), "w"), indent=1)
    F = sum(s["frames"] for s in summ.values()); w = lambda k: sum(s[k] * s["frames"] for s in summ.values()) / F
    print(f"{len(summ)} rounds, {F} frames, {len(events)} punches")
    print(f"both boxers matched to an RTMPose person: {w('both'):.1%} of frames | only one: {w('one'):.1%} | neither: {1 - w('both') - w('one'):.1%}")
    print(f"other people per frame (referee, crowd): {w('others_per_frame'):.1f}")
    worst = sorted(summ.items(), key=lambda kv: kv[1]["both"])[:5]
    print("lowest 'both matched' rounds:", [(k[2:40], round(v["both"], 2)) for k, v in worst])
    print("frame count mismatches (video frames covered vs pose file):", sum(abs(s["gt_frames"] - s["frames"] * round(60 / 30 if s["fps"] < 31 and False else 1)) > 5 for s in summ.values()))

if __name__ == "__main__":
    main(sys.argv[1])
