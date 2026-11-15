"""Bag route, feasibility test 1: numbers and contact sheets from the OWLv2 detections (no torch here).

    python bagroute/analyse.py            -> runs/bagroute/summary.json, summary.txt, {id}_target.jpg
Reads runs/bagroute/{id}_det.json, external/pose/{id}.npz (MediaPipe, B1/C1/C2 3-minute test segments),
external/pose_rtm/{id}.npz (RTMPose, A1/A2).
"""
import json, os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "bagroute")
CLIPS = ["B1_bob_miri", "C1_heavybag_miri", "C2_standingbag_gosia", "A1_sparring_gym_dark", "A2_sparring_ring"]
BAG, DUMMY, PERSON = [0, 1, 2], [3, 4, 5], [6]
GLOVE = 0.5         # glove point = wrist + GLOVE * (wrist - elbow)
THR = 0.20          # "detected" threshold for the summaries (0.30 also reported)
L = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); L.append(s)

def iou(a, b):
    a = np.asarray(a, float); b = np.atleast_2d(np.asarray(b, float))
    iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)

def nms(dets, thr=0.6):
    keep = []
    for d in sorted(dets, key=lambda d: -d["s"]):
        if not keep or iou(d["b"], [k["b"] for k in keep]).max() < thr:
            keep.append(d)
    return keep

def group(fr, prompts, thr=0.0):
    return nms([d for d in fr["dets"] if d["p"] in prompts and d["s"] >= thr])

def sites(frames, prompts, thr=THR, link=0.5):
    """Greedy clustering of detections over time into fixed places (a static object = one site)."""
    S = []
    for i, fr in enumerate(frames):
        for d in group(fr, prompts, thr):
            best, bi = 0, None
            for k, s in enumerate(S):
                v = iou(d["b"], s["box"])[0]
                if v > best:
                    best, bi = v, k
            if bi is not None and best >= link:
                s = S[bi]
                if s["idx"][-1] == i:
                    continue
                s["idx"].append(i); s["boxes"].append(d["b"]); s["sc"].append(d["s"]); s["pr"].append(d["p"])
                s["box"] = np.median(np.array(s["boxes"][-15:]), 0)
            else:
                S.append({"idx": [i], "boxes": [d["b"]], "sc": [d["s"]], "pr": [d["p"]], "box": np.array(d["b"])})
    return S

def load_wrists(cid):
    """-> dict frame -> list of persons, each (box xyxy, [wrist xy ...])"""
    out = {}
    p_rtm = os.path.join(ROOT, "external", "pose_rtm", cid + ".npz"); p_mp = os.path.join(ROOT, "external", "pose", cid + ".npz")
    if os.path.exists(p_rtm):
        d = np.load(p_rtm); kp, sc, bx, fr, n = d["kp"], d["score"], d["box"], d["frame"], d["n"]
        for i in range(len(fr)):
            ps = []
            for j in range(int(n[i])):
                wr = [kp[i, j, w] + GLOVE * (kp[i, j, w] - kp[i, j, e]) for e, w in ((7, 9), (8, 10)) if sc[i, j, w] > 0.3 and sc[i, j, e] > 0.3]
                ps.append((bx[i, j], wr, float(np.mean(sc[i, j]))))
            out[int(fr[i])] = ps
        return out, "rtmpose", float(d["fps"])
    if os.path.exists(p_mp):
        d = np.load(p_mp); im, fr, n = d["img"], d["frame"], d["npose"]; W, H = int(d["width"]), int(d["height"])
        for i in range(len(fr)):
            ps = []
            for j in range(int(n[i])):
                xy = im[i, j, :, :2] * (W, H); vis = im[i, j, :, 3]
                ok = vis > 0.5
                if ok.sum() < 4:
                    continue
                box = np.r_[xy[ok].min(0), xy[ok].max(0)]
                ps.append((box, [xy[w] + GLOVE * (xy[w] - xy[e]) for e, w in ((13, 15), (14, 16)) if vis[w] > 0.5 and vis[e] > 0.5], float(vis.mean())))
            out[int(fr[i])] = ps
        return out, "mediapipe", float(d["fps"])
    return None, None, None

def gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0); dy = max(b[1] - a[3], a[1] - b[3], 0)
    return float(np.hypot(dx, dy))

def reach(box, pose, fps_src, nbig=2, margin=0.10, frames=None):
    """Share of pose frames in which a wrist of one of the nbig largest people is inside the (10%-grown) box."""
    w, h = box[2] - box[0], box[3] - box[1]
    g = [box[0] - margin * w, box[1] - margin * h, box[2] + margin * w, box[3] + margin * h]
    hit = tot = 0; secs = {}
    for f, ps in pose.items():
        if frames is not None and not (frames[0] <= f < frames[1]):
            continue
        ps = sorted(ps, key=lambda p: -(p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))[:nbig]
        # a person whose own box IS this box (the dummy seen as a person, or a boxer called a dummy) does not count
        wr = [w_ for p in ps if iou(box, p[0])[0] < 0.5 for w_ in p[1]]
        if not ps:
            continue
        tot += 1
        h_ = any(g[0] <= x <= g[2] and g[1] <= y <= g[3] for x, y in wr)
        hit += h_; k = int(f / fps_src); secs[k] = secs.get(k, False) or h_
    return hit / max(tot, 1), tot, float(np.mean(list(secs.values()))) if secs else 0.0

def sheet(cid, D, S_all, path, n=12):
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", cid + ".mp4"))
    fr = D["frames"]; pick = [fr[int(i)] for i in np.linspace(0, len(fr) - 1, n)]
    tiles = []
    for f in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame"]); ok, im = cap.read()
        if not ok:
            continue
        for prompts, col in ((PERSON, (255, 255, 255)), (BAG, (0, 200, 255)), (DUMMY, (255, 0, 255))):
            ds = group(f, prompts, THR)
            for k, d in enumerate(ds):
                x0, y0, x1, y1 = [int(v) for v in d["b"]]
                top = k == 0 and prompts is not PERSON
                cv2.rectangle(im, (x0, y0), (x1, y1), col, 5 if top else 2)
                cv2.putText(im, f"{D['prompts'][d['p']][2:14]} {d['s']:.2f}", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7 * im.shape[1] / 1280, col, 2)
        cv2.putText(im, f"{cid[:2]} t={f['t']:.0f}s", (10, im.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * im.shape[1] / 1280, (0, 255, 0), 3)
        tiles.append(cv2.resize(im, (640, int(640 * im.shape[0] / im.shape[1]))))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

def main():
    man = json.load(open(os.path.join(ROOT, "external", "manifest.json")))
    summ = {}
    for cid in CLIPS:
        p = os.path.join(OUT, cid + "_det.json")
        if not os.path.exists(p):
            say(f"[{cid}] no detections yet"); continue
        D = json.load(open(p)); fr = D["frames"]; P = D["prompts"]; W, H = D["width"], D["height"]; diag = float(np.hypot(W, H))
        seg = man[cid].get("segment_frames")
        say(f"\n===== {cid}  {W}x{H}, {len(fr)} sampled frames (every {D['every_s']} s, whole file {fr[-1]['t']:.0f} s), "
            f"{D['sec_per_frame']:.2f} s/frame on {D['device']}; test segment frames {seg}")
        R = {"n_frames": len(fr), "sec_per_frame": D["sec_per_frame"], "prompts": {}}
        # --- per prompt: is something found, is the top box stable
        say("prompt                          | >=0.20 | >=0.30 | med top score | med IoU(consecutive top box) | centre step med / p90 px")
        for pi, name in enumerate(P):
            top = [max((d for d in f["dets"] if d["p"] == pi), key=lambda d: d["s"], default=None) for f in fr]
            sc = np.array([t["s"] if t else 0 for t in top])
            ious, steps = [], []
            for a, b in zip(top[:-1], top[1:]):
                if a and b and a["s"] >= THR and b["s"] >= THR:
                    ious.append(iou(a["b"], b["b"])[0])
                    steps.append(np.hypot((a["b"][0] + a["b"][2] - b["b"][0] - b["b"][2]) / 2, (a["b"][1] + a["b"][3] - b["b"][1] - b["b"][3]) / 2))
            r = dict(share20=float((sc >= .2).mean()), share30=float((sc >= .3).mean()), med_score=float(np.median(sc)),
                     med_iou=float(np.median(ious)) if ious else None, step_med=float(np.median(steps)) if steps else None,
                     step_p90=float(np.percentile(steps, 90)) if steps else None)
            R["prompts"][name] = r
            say(f"{name:31s} | {r['share20']:6.1%} | {r['share30']:6.1%} | {r['med_score']:13.2f} | "
                + (f"{r['med_iou']:28.2f} | {r['step_med']:.0f} / {r['step_p90']:.0f}" if ious else "   -"))
        # --- fixed places (sites): static objects that keep being detected
        pose, kind, pfps = load_wrists(cid)
        R["pose"] = kind
        fr_all = fr
        if pose:
            lo, hi = min(pose), max(pose); fr = [f for f in fr_all if lo <= f["frame"] <= hi]
        say(f"   (places, wrist reach and people counts below: {len(fr)} sampled frames, t = {fr[0]['t']:.0f}-{fr[-1]['t']:.0f} s"
            + (f", the range that has {kind} skeletons)" if pose else ")"))
        if pose:
            boxer_h = float(np.median([max(p[0][3] - p[0][1] for p in ps) for ps in pose.values() if ps]))
        for gname, prompts in (("bag prompts", BAG), ("dummy prompts", DUMMY)):
            S = [s for s in sites(fr, prompts) if len(s["idx"]) >= 0.2 * len(fr)]
            S.sort(key=lambda s: -len(s["idx"]))
            say(f"-- persistent places for {gname} (same place, IoU>=0.5, in >=20% of sampled frames, score>={THR}): {len(S)}")
            R[gname] = []
            for s in S[:8]:
                B = np.array(s["boxes"]); med = np.median(B, 0); c = np.c_[(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2]
                drift = float(np.median(np.hypot(*(c - np.median(c, 0)).T))); dmax = float(np.percentile(np.hypot(*(c - np.median(c, 0)).T), 95))
                best = max(set(s["pr"]), key=s["pr"].count)
                # longest run of consecutive samples
                idx = np.array(s["idx"]); runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1); longest = max(len(r) for r in runs) * D["every_s"]
                e = dict(box=[round(float(v)) for v in med], present=len(idx) / len(fr), med_score=float(np.median(s["sc"])), prompt=P[best],
                         centre_dev_med_px=drift, centre_dev_p95_px=dmax, longest_run_s=longest, area_frac=float((med[2] - med[0]) * (med[3] - med[1]) / (W * H)))
                msg = (f"   box {e['box']} ({e['area_frac']:.1%} of frame) present {e['present']:.0%}, longest unbroken {longest:.0f} s, med score {e['med_score']:.2f}, "
                       f"most often '{P[best]}', centre deviation med {drift:.0f} px / p95 {dmax:.0f} px")
                if pose:
                    rr, tot, rs = reach(med, pose, D["fps"]); e["glove_reach_frames"] = rr; e["pose_frames"] = tot; e["glove_reach_seconds"] = rs
                    # is this place also a "person" for the pose model?
                    e["height_vs_boxer"] = float((med[3] - med[1]) / boxer_h)
                    msg += f" | height = {e['height_vs_boxer']:.2f} x median boxer height"
                    ov = [max([iou(med, p[0])[0] for p in ps], default=0) for ps in pose.values()]
                    e["pose_person_overlap_share"] = float(np.mean(np.array(ov) >= 0.5))
                    msg += f" | GLOVE REACH: glove point of one of the 2 largest people inside the box in {rs:.0%} of seconds ({rr:.1%} of {tot} pose frames) | a pose-model person coincides (IoU>=0.5) in {e['pose_person_overlap_share']:.1%}"
                po = [max([iou(med, d["b"])[0] for d in group(f, PERSON, THR)], default=0) for f in fr]
                e["owl_person_overlap_share"] = float(np.mean(np.array(po) >= 0.5))
                msg += f" | OWLv2 'a person' box coincides in {e['owl_person_overlap_share']:.0%}"
                say(msg); R[gname].append(e)
        # --- moving detections: top target box vs the boxers (sparring false alarms)
        if pose and kind == "rtmpose":
            for gname, prompts in (("bag prompts", BAG), ("dummy prompts", DUMMY)):
                cat = {"is a boxer (IoU>=0.5 with a person box)": 0, "touching/overlapping a boxer": 0, "near (<0.5 boxer heights)": 0, "far": 0, "no pose": 0}
                sc_by = {k: [] for k in cat}; nd = 0
                for f in fr:
                    ds = group(f, prompts, THR)
                    if not ds:
                        continue
                    nd += 1; d = ds[0]; ps = sorted(pose.get(f["frame"], []), key=lambda p: -(p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))[:2]
                    if not ps:
                        k = "no pose"
                    else:
                        io = max(iou(d["b"], p[0])[0] for p in ps); g = min(gap(d["b"], p[0]) / max(p[0][3] - p[0][1], 1) for p in ps)
                        k = list(cat)[0] if io >= 0.5 else list(cat)[1] if g == 0 else list(cat)[2] if g < 0.5 else list(cat)[3]
                    cat[k] += 1; sc_by[k].append(d["s"])
                say(f"-- top '{gname}' detection (score>={THR}) in {nd}/{len(fr)} frames; where it is relative to the 2 largest RTMPose people:")
                for k in cat:
                    if cat[k]:
                        say(f"     {k}: {cat[k]} frames ({cat[k] / len(fr):.0%} of all), score med {np.median(sc_by[k]):.2f} max {np.max(sc_by[k]):.2f}")
                R[gname + " vs boxers"] = {k: [cat[k], float(np.median(sc_by[k])) if sc_by[k] else None] for k in cat}
        # --- people
        for pt in (0.10, THR):
            npers = np.array([len(group(f, PERSON, pt)) for f in fr])
            R[f"owl_person_count_share_{pt}"] = {str(k): float((npers == k).mean()) for k in (0, 1, 2)} | {"3+": float((npers >= 3).mean())}
            say(f"-- OWLv2 'a person' boxes (score>={pt}, NMS) per sampled frame: " + ", ".join(f"{k}: {v:.0%}" for k, v in R[f"owl_person_count_share_{pt}"].items()))
        if pose:
            n = np.array([len(v) for v in pose.values()])
            R["pose_count_share"] = {str(k): float((n == k).mean()) for k in (0, 1, 2)} | {"3+": float((n >= 3).mean())}
            say(f"-- {kind} people per frame ({len(n)} frames): " + ", ".join(f"{k}: {v:.1%}" for k, v in R["pose_count_share"].items()))
        fr = fr_all
        sheet(cid, D, None, os.path.join(OUT, cid + "_target.jpg"))
        summ[cid] = R
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "summary.txt"), "w").write("\n".join(L) + "\n")

if __name__ == "__main__":
    main()
