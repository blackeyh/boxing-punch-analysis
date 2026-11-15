"""Both-gloves scan: for every counted punch the pixel model scans the detector's hand AND the other hand; the hand
whose best "centred on a contact" score is higher wins (optionally only if it wins by a margin).
Inputs: cl_large_kam*.npz (detector's hand) and scan_v3b.npz (other hand), hpc_rtm/lists/chain_clips_v3.json.
    python olympic/fullchain_scan_hands.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MAP = {"head": 0, "body": 0, "block": 1, "miss": 2, None: -1}

def main():
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); ev = json.load(open(os.path.join(HERE, "events.json"))); N = len(clips)
    S1 = np.full((N, 13, 4), np.nan)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", "cl_large_kam*.npz"))):
        z = np.load(f); S1[z["chain_idx"]] = z["chain_scan"]; offs = z["scan_offsets"]
    S2 = np.load(os.path.join(ROOT, "external", "landed_ft", "scan_v3b.npz"))["scan"]
    lab = np.array([MAP[c["label"]] for c in clips]); n_landed = sum(e["outcome"] in ("head", "body") for e in ev)
    by = {}
    for e in ev:
        by.setdefault("O_" + e["video"], []).append(e)
    ref = {}
    for n, c in enumerate(clips):
        if c["label"] is None:
            continue
        cand = [e for e in by[c["id"]] if c["f0"] - 10 <= (e["start"] + e["end"]) / 2 <= c["f1"] + 10 and e["outcome"] == c["label"] and e["hand"] == c["label_hand"]]
        if cand:
            e = min(cand, key=lambda e: abs((e["start"] + e["end"]) / 2 - (c["f0"] + c["f1"]) / 2)); ref[n] = ((e["start"] + e["end"]) / 2, e["hand"])
    c1, c2 = 1 - S1[:, :, 3], 1 - S2[:, :, 3]; k1, k2 = c1.argmax(1), c2.argmax(1); m1, m2 = c1.max(1), c2.max(1)
    det_hand = np.array([c["hand"] for c in clips]); other = np.where(det_hand == "left", "right", "left")
    print(f"{N} counted punches; detector's hand has the higher contact score in {np.mean(m1 >= m2):.1%}")
    for margin in (None, 0.0, 0.1, 0.3):
        use2 = np.zeros(N, bool) if margin is None else (m2 > m1 + margin)
        k = np.where(use2, k2, k1); S = np.where(use2[:, None, None], S2, S1); p3 = S[np.arange(N), k, :3]; p3 = p3 / p3.sum(1, keepdims=True); hand = np.where(use2, other, det_hand); d = offs[k]
        idx = np.array(sorted(ref)); dt = np.array([clips[n]["coarse"] + d[n] - ref[n][0] for n in idx]); hok = np.array([hand[n] == ref[n][1] for n in idx])
        m = lab >= 0; yh = p3[m].argmax(1); y = lab[m]; cm = np.zeros((3, 3), int); np.add.at(cm, (y, yh), 1)
        called = p3.argmax(1) == 0; tp = (called & (lab == 0)).sum(); rc, pr = tp / n_landed, tp / called.sum()
        name = "detector's hand only" if margin is None else f"pixels choose the hand (other glove wins by > {margin})"
        print(f"  {name:48s} switched {use2.mean():5.1%} | hand right {hok.mean():.1%} | timing +-2 {np.mean(np.abs(dt) <= 2):.1%} | 3-way right {np.trace(cm) / cm.sum():.1%} "
              f"(missed {cm[2, 2] / cm[2].sum():.0%}, blocked {cm[1, 1] / cm[1].sum():.0%}, landed {cm[0, 0] / cm[0].sum():.0%}) | landed end to end: found {rc:.1%}, real {pr:.1%}, F1 {2 * rc * pr / (rc + pr):.3f}")

if __name__ == "__main__":
    main()
