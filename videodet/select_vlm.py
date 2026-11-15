"""Pick the windows the VLM (vlm_check.py) is asked about -> videodet/vlm_selection.json.

Per video (seeded): 25 highest-scoring GAP windows by the main detector (de-duplicated >= 32
frames apart), 15 random gap windows, 10 lowest-scoring labelled-punch windows (de-duplicated),
10 random labelled-punch windows. The random groups calibrate the VLM itself: how often it says
Yes on typical gaps and on typical labelled punches.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import windows as Wd  # noqa: E402

DET = sys.argv[1] if len(sys.argv) > 1 else "videomae_probe"


def dedup(order, st, k):
    out = []
    for i in order:
        if all(abs(int(st[i]) - int(st[j])) >= 32 for j in out):
            out.append(i)
        if len(out) == k:
            break
    return out


rng = np.random.default_rng(0)
sel = []
for v in range(1, 11):
    s = np.load(HERE.parent / "detect" / "scores" / DET / f"V{v}.npz")["score"]
    crop = np.load(HERE / "cache" / "feats" / f"V{v}.npz")["crop"]
    lab = Wd.window_labels(v)
    st = lab["start"]
    gap, pun = np.where(lab["is_gap"])[0], np.where(lab["is_punch"])[0]
    groups = {"top_gap": dedup(gap[np.argsort(-s[gap])], st, 25),
              "rand_gap": list(rng.choice(gap, 15, replace=False)),
              "low_punch": dedup(pun[np.argsort(s[pun])], st, 10),
              "rand_punch": list(rng.choice(pun, 10, replace=False))}
    for g, idx in groups.items():
        for i in idx:
            sel.append(dict(video=v, start=int(st[i]), crop=[float(c) for c in crop[i]], group=g,
                            score=float(s[i]), label=str(lab["label"][i])))
(HERE / "vlm_selection.json").write_text(json.dumps(sel, indent=0))
print(len(sel), "windows")
