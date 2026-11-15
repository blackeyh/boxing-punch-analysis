"""Score VLM benchmark answers against our labels.  python external/vlm_score.py runs/<run>/vlm_<m>.json ...

  punch vs no-punch: balanced accuracy at P(yes) > 0.5 (PASS BAR, fixed before running: >= 85%), AUC
  type (labelled punches the VLM called Yes): hand + kind mapped to our class with the boxer's stance
  (orthodox: left = lead; V4 southpaw: right = lead); also hand-only and kind-only accuracy.
"""
import json
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

KIND = {"Jab": "straight", "Cross": "straight", "Lead Hook": "hook", "Rear Hook": "hook",
        "Lead Uppercut": "uppercut", "Rear Uppercut": "uppercut"}
LEAD = {"Jab", "Lead Hook", "Lead Uppercut"}
PASS = 0.85


def to_class(hand, kind, stance):
    if hand is None or kind is None:
        return None
    lead = (hand == "left") == (stance == "orthodox")
    return {("straight", True): "Jab", ("straight", False): "Cross", ("hook", True): "Lead Hook",
            ("hook", False): "Rear Hook", ("uppercut", True): "Lead Uppercut", ("uppercut", False): "Rear Uppercut"}[(kind, lead)]


def report(path):
    R = json.loads(open(path).read())
    y = np.array([r["label"] != "No Punch" for r in R]); p = np.array([r["p_yes"] for r in R]); yh = p > 0.5
    tpr, tnr = yh[y].mean(), (~yh[~y]).mean(); bal = (tpr + tnr) / 2
    print(f"== {path}  ({len(R)} windows)")
    print(f"  punch vs no-punch: balanced {bal:.1%} -> {'PASS' if bal >= PASS else 'FAIL'} (bar {PASS:.0%}) | "
          f"says Yes to real punches {tpr:.0%}, says No to no-punch {tnr:.0%} | AUC {roc_auc_score(y, p):.3f}")
    vids = sorted({r["video"] for r in R})
    print("  per video (Yes on punches / No on no-punch): " + "  ".join(
        f"V{v} {np.mean([pp > 0.5 for r, pp in zip(R, p) if r['video'] == v and r['label'] != 'No Punch']):.0%}/"
        f"{np.mean([pp <= 0.5 for r, pp in zip(R, p) if r['video'] == v and r['label'] == 'No Punch']):.0%}" for v in vids))
    P = [r for r, h in zip(R, yh) if r["label"] != "No Punch" and h]
    cls = [to_class(r["hand"], r["kind"], r["stance"]) for r in P]
    ok = [c == r["label"] for c, r in zip(cls, P) if c is not None]
    kind_ok = [r["kind"] == KIND[r["label"]] for r in P if r["kind"]]
    hand_ok = [((r["hand"] == "left") == (r["stance"] == "orthodox")) == (r["label"] in LEAD) for r in P if r["hand"]]
    print(f"  type on {len(P)} detected punches: exact class {np.mean(ok) if ok else float('nan'):.0%} (of {len(ok)} parsable) | "
          f"kind {np.mean(kind_ok) if kind_ok else float('nan'):.0%} | lead/rear hand {np.mean(hand_ok) if hand_ok else float('nan'):.0%} (chance 50%)")
    return bal


if __name__ == "__main__":
    for f in sys.argv[1:]:
        report(f)
