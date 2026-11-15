"""'Toward the opponent' gate for sparring (training-free): a counted punch is kept only if the punching
hand actually goes at the other fighter.

For each counted punch of fighter k (from run_test's summary.json):
  hand    = lead hand for Jab / Lead Hook / Lead Uppercut, rear hand otherwise (stance-aware:
            orthodox lead = left wrist, southpaw lead = right wrist)
  target  = the opponent's head (nose) and shoulder centre, from the identity tracker
  d(t)    = distance wrist -> nearest target point, in the PUNCHER's torso lengths, over the event +-3 frames
  KEEP if  d_start - d_min >= APPROACH   (the hand travels toward the opponent)
       or  d_min <= REACH                (the hand gets close to the opponent)
       or  the opponent is not tracked during the event (no evidence either way)
Settings FIXED before looking at any sparring score: APPROACH = 0.25, REACH = 1.0 torso lengths.
Writes a gated copy of summary.json (same format) so the same scorers apply.
    python external/opponent_gate.py RUN_SUMMARY.json OUT_SUMMARY.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
APPROACH, REACH, PAD = 0.25, 1.0, 3
LEAD_TYPES = {"Jab", "Lead Hook", "Lead Uppercut"}


def main(src, dst):
    S = json.loads(Path(src).read_text())
    t = np.load(ROOT / "external" / "tracks" / f"{S['clip']}.npz")
    W, H = float(t["W"]), float(t["H"])
    P = [t[f"f{k}_xy"] * [W, H] for k in (1, 2)]; pres = [t[f"f{k}_present"] for k in (1, 2)]
    stats = []
    for k, f in enumerate(S["fighters"]):
        me, op = P[k], P[1 - k]
        torso = np.linalg.norm((me[:, 5] + me[:, 6]) / 2 - (me[:, 11] + me[:, 12]) / 2, axis=1)
        orth = f["stance"]["call"] == "orthodox"
        kept, dropped = [], 0
        for e in f["events"]:
            lead = e["type"] in LEAD_TYPES
            wrist = 9 if (lead == orth) else 10                              # COCO 9 = left wrist, 10 = right
            a, b = max(0, e["start"] - PAD), min(len(me) - 1, e["end"] + PAD)
            idx = [i for i in range(a, b + 1) if pres[k][i] and pres[1 - k][i]]
            if len(idx) < 2:
                kept.append(dict(e, gate="no_opponent")); continue
            tl = np.median(torso[idx]) + 1e-6
            tgt = np.stack([op[idx, 0], (op[idx, 5] + op[idx, 6]) / 2], 1)  # nose, shoulder centre
            d = np.min(np.linalg.norm(tgt - me[idx, wrist][:, None], axis=2), axis=1) / tl
            appr, dmin = float(d[0] - d.min()), float(d.min())
            if appr >= APPROACH or dmin <= REACH:
                kept.append(dict(e, gate="kept", approach=round(appr, 2), d_min=round(dmin, 2)))
            else:
                dropped += 1
        stats.append((len(f["events"]), len(kept), dropped))
        f["events"] = kept
        f["counts"] = {c: sum(x["type"] == c for x in kept) for c in f["counts"]}
        f["total"] = len(kept)
    S["gate"] = dict(approach=APPROACH, reach=REACH, pad=PAD)
    Path(dst).write_text(json.dumps(S, indent=1))
    for k, (n, kk, dd) in enumerate(stats):
        print(f"fighter {k + 1}: {n} counted -> {kk} kept, {dd} dropped (hand did not go at the opponent)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
