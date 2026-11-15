"""Score a run's counted punches against the user's sparring labels (review/sparring_label/label.html).

Labels: one per thrown punch (moment of full extension), fighter red/green headgear, type or "unsure".
Mapping (checked on the rendered A2 video): Fighter 1 = RED headgear, Fighter 2 = GREEN headgear.
Matching per fighter, in time order: a label is FOUND if an unused counted punch of that fighter spans
its frame (event start - TOL .. end + TOL). Only counted punches inside the labelled stretch are scored.
  found       = labelled punches that were counted                  (1 - found = missed)
  real        = counted punches that match a labelled punch         (1 - real = fake counts)
  type_ok     = among found punches with a known type, model type == label type
  wrong_who   = missed labels that the OTHER fighter's counter caught (identity / attribution error)
    /opt/anaconda3/bin/python external/score_sparring.py LABELS.json RUN_DIR [RUN_DIR ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOL = 6                                    # frames (0.2 s at 30 fps)
WHO = {"red": 0, "green": 1}


def score(labels_path, run_dir):
    L = json.loads(Path(labels_path).read_text())
    cid = L["clip"]; f_lo = L["frame_offset"]; f_hi = f_lo + int(round((L["seconds"][1] - L["seconds"][0]) * L["fps"]))
    S = json.loads((Path(run_dir) / "summary.json").read_text())
    pose = np.load(ROOT / "external" / "pose" / f"{cid}.npz")
    frames = pose["frame"]
    step = max(1, int(round(float(pose["fps"]) / S["fps"])))   # run_test may have halved a 60 fps track
    frames = frames[::step]
    ev = []
    for k, f in enumerate(S["fighters"]):
        e = [dict(e, f0=int(frames[e["start"]]), f1=int(frames[e["end"]])) for e in f["events"]]
        ev.append([x for x in e if f_lo <= x["f1"] and x["f0"] <= f_hi])
    out = {}
    for name, k in WHO.items():
        labs = sorted([x for x in L["labels"] if x["fighter"] == name], key=lambda x: x["frame"])
        used = set(); found = []; type_n = type_ok = 0
        for lab in labs:
            hit = next((j for j, e in enumerate(ev[k]) if j not in used and e["f0"] - TOL <= lab["frame"] <= e["f1"] + TOL), None)
            found.append(hit is not None)
            if hit is not None:
                used.add(hit)
                if not lab["type"].startswith("Punch"):
                    type_n += 1; type_ok += ev[k][hit]["type"] == lab["type"]
        other = ev[1 - k]
        wrong = sum(1 for lab, fd in zip(labs, found) if not fd and any(e["f0"] - TOL <= lab["frame"] <= e["f1"] + TOL for e in other))
        n_lab, n_ev = len(labs), len(ev[k])
        out[name] = dict(labelled=n_lab, counted=n_ev, found=sum(found) / max(n_lab, 1), real=len(used) / max(n_ev, 1),
                         type_ok=type_ok / max(type_n, 1) if type_n else None, type_n=type_n, wrong_who=wrong)
    return out


if __name__ == "__main__":
    labels, runs = sys.argv[1], sys.argv[2:]
    for r in runs:
        res = score(labels, r)
        print(f"== {r}")
        for name, x in res.items():
            print(f"  {name:5s} labelled {x['labelled']:3d} counted {x['counted']:3d} | found {x['found']:.0%} | real {x['real']:.0%} | "
                  f"type correct {('%.0f%%' % (100 * x['type_ok'])) if x['type_ok'] is not None else '-'} (of {x['type_n']}) | "
                  f"missed but credited to the other fighter {x['wrong_who']}")
