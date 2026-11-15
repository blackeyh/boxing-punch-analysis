"""Step 4: grade the CURRENT counter (unchanged model + tracker) against the referee-labelled punches.

Labels (olympic/events.json): every thrown punch with hand (left/right) and outcome (head/body/block/miss);
they mark the contact moment (median 4 frames at 50 fps) and do NOT say which boxer threw it.
Counted punches: runs/external/O_*/summary.json (per tracked fighter, type, start/end).
Matching, in time order, one-to-one: a label is FOUND by a counted punch whose span (grown by TOL) contains
the label's middle frame.
  found  = labelled punches that were counted            (1 - found = missed)
  real   = counted punches that match a labelled punch   (1 - real = counts with no label: fake OR unlabelled)
  hand   = among found: counted hand (lead/rear + stance call) == labelled hand
  who    = among found: the counting fighter is the one whose wrist is nearest the referee's box
  chance = the same 'found' with the labels shifted by SHIFT seconds (what random overlap alone gives)
    python olympic/score_counter.py [ids...]
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOL = 10              # video frames (0.2 s at 50 fps)
SHIFT = 7.0           # seconds, for the chance baseline
LEAD = {"Jab", "Lead Hook", "Lead Uppercut"}

def load(cid):
    S = json.loads((ROOT / "runs" / "external" / cid / "summary.json").read_text())
    T = np.load(ROOT / "external" / "tracks" / f"{cid}.npz"); fr = T["frame"]; W, H = float(T["W"]), float(T["H"])
    counts = []
    for k, f in enumerate(S["fighters"]):
        south = f["stance"]["call"] == "southpaw"
        for e in f["events"]:
            lead = e["type"] in LEAD
            counts.append(dict(f0=int(fr[e["start"]]), f1=int(fr[min(e["end"], len(fr) - 1)]), fighter=k, type=e["type"],
                               hand="left" if lead != south else "right"))
    wr = [T[f"f{k}_xy"][:, [9, 10]] * [W, H] for k in (1, 2)]; pres = [T[f"f{k}_present"] for k in (1, 2)]
    return S, sorted(counts, key=lambda c: c["f0"]), fr, wr, pres

def nearest_fighter(e, fr, wr, pres):
    i = int(np.clip(np.searchsorted(fr, (e["start"] + e["end"]) // 2), 0, len(fr) - 1))
    b = e["boxes"][0]; c = np.array([(b[1] + b[3]) / 2, (b[2] + b[4]) / 2])
    d = [np.linalg.norm(wr[k][i] - c, axis=1).min() if pres[k][i] else np.inf for k in (0, 1)]
    return (int(np.argmin(d)) if np.isfinite(min(d)) else None), min(d)

def match(labels, counts, shift=0):
    used = set(); pairs = []
    for li, e in enumerate(labels):
        m = (e["start"] + e["end"]) // 2 + shift
        for ci, c in enumerate(counts):
            if ci not in used and c["f0"] - TOL <= m <= c["f1"] + TOL:
                used.add(ci); pairs.append((li, ci)); break
    return pairs

def main():
    ev = json.load(open(ROOT / "olympic" / "events.json"))
    ids = sys.argv[1:] or sorted(p.name for p in (ROOT / "runs" / "external").glob("O_*") if (p / "summary.json").exists())
    A = dict(lab=0, cnt=0, found=0, chance=0, hand=0, who=0, who_n=0, near=0, sec=0.0)
    by_out = {}
    rows = []
    for cid in ids:
        S, counts, fr, wr, pres = load(cid)
        labels = [e for e in ev if "O_" + e["video"] == cid and e["end"] <= fr[-1]]
        pairs = match(labels, counts)
        span = int(fr[-1]); sh = int(SHIFT * 50)
        chance = len(match([dict(e, start=(e["start"] + sh) % span, end=(e["start"] + sh) % span + e["end"] - e["start"]) for e in labels], counts))
        hand = sum(labels[l]["hand"] == counts[c]["hand"] for l, c in pairs)
        who = who_n = near = 0
        for e in labels:
            k, d = nearest_fighter(e, fr, wr, pres); near += d < 80
        for l, c in pairs:
            k, d = nearest_fighter(labels[l], fr, wr, pres)
            if k is not None and d < 150:
                who_n += 1; who += k == counts[c]["fighter"]
        fl = {l for l, _ in pairs}
        for i, e in enumerate(labels):
            o = by_out.setdefault(e["outcome"], [0, 0]); o[0] += 1; o[1] += i in fl
        seen = [f["seen_fraction"] for f in S["fighters"]]; st = [f["stance"]["call"][0] + ("?" if f["stance"]["uncertain"] else "") for f in S["fighters"]]
        rows.append(f"{cid:20s} labels {len(labels):3d} counted {len(counts):4d} | found {len(pairs)/max(len(labels),1):4.0%} (chance {chance/max(len(labels),1):4.0%}) "
                    f"real {len(pairs)/max(len(counts),1):4.0%} hand {hand/max(len(pairs),1):4.0%} who {who/max(who_n,1):4.0%} | wrist@box {near/max(len(labels),1):4.0%} seen {seen[0]:.0%}/{seen[1]:.0%} stance {'/'.join(st)}")
        for k, v in dict(lab=len(labels), cnt=len(counts), found=len(pairs), chance=chance, hand=hand, who=who, who_n=who_n, near=near, sec=S["seconds"]).items():
            A[k] += v
    print("\n".join(rows))
    print(f"\nALL {len(ids)} bouts, {A['sec']/60:.0f} min: {A['lab']} labelled punches ({A['lab']/A['sec']*60:.0f}/min), {A['cnt']} counted ({A['cnt']/A['sec']*60:.0f}/min)")
    print(f"  found {A['found']/A['lab']:.1%}  (chance level {A['chance']/A['lab']:.1%})")
    print(f"  real  {A['found']/A['cnt']:.1%}  (counts with a label; the rest are fake or unlabelled)")
    print(f"  hand right {A['hand']/max(A['found'],1):.1%} of found (50% = coin flip)")
    print(f"  right fighter {A['who']/max(A['who_n'],1):.1%} of found (n={A['who_n']})")
    print(f"  a tracked fighter's wrist within 80 px of the referee's box: {A['near']/A['lab']:.1%} of labelled punches")
    print("  found by outcome: " + ", ".join(f"{o} {b/a:.0%} of {a}" for o, (a, b) in sorted(by_out.items())))
    json.dump(json.loads(json.dumps(dict(A, by_outcome=by_out, tol_frames=TOL), default=float)), open(ROOT / "runs" / "olympic_score_baseline.json", "w"), indent=1)

if __name__ == "__main__":
    main()
