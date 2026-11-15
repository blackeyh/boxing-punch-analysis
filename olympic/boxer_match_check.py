"""Does the chain score credit a referee label to the RIGHT boxer's counted punch? The referee does not name the boxer;
puncher = the stream whose labelled-hand glove is nearest the referee's box at the contact. For every label: is there a
counted punch (span grown by 0.2 s) around it in the puncher's stream, in the other stream, or both; and how often the
hand agrees in each case (a wrong-boxer match should agree about half the time).
    python olympic/boxer_match_check.py
"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import punch_tcn_oly as po
import punch_tcn as pt
TOL = 10

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); base = po.base_models(); n = dict(); hd = dict(); gap = []
    for f in sorted(glob.glob(os.path.join(po.ROOT, "external", "chain_streams", "O_*.npz"))):
        d = po.load(os.path.basename(f)[:-4], ev, base); fr = d["frame"]; K = d["K"]
        for l in d["labs"]:
            if l["stream"] < 0:
                continue
            m = (l["raw"]["start"] + l["raw"]["end"]) // 2; hit = []
            for b in (0, 1):
                c = [(a, e, k) for a, e, k in d["evs"][b] if fr[a] - TOL <= m <= fr[e] + TOL]; hit.append(min(c, key=lambda s: abs((fr[s[0]] + fr[s[1]]) / 2 - m)) if c else None)
            s = l["stream"]; key = ("puncher only" if hit[1 - s] is None else "both boxers") if hit[s] is not None else ("OTHER boxer only" if hit[1 - s] is not None else "neither")
            n[key] = n.get(key, 0) + 1
            for who, h in (("puncher's punch", hit[s]), ("other boxer's punch", hit[1 - s])):
                if h is not None:
                    o = hd.setdefault(who, [0, 0]); o[0] += 1; o[1] += (0 if h[2] % 2 == 1 else 1) == l["hand"]
            # how clear is the puncher assignment: distance of each boxer's labelled-hand glove to the referee box centre, in shoulder-hip lengths
            b = np.array(l["raw"]["boxes"]); c = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2]); w = 9 + l["hand"]; i = l["i"]
            dd = [np.linalg.norm(K[q, i, w] + 0.4 * (K[q, i, w] - K[q, i, w - 2]) - c) / max(np.linalg.norm(K[q, i, [5, 6]].mean(0) - K[q, i, [11, 12]].mean(0)), 20) for q in (0, 1)]
            if np.isfinite(dd).all():
                gap.append((min(dd), max(dd)))
    N = sum(n.values()); print(f"{N} referee labels with a puncher stream")
    for k in ("puncher only", "both boxers", "OTHER boxer only", "neither"):
        print(f"  counted punch around the label: {k:18s} {n.get(k, 0):5d} ({n.get(k, 0) / N:.1%})")
    for k, (a, b) in hd.items():
        print(f"  hand agrees with the referee, {k:22s} {b / a:.1%} of {a}")
    g = np.array(gap); print(f"  puncher's glove to the referee box (torso lengths) median {np.median(g[:, 0]):.2f}; other boxer's same-hand glove median {np.median(g[:, 1]):.2f}; "
                             f"clear (other at least 2x farther) {np.mean(g[:, 1] >= 2 * g[:, 0]):.1%}")

if __name__ == "__main__":
    main()
