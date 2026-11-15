"""Who gets the credit? Per bout, each fighter's share of the punches thrown: referee (label -> boxer whose labelled-hand
glove is nearest the referee's box) vs the detector's counts per boxer. Variants: all labels or only the CLEAR ones
(other boxer's glove at least twice as far), and the detector with / without the opponent's skeleton as input.
    python olympic/credit_check.py
"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, torch
import punch_tcn_oly as po
import punch_tcn as pt
TOL = 10

def models(name):
    out = []
    for f in sorted(glob.glob(os.path.join(po.ROOT, "runs", name, "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); out.append(m)
    return out

def main():
    ev = json.load(open(os.path.join(HERE, "events.json"))); ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(po.ROOT, "external", "chain_streams", "O_*.npz")))
    D = [po.load(c, ev, po.base_models()) for c in ids]
    for d in D:                                                    # clear = other boxer's same-hand glove at least twice as far from the referee's box
        K = d["K"]
        for l in d["labs"]:
            b = np.array(l["raw"]["boxes"]); c = np.array([(b[:, 1].min() + b[:, 3].max()) / 2, (b[:, 2].min() + b[:, 4].max()) / 2]); w = 9 + l["hand"]; i = l["i"]
            dd = [np.linalg.norm(K[q, i, w] + 0.4 * (K[q, i, w] - K[q, i, w - 2]) - c) for q in (0, 1)]; l["clear"] = bool(np.isfinite(dd).all() and max(dd) >= 2 * min(dd))
    for name, thr, noopp in (("punch-tcn", 0.40, False), ("punch-tcn-noopp", None, True)):
        ms = models(name); thr = thr or float(np.mean([json.load(open(f))["threshold"] for f in glob.glob(os.path.join(po.ROOT, "runs", name, "results_seed*.json"))]))
        sh = []; n = dict(p=0, both=0, other=0, none=0); cnt = 0; lab = 0
        for d in D:
            X = d["X"][:2].copy()
            if noopp:
                X[..., pt.OPP] = 0; X[..., -1] = 0
            P = np.mean([pt.predict(m, X) for m in ms], 0); evs = [pt.events(P[b], d["M"][b], thr) for b in (0, 1)]; fr = d["frame"]; cnt += len(evs[0]) + len(evs[1])
            L = [l for l in d["labs"] if l["stream"] >= 0]; lab += len(d["labs"])
            for l in L:
                hit = [any(fr[a] - TOL <= l["mid"] <= fr[e] + TOL for a, e, _ in evs[b]) for b in (0, 1)]; s = l["stream"]
                n["p" if hit[s] and not hit[1 - s] else "both" if hit[s] else "other" if hit[1 - s] else "none"] += 1
            a_all = np.mean([l["stream"] == 0 for l in L]); a_clr = np.mean([l["stream"] == 0 for l in L if l["clear"]]); sh.append((a_all, a_clr, len(evs[0]) / max(len(evs[0]) + len(evs[1]), 1)))
        sh = np.array(sh); N = sum(n.values())
        print(f"\n{name} (threshold {thr:.2f}): {cnt} counted for {lab} labels | around a label: puncher only {n['p'] / N:.1%}, both boxers {n['both'] / N:.1%}, other boxer only {n['other'] / N:.1%}, neither {n['none'] / N:.1%}")
        for k, nm in ((0, "all labels"), (1, "clear labels only")):
            print(f"   fighter A's share of punches thrown, referee ({nm}) vs counted: correlation {np.corrcoef(sh[:, k], sh[:, 2])[0, 1]:.2f}, mean error {np.mean(np.abs(sh[:, k] - sh[:, 2])) * 100:.1f} points; referee spread {sh[:, k].std() * 100:.1f}, counted spread {sh[:, 2].std() * 100:.1f}")
    print(f"\n   referee share from all labels vs from clear labels only: correlation {np.corrcoef(sh[:, 0], sh[:, 1])[0, 1]:.2f}")

if __name__ == "__main__":
    main()
