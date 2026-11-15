"""Score the pixel contact + landed model behind our own detector (Olympic bouts, held-out gym for every punch).
Inputs: hpc_rtm/lists/chain_clips_v3.json (5,198 counted punches; coarse contact from the skeleton model) and
external/landed_ft/cl_large_kam*.npz (per punch: 13 scan positions x [landed, blocked, missed, off-centre]).
Contact = the scan position with the lowest off-centre score; verdict = most likely of landed/blocked/missed there
("at the peak") or averaged over positions weighted by how centred each looks ("weighted").
    python olympic/fullchain_scan.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
MAP = {"head": 0, "body": 0, "block": 1, "miss": 2, None: -1}

def main():
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); ev = json.load(open(os.path.join(HERE, "events.json"))); N = len(clips)
    S = np.full((N, 13, 4), np.nan); offs = None
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", (sys.argv[1] if len(sys.argv) > 1 else "cl_large") + "_kam*.npz"))):
        z = np.load(f); S[z["chain_idx"]] = z["chain_scan"]; offs = z["scan_offsets"]
        h = json.loads(str(z["hist"])); print(os.path.basename(f), "best epoch", int(z["best_epoch"]), "| referee clips scanned alone: within +-2 frames",
              f"{np.mean(np.abs(offs[z['scan'][:, :, 3].argmin(1)]) <= 2):.1%}, +-4 {np.mean(np.abs(offs[z['scan'][:, :, 3].argmin(1)]) <= 4):.1%}")
    have = ~np.isnan(S[:, 0, 0]); lab = np.array([MAP[c["label"]] for c in clips]); cen = 1 - S[:, :, 3]
    k = np.nan_to_num(cen, nan=-1).argmax(1); d = offs[k]; p3 = S[np.arange(N), k, :3]; p3 = p3 / p3.sum(1, keepdims=True)
    wgt = np.nan_to_num(cen) ** 2; pw = (S[:, :, :3] * wgt[:, :, None]).sum(1); pw = pw / np.maximum(pw.sum(1, keepdims=True), 1e-9)
    by = {}
    for e in ev:
        by.setdefault("O_" + e["video"], []).append(e)
    rows = []
    for n, c in enumerate(clips):
        if c["label"] is None or not have[n]:
            continue
        cand = [e for e in by[c["id"]] if c["f0"] - 10 <= (e["start"] + e["end"]) / 2 <= c["f1"] + 10 and e["outcome"] == c["label"] and e["hand"] == c["label_hand"]]
        if cand:
            e = min(cand, key=lambda e: abs((e["start"] + e["end"]) / 2 - (c["f0"] + c["f1"]) / 2)); mid = (e["start"] + e["end"]) / 2
            rows.append((c["old_frame"] - mid, c["coarse"] - mid, c["coarse"] + d[n] - mid, c["hand"] == e["hand"]))
    r = np.array(rows, float); print(f"\nTIMING on {len(r)} counted punches matched to a referee label (frames at 50 fps):")
    for name, col in (("greatest arm extension", 0), ("skeleton contact model", 1), ("skeleton model + pixel scan", 2)):
        print(f"  {name:28s} within +-2 frames {np.mean(np.abs(r[:, col]) <= 2):.1%}, +-5 {np.mean(np.abs(r[:, col]) <= 5):.1%}, median error {np.median(np.abs(r[:, col])):.1f}")
    print(f"  hand right (detector) {r[:, 3].mean():.1%}")
    n_landed = sum(e["outcome"] in ("head", "body") for e in ev); res = {}
    for name, P in (("verdict at the peak", p3), ("verdict weighted over the scan", pw)):
        m = have & (lab >= 0); yh = P[m].argmax(1); y = lab[m]; cm = np.zeros((3, 3), int); np.add.at(cm, (y, yh), 1)
        called = have & (P.argmax(1) == 0); tp = (called & (lab == 0)).sum(); rc, pr = tp / n_landed, tp / max(called.sum(), 1)
        print(f"\n{name}: on counted punches matching a label, 3-way right {np.trace(cm) / cm.sum():.1%} (always 'landed' {np.mean(y == 0):.1%}); "
              f"missed caught {cm[2, 2] / cm[2].sum():.0%}, blocked caught {cm[1, 1] / cm[1].sum():.0%}, landed kept {cm[0, 0] / cm[0].sum():.0%}")
        print(f"   landed end to end: found {rc:.1%}, real {pr:.1%}, F1 {2 * rc * pr / (rc + pr):.3f}   (every count called landed: found 79.5%, real 44.1%, F1 0.568)")
        res[name] = dict(acc=float(np.trace(cm) / cm.sum()), cm=cm.tolist(), landed_found=float(rc), landed_real=float(pr))
    print("\nhistory so far, 3-way right behind our detector: 61.0% (arm extension) -> 65.2% (skeleton contact model); with the referee's moment and hand: 74.2%")
    json.dump(res, open(os.path.join(ROOT, "runs", "olympic_fullchain_scan.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
