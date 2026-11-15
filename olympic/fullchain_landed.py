"""End of the chain on the Olympic bouts: LANDED punches counted, from raw video.
Counted punches come from olympic/fullchain.py (hpc_rtm/lists/chain_clips.json); each was judged landed / blocked /
missed by the fine-tuned VideoMAE-L that never saw that gym (external/landed_ft/landed_chain_large_*.npz).
  landed found   = referee-labelled LANDED punches (head or body) that we counted AND called landed
  landed real    = of the punches we call landed, the share that match a referee-labelled landed punch
                   (a count with no label, or matching a blocked / missed label, is wrong)
Reference: calling every counted punch landed.
    python olympic/fullchain_landed.py
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def main():
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips.json"))); N = len(clips); P = np.full((N, 3), np.nan)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", sys.argv[1] if len(sys.argv) > 1 else "landed_chain_large_kam*.npz"))):
        z = np.load(f); P[z["extra_idx"]] = z["extra_P"]
    have = ~np.isnan(P[:, 0]); lab = np.array([{"head": 0, "body": 0, "block": 1, "miss": 2, None: -1}[c["label"]] for c in clips])
    by = json.load(open(os.path.join(ROOT, "runs", "olympic_fullchain.json")))["by_outcome"]; n_landed = by["head"][0] + by["body"][0]; n_all = sum(v[0] for v in by.values())
    print(f"{N} counted punches, judged {have.sum()}; {(lab >= 0).sum()} match a referee label ({np.mean(lab[lab >= 0] == 0):.1%} landed, {np.mean(lab[lab >= 0] == 1):.1%} blocked, {np.mean(lab[lab >= 0] == 2):.1%} missed); "
          f"referee labels: {n_all} punches, {n_landed} landed")
    m = have & (lab >= 0); yh = P[m].argmax(1); y = lab[m]; cm = np.zeros((3, 3), int); np.add.at(cm, (y, yh), 1)
    print(f"\nOn the counted punches that match a label: 3-way right {np.trace(cm) / cm.sum():.1%} (always 'landed' {np.mean(y == 0):.1%}); "
          f"missed caught {cm[2, 2] / max(cm[2].sum(), 1):.0%}, blocked caught {cm[1, 1] / max(cm[1].sum(), 1):.0%}, landed kept {cm[0, 0] / max(cm[0].sum(), 1):.0%}")
    print("   rows truth, columns called (landed, blocked, missed):", cm.tolist())
    print(f"   unmatched counts (no referee label near them, {int((have & (lab < 0)).sum())}): called landed {np.mean(P[have & (lab < 0)].argmax(1) == 0):.1%}")
    res = {}
    print("\nLANDED punches, end to end:")
    def line(name, called):
        tp = (called & (lab == 0)).sum(); r, p = tp / n_landed, tp / max(called.sum(), 1)
        print(f"  {name:44s} call landed {called.sum():5d} | landed found {r:.1%} | landed real {p:.1%} | F1 {2 * r * p / max(r + p, 1e-9):.3f}")
        res[name] = dict(called=int(called.sum()), found=float(r), real=float(p), f1=float(2 * r * p / max(r + p, 1e-9)))
    line("every counted punch called landed", np.ones(N, bool))
    line("landed model, most likely class", have & (np.nan_to_num(P).argmax(1) == 0))
    for t in (0.4, 0.5, 0.6, 0.7):
        line(f"landed model, P(landed) >= {t}", have & (np.nan_to_num(P[:, 0]) >= t))
    json.dump(res, open(os.path.join(ROOT, "runs", "olympic_fullchain_landed.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
