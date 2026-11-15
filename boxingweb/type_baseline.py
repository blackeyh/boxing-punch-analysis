"""Punch TYPE from the skeleton detector on the labelled BoxingWeb test punches (truth span, truth boxer), as the
baseline for a pixel type model: over the labelled span, sum the detector's class probabilities.
  family (straight / hook / uppercut) given the true hand; and all 6 classes (hand + family).
Writes runs/type_baseline.npz (per bw_clips.json test clip: skeleton family probabilities) for later fusion.
    python boxingweb/type_baseline.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mem_guard  # noqa: F401
import numpy as np, torch
import punch_tcn as pt
FAM = {"straight": 0, "hook": 1, "uppercut": 2}

def main():
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "bw_clips.json"))); models = []
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "punch-tcn", "model_seed*.pt"))):
        m = pt.TCN(157).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True)); models.append(m)
    side = {(e["video"], e["start"], e["end"], e["technique"]): 0 if e["side"] == "red" else 1 for e in json.load(open(os.path.join(HERE, "events.json")))}
    cache = {}; rows = []; P3 = np.full((len(clips), 3), np.nan)
    for n, c in enumerate(clips):
        if c["split"] != "test":
            continue
        if c["id"] not in cache:
            z = np.load(os.path.join(ROOT, "external", "punch_ds", c["id"] + ".npz")); cache[c["id"]] = (np.mean([pt.predict(m, z["X"][:2]) for m in models], 0), z["Y"][:2])
        P, Y = cache[c["id"]]; b = [side[(c["id"], c["start"], c["end"], c["technique"])]]; y6 = int(Y[b[0], c["frame"]])
        if y6 == 0:
            continue
        p = P[b[0], c["start"]:c["end"] + 1, 1:].sum(0); h = (y6 - 1) % 2; fam = (y6 - 1) // 2; pf = p[[h, h + 2, h + 4]]; P3[n] = pf / max(pf.sum(), 1e-9)
        rows.append((fam, int(pf.argmax()), y6, int(p.argmax()) + 1))
    r = np.array(rows); cm = np.zeros((3, 3), int); np.add.at(cm, (r[:, 0], r[:, 1]), 1)
    print(f"{len(r)} labelled BoxingWeb test punches; share straight / hook / uppercut: {np.bincount(r[:, 0], minlength=3) / len(r)}")
    print(f"skeleton detector on the labelled span: family right (true hand given) {np.mean(r[:, 0] == r[:, 1]):.1%} (always 'straight' {np.mean(r[:, 0] == 0):.1%}); all 6 classes right {np.mean(r[:, 2] == r[:, 3]):.1%}")
    print("   per family recall: " + ", ".join(f"{k} {cm[v, v] / cm[v].sum():.0%}" for k, v in FAM.items()) + f" | macro {np.mean([cm[v, v] / cm[v].sum() for v in range(3)]):.1%}")
    np.savez_compressed(os.path.join(ROOT, "runs", "type_baseline.npz"), P3=P3)

if __name__ == "__main__":
    main()
