"""Score the fine-tuned video models for landed / blocked / missed (HPC outputs in external/landed_ft/).
Each file: test probabilities for the held-out gym, at the best-validation epoch and at the last epoch.
    python olympic/landed_ft_score.py [base|large ...]
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import landed_probe as lp

def main():
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "landed_clips.json"))); y = np.array([c["y"] for c in clips]); N = len(y); out = {}
    print(f"{N} punches: landed {np.mean(y == 0):.1%}, blocked {np.mean(y == 1):.1%}, missed {np.mean(y == 2):.1%}")
    for model in (sys.argv[1:] or ["base", "large"]):
        fs = sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", f"landed_ft_{model}_kam*.npz")))
        if len(fs) < 2:
            print(model, "not complete:", [os.path.basename(f) for f in fs]); continue
        for key in ("P_best", "P_last"):
            P = np.zeros((N, 3)); ep = []
            for f in fs:
                z = np.load(f); P[z["idx"]] = z[key]; ep.append(int(z["best_epoch"]))
            out[f"{model}/{key}"] = lp.report(f"fine-tuned VideoMAE-{model} ({'best-validation epoch ' + str(ep) if key == 'P_best' else 'last epoch'})", y, P)
            if key == "P_best":
                b = (y == 0).astype(int); s = P[:, 0]
                for t in (0.5, 0.6, 0.7, 0.8):
                    m = s >= t; print(f"      call 'landed' when P >= {t}: {m.mean():.0%} of punches, {b[m].mean():.1%} of them landed; catches {b[m].sum() / b.sum():.0%} of landed")
                yh = P.argmax(1); cm = np.zeros((3, 3), int); np.add.at(cm, (y, yh), 1); print("      rows truth, columns predicted (landed, blocked, missed):", cm.tolist())
    json.dump(out, open(os.path.join(ROOT, "runs", "olympic_landed_finetune.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
