"""Window-level score of our punch counter on sparring, with the VLM as the (validated) labeller.

    python external/vlm_score_sparring.py VLM_SPAR.json RUN_SUMMARY.json --bench VLM_BENCH.json [--show K]

Cut-off for the VLM's P(yes): the value that maximises balanced agreement on the OWN-VIDEO benchmark
(the sparring answers are never used to choose it).
Per fighter and window (16 frames): VLM says punch / no punch; our model "counted" if one of its counted
punches for that fighter overlaps the window.
  found   = of the windows where the VLM sees a punch, share where our model counted one
  real    = of the windows where our model counted one, share where the VLM sees a punch
  also raw agreement and Cohen's kappa. Disagreements are listed for visual spot checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def bench_cutoff(path):
    R = json.loads(Path(path).read_text())
    y = np.array([r["label"] != "No Punch" for r in R]); p = np.array([r["p_yes"] for r in R])
    best, bt = -1, 0.5
    for t in np.unique(np.r_[p, 0.5]):
        yh = p > t; b = (yh[y].mean() + (~yh[~y]).mean()) / 2
        if b > best:
            best, bt = b, t
    return float(bt), float(best)


def kappa(a, b):
    po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vlm"); ap.add_argument("summary"); ap.add_argument("--bench", required=True)
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()
    t, b = bench_cutoff(args.bench)
    print(f"VLM cut-off from own-video benchmark: P(yes) > {t:.4f} (balanced {b:.1%} there)")
    V = json.loads(Path(args.vlm).read_text())
    S = json.loads(Path(args.summary).read_text())
    frames = np.load(ROOT / "external" / "pose" / f"{S['clip']}.npz")["frame"]
    ev = [[(int(frames[e["start"]]), int(frames[e["end"]])) for e in f["events"]] for f in S["fighters"]]
    out = {}
    for k, name in ((1, "red"), (2, "green")):
        W = [w for w in V if w["fighter"] == k]
        vy = np.array([w["p_yes"] > t for w in W])
        my = np.array([any(f0 <= w["video_frame"] + w["n"] - 1 and f1 >= w["video_frame"] for f0, f1 in ev[k - 1]) for w in W])
        found = my[vy].mean() if vy.any() else float("nan"); real = vy[my].mean() if my.any() else float("nan")
        out[name] = dict(windows=len(W), vlm_yes=float(vy.mean()), model_yes=float(my.mean()), found=float(found), real=float(real),
                         agree=float(np.mean(vy == my)), kappa=float(kappa(vy, my)))
        print(f"{name:5s} {len(W)} windows | VLM sees a punch in {vy.mean():.0%}, our model in {my.mean():.0%} | "
              f"found {found:.0%} | real {real:.0%} | agreement {np.mean(vy == my):.0%} (kappa {kappa(vy, my):.2f})")
        if args.show:
            dis = [(w, "model only") for w, a, m in zip(W, vy, my) if m and not a][: args.show] + \
                  [(w, "VLM only") for w, a, m in zip(W, vy, my) if a and not m][: args.show]
            for w, why in dis:
                print(f"    {why:10s} frame {w['video_frame']} ({(w['video_frame'] - 150) / 29.97:.1f} s) P(yes) {w['p_yes']:.3f} '{w['answer']}'")
    Path(args.vlm).with_suffix(".score.json").write_text(json.dumps(dict(cutoff=t, bench_balanced=b, fighters=out), indent=1))


if __name__ == "__main__":
    main()
