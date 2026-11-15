"""Summarise the Qwen3-VL answers (videodet/vlm_answers.json) per group and per video.

The random groups calibrate the VLM: rand_punch = how often it says Yes to a typical LABELLED punch
(its sensitivity), rand_gap = how often it says Yes to a typical gap window. If top_gap (highest
detector scores among gap windows) gets a Yes rate near rand_punch, those windows look like real
punches to an independent model, i.e. likely missed annotations."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ans = json.loads((HERE / "vlm_answers.json").read_text())
groups = ["rand_punch", "low_punch", "top_gap", "rand_gap"]
out = {"n": len(ans), "groups": {}, "per_video_top_gap": {}}
print(f"{len(ans)} windows; Yes = P(yes) > 0.5 from next-token logits of Qwen3-VL-8B-Instruct")
print("| group | n | VLM Yes rate | mean P(yes) | mean detector score |")
print("|---|---|---|---|---|")
for g in groups:
    a = [x for x in ans if x["group"] == g]
    p = np.array([x["p_yes"] for x in a])
    d = dict(n=len(a), yes_rate=float((p > 0.5).mean()), mean_p=float(p.mean()), mean_score=float(np.mean([x["score"] for x in a])))
    out["groups"][g] = d
    print(f"| {g} | {d['n']} | {d['yes_rate']:.0%} | {d['mean_p']:.2f} | {d['mean_score']:.2f} |")
print("\ntop_gap Yes rate per video (n=25 each) vs rand_gap (n=15) and rand_punch (n=10):")
print("| video | top_gap | rand_gap | rand_punch | low_punch |")
print("|---|---|---|---|---|")
for v in range(1, 11):
    r = {g: np.mean([x["p_yes"] > 0.5 for x in ans if x["group"] == g and x["video"] == v]) for g in groups}
    out["per_video_top_gap"][v] = {g: float(r[g]) for g in groups}
    print(f"| V{v} | {r['top_gap']:.0%} | {r['rand_gap']:.0%} | {r['rand_punch']:.0%} | {r['low_punch']:.0%} |")
# does the detector score rank gap windows the way the VLM does?
gaps = [x for x in ans if x["group"] in ("top_gap", "rand_gap")]
mass = np.array([x["mass"] for x in ans])
out["min_yes_no_mass"] = float(mass.min())
print(f"\nYes+No probability mass: min {mass.min():.3f}, median {np.median(mass):.3f} (answers are well-formed if ~1)")
(HERE / "vlm_summary.json").write_text(json.dumps(out, indent=1))
