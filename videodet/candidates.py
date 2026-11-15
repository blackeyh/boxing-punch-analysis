"""Candidate MISSED punches: runs of consecutive gap windows whose videomae_probe score is >= the
video's own 95%-recall threshold (t95: 95% of that video's labelled-punch windows score above it),
merged into events. Output videodet/missed_candidates.csv (one row per event, ranked by peak
score within each video): video, first/last frame covered, n_windows, peak/mean score, type guess
at the peak, time (mm:ss). For cross-checking with the skeleton detector; not ground truth."""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import windows as Wd  # noqa: E402

rows = []
for v in range(1, 11):
    d = np.load(HERE.parent / "detect" / "scores" / "videomae_probe" / f"V{v}.npz")
    L = Wd.window_labels(v)
    s, g = d["score"], L["is_gap"]
    t95 = np.quantile(s[L["is_punch"]], 0.05)
    hit = g & (s >= t95)
    i = 0
    while i < len(s):
        if not hit[i]:
            i += 1; continue
        j = i
        while j + 1 < len(s) and hit[j + 1]:
            j += 1
        k = i + int(np.argmax(s[i:j + 1]))
        f0, f1 = int(L["start"][i]), int(L["start"][j]) + Wd.W - 1
        t = f0 / Wd.fps(v)
        rows.append(dict(video=v, first_frame=f0, last_frame=f1, n_windows=j - i + 1, peak_score=round(float(s[k]), 4),
                         mean_score=round(float(s[i:j + 1].mean()), 4), peak_start=int(L["start"][k]),
                         type_guess=str(d["type"][k]), time=f"{int(t // 60)}:{t % 60:04.1f}", t95_video=round(float(t95), 4)))
        i = j + 1
rows.sort(key=lambda r: (r["video"], -r["peak_score"]))
with open(HERE / "missed_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
for v in range(1, 11):
    r = [x for x in rows if x["video"] == v]
    print(f"V{v}: {len(r)} candidate events, {sum(x['peak_score'] >= 0.5 for x in r)} with peak >= 0.5")
print("total", len(rows))
