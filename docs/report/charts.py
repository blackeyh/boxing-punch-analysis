"""Charts for the project report. Every number is copied from PROJECT_LOG.md (see docs/research_log.md).
    python docs/report/charts.py   ->  docs/media/charts/*.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media", "charts")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#1f2430", "#6b7280", "#e5e7eb"
ACC, BASE, BAD = "#d9480f", "#4c6ef5", "#adb5bd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200); plt.close(fig)


def bars(name, labels, values, title, ylabel, highlight=(), ref=None, ref_label=None, fmt="{:.1f}", ylim=None, size=(7.2, 3.2), colors=None):
    fig, ax = plt.subplots(figsize=size)
    cols = colors or [ACC if i in highlight else BASE for i in range(len(values))]
    b = ax.bar(range(len(values)), values, color=cols, width=0.62)
    for r, v in zip(b, values):
        ax.text(r.get_x() + r.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    if ref is not None:
        ax.axhline(ref, color=MUTED, ls="--", lw=1); ax.text(len(values) - 0.5, ref, ref_label, ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(values))); ax.set_xticklabels(labels, fontsize=8.2); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold"); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    if ylim:
        ax.set_ylim(*ylim)
    save(fig, name)


# 1. Phase 1: punch type on BoxingVI, leave-one-video-out balanced accuracy
bars("phase1_type.png",
     ["majority\nclass", "raw 2D\n(RF)", "+ normal-\nisation", "+ southpaw\nmirrored", "+ velocity", "1D-CNN", "CNN 2D +\nMP 3D", "3-model\nensemble"],
     [15.7, 46.5, 57.1, 63.1, 66.8, 72.7, 73.7, 75.2],
     "Punch type, 6 classes, tested on unseen videos (LOVO)", "balanced accuracy (%)", highlight=(6, 7), ref=60, ref_label="goal 60%", ylim=(0, 85))

# 2. Pose ceiling on real bouts
fig, ax = plt.subplots(figsize=(7.2, 3.3)); import numpy as np
x = np.arange(2); w = 0.36
for k, (lab, vals, c) in enumerate((("MediaPipe", [45.9, 15.3], BAD), ("RTMPose", [99.6, 96.2], ACC))):
    r = ax.bar(x + (k - 0.5) * w, vals, w, color=c, label=lab)
    for rr, v in zip(r, vals):
        ax.text(rr.get_x() + rr.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(["puncher has a skeleton\nat the labelled punch", "both boxers have a skeleton\n(contact punches)"])
ax.set_ylabel("% of labelled punches"); ax.set_ylim(0, 112); ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("Pose step on real Olympic bouts (231 labelled punches)", loc="left", fontsize=11, fontweight="bold"); save(fig, "pose_ceiling.png")

# 3. Old vs new chain on the Olympic bouts (never trained on them)
fig, ax = plt.subplots(figsize=(7.2, 3.6)); labs = ["punches found", "counts matching\na label", "F1 x 100", "hand right"]
old, new = [24.8, 47.7, 32.6, 60.6], [79.4, 70.3, 74.6, 79.7]
for k, (lab, vals, c) in enumerate((("old: MediaPipe + model from single-boxer videos", old, BAD), ("new chain: RTMPose + selector + re-ID + TCN", new, ACC))):
    r = ax.bar(np.arange(4) + (k - 0.5) * w, vals, w, color=c, label=lab)
    for rr, v in zip(r, vals):
        ax.text(rr.get_x() + rr.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(np.arange(4)); ax.set_xticklabels(labs); ax.set_ylim(0, 100); ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.set_title("29 Olympic bouts, 4,605 referee-labelled punches", loc="left", fontsize=11, fontweight="bold"); save(fig, "olympic_chain.png")

# 4. Identity: who is who
bars("identity.png", ["motion\ncontinuity*", "ResNet-50\n(generic)", "DINOv2-S\n(generic)", "OSNet\n(re-ID)", "OSNet-AIN\n(re-ID)"],
     [84.1, 94.1, 84.3, 99.8, 99.8], "Keeping boxer A and B apart for a whole round (BoxingWeb test, colour)",
     "% of frames with the right identity", highlight=(3, 4), ylim=(0, 108), size=(7.2, 3.0))

# 5. Landed / blocked / missed
bars("landed.png", ["always\n'landed'", "skeleton\ncues", "frozen video\nfeatures", "fine-tuned\nVideoMAE-B", "fine-tuned\nVideoMAE-L", "L + BoxingWeb\n(2 models avg)"],
     [62.1, 57.1, 65.7, 71.3, 74.2, 76.1], "Landed / blocked / missed at the referee's contact moment",
     "3-way accuracy (%)", highlight=(4, 5), ylim=(0, 90), colors=[BAD, BASE, BASE, BASE, ACC, ACC])
bars("landed_chain.png", ["always\n'landed'", "arm\nextension", "skeleton\ncontact model", "pixel\ncontact scan", "+ fair\nlabel matching"],
     [62.8, 61.0, 65.2, 66.5, 69.1], "Landed step behind our own detector (timing and hand from the chain)",
     "3-way accuracy (%)", highlight=(4,), ylim=(0, 85), colors=[BAD, BASE, BASE, BASE, ACC])

# 6. Sparring failure + fixes tried (share of counts that are real punches, VLM-judged, A2 first 2 min)
bars("sparring.png", ["red boxer", "green boxer"], [41, 50], "Old model on sparring: share of counts that are real punches (AI-judged)",
     "% of counts", ylim=(0, 100), ref=50, ref_label="half", size=(5.5, 2.8), colors=[BAD, BAD])
print("charts ->", os.path.abspath(OUT))
