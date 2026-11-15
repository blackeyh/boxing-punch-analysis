# videodet: an independent, pixel-only punch detector

*Written by the videodet background agent (2026-11-13). Saved by the main session because the
agent's harness does not let subagents write report files.*

Branch `videodet`. Scores every shared window (`detect/WINDOWS.md`: 16 frames, stride 4, 77,925 windows over V1..V10) with "is a punch being thrown here?", **from the video pixels only**. No skeleton or pose file is read anywhere. Labels are used only for evaluation and as the LOVO probe's training targets.

## TL;DR
* **Detector to use: `detect/scores/videomae_probe/`.** Frozen **VideoMAE-Large (Kinetics-400)** features of a person-centred crop, plus a **leave-one-video-out (LOVO) logistic probe**. Labelled-punch vs gap windows: **mean per-video ROC-AUC 0.930** (range 0.83 to 0.98), pooled 0.881. At one global threshold (0.056) that catches 95% of labelled-punch windows, **91 to 100% of labelled punches** per video have a window above it, but **9 to 64% of gap windows** per video also score above it.
* Zero-shot baselines, also saved: `videomae_zs` (Kinetics "punching bag" + "punching person" probability), mean AUC 0.716; `xclip_zs` (X-CLIP prompts), mean AUC 0.726.
* **Gap windows are not clean negatives.** Qwen3-VL-8B says "punch" for 64% of the detector's top-scoring gap windows, vs 56% of random labelled punches and 13% of random gap windows. Visual QC agrees. So the gap counts above are upper bounds on true false alarms.

## 1. Model survey and choice
Hardware: RTX 5000 Ada (32 GB); budget about 6 GPU-hours; 77,925 windows.

| Family | Candidates (Hugging Face, Nov 2026) | Licence | Verdict |
|---|---|---|---|
| (c) action-recognition backbone, Kinetics-400 | `MCG-NJU/videomae-{base,large}-finetuned-kinetics` (16 × 224², safetensors) | CC-BY-NC-4.0 | **Main feature extractor.** Kinetics-400 includes "punching bag" and "punching person (boxing)". B + L + X-CLIP together run at 31 to 42 windows/s. |
| (a) zero-shot video-text | `microsoft/xclip-base-patch32`; also looked at `-patch16-zero-shot`, `-large-patch14` | MIT | B/32 used for prompts and as an extra embedding (cheap, same pass) |
| (b) open VLM | `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 | Qwen3-VL-8B used on a 600-window subset as an independent adjudicator (13 s/window here, too slow for all windows) |
| others | V-JEPA 2 (SSv2/Diving48 heads), VideoMAE-SSv2 (.bin only), InternVideo2 (gated, needs remote code) | | not used |

Design:
1. **Crop first.** Boxers fill only about 1/3 of the frame height and are off-centre in some videos. A pixel person detector (torchvision Faster R-CNN, COCO) runs every 8 frames; the largest person box is median-smoothed and interpolated. Each window gets **one fixed square crop**: 1.25× the union of the boxer's boxes over its 16 frames.
2. **Zero-shot alone is not enough**, because "punching bag" fires on every window of a bag workout. Frozen features plus a small **LOVO** probe learn the moment of a punch. Each video's scores come from a model that never saw that video.
3. VideoMAE-L features beat Base, X-CLIP and every concatenation (§3).

## 2. Pipeline (`videodet/`)
| Step | Script | Where | Output |
|---|---|---|---|
| windows + labels | `windows.py` | | `window_labels(v)`: start, overlap, is_punch, is_gap, label |
| weights | `fetch_weights.py` | HPC login node | pinned revisions; SHA-256 in `SOURCES.md` |
| boxes, crops, features | `extract.py` | HPC GPU, 2 jobs | per window: VideoMAE-B/L K400 logits + pooled features (768 / 1024), X-CLIP logits for 16 prompts + 512-d embedding, crop box. Stored in `videodet/cache/` (git-ignored, ~560 MB) |
| zero-shot scores, LOVO probe, validation | `score.py` | Mac CPU | `detect/scores/*`, `validation.md` / `.json` |
| probe grid | `probe_grid.py` | HPC CPU job | `probe_grid.json`, `probe_grid_extra.json` |
| QC sheets | `qc.py` | Mac | `qc/*.png` |
| missed-punch candidates | `candidates.py` | Mac | `missed_candidates.csv` |
| VLM second opinion | `select_vlm.py`, `vlm_check.py`, `vlm_summary.py` | HPC GPU | `vlm_selection.json`, `vlm_answers.json`, `vlm_summary.json` |

Decoding: OpenCV, sequential; frame k is the k-th decoded frame, as in video3d (offset 0 was verified there). Window starts are asserted equal to WINDOWS.md.

**X-CLIP prompts.** Score = softmax mass on the first 7.
* Punch: "a person throwing a punch", "a boxer throwing a jab", "...a straight cross punch", "...a hook punch", "...an uppercut", "a person punching a heavy bag", "a person shadow boxing and throwing punches".
* Not punch: "a boxer standing still in guard position", "a boxer moving their feet", "a boxer bouncing on their feet with hands up", "a boxer bobbing and weaving", "a person standing with arms down resting", "a person talking to the camera", "a person stretching", "a person walking", "a person doing squats".

**Probe.**
* Targets: labelled-punch windows are positives. Gap windows (no overlap, at least 8 frames from any punch) are negatives. Ambiguous windows are not used.
* Features are z-scored with the training videos' statistics.
* Model: logistic regression with balanced class weights and L2 (C = 3e-4), fitted by full-batch L-BFGS. The objective is the same as scikit-learn's; see `score.fit_softmax`.
* For held-out video k, the probe trains on the other 9 videos only.
* The `type` / `hand` extras come from a separate LOVO 7-way probe (C = 1e-3).

## 3. Probe selection (LOVO grid, 28 configurations)
| Config | mean per-video AUC | pooled AUC |
|---|---|---|
| **Large, C = 3e-4, binary (chosen)** | **0.930** | **0.881** |
| Large, C = 1e-4, binary | 0.930 | 0.882 |
| Large, C = 1e-3, binary | 0.928 | 0.871 |
| Base+Large, C = 1e-3, binary | 0.926 | 0.846 |
| Large, C = 1e-3, 7-way | 0.919 | 0.872 |
| Base, C = 1e-3, binary | 0.916 | 0.834 |
| Large, C = 1e-3, binary, per-video standardised | 0.908 | 0.822 |
| first run: Large, C = 0.1, 7-way | 0.888 | 0.775 |
| X-CLIP embedding, C = 0.1 | 0.804 | 0.701 |

* **Strong L2 was the main gain.** With weak L2 the probe learned video identity: its probability scale varied wildly across videos and the global threshold collapsed to 0.000.
* Per-video standardisation did not help.
* Picking the best of about 30 LOVO configurations is a mild selection bias. The top five are within 0.004 AUC of each other.

## 4. Validation (labelled-punch vs gap windows)
Columns:
* **AUC**: against all gap windows. **AUC in-span**: only gap windows between the first and last labelled punch.
* **rec@x**: share of labelled-punch windows scoring at least x.
* **gap > t95(video)**: gap windows above the threshold that catches 95% of that video's labelled-punch windows.
* **gap > t95(global)**: the same with one global threshold (t = 0.056), with window-level and punch-level recall. A punch counts as found if any window holding at least 50% of it is above t.

Full tables for all detectors: `validation.md`.

| video | punch win | gap win (in span) | AUC | AUC in-span | rec@0.3 | rec@0.5 | rec@0.7 | gap > t95(video) | gap > t95(global) | window recall at global t | punch recall at global t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | 5348 | 5367 (1904) | 0.910 | 0.892 | 0.78 | 0.47 | 0.26 | 1159 (22%) | 1664 (31%) | 0.99 | 1.00 |
| V2 | 650 | 7668 (1174) | 0.903 | 0.985 | 1.00 | 0.99 | 0.90 | 1417 (18%) | 4927 (64%) | 1.00 | 1.00 |
| V3 | 2148 | 5187 (2560) | 0.978 | 0.962 | 0.96 | 0.89 | 0.55 | 303 (6%) | 1524 (29%) | 0.99 | 1.00 |
| V4 | 1526 | 6671 (2415) | 0.928 | 0.963 | 1.00 | 0.99 | 0.89 | 1081 (16%) | 3120 (47%) | 1.00 | 1.00 |
| V5 | 1823 | 992 (909) | 0.952 | 0.947 | 0.74 | 0.45 | 0.20 | 224 (23%) | 375 (38%) | 0.98 | 1.00 |
| V6 | 2493 | 6357 (5746) | 0.834 | 0.816 | 0.28 | 0.10 | 0.03 | 2930 (46%) | 1485 (23%) | 0.76 | 0.91 |
| V7 | 636 | 5813 (2682) | 0.945 | 0.973 | 0.98 | 0.94 | 0.86 | 923 (16%) | 2666 (46%) | 1.00 | 1.00 |
| V8 | 730 | 4450 (2945) | 0.925 | 0.949 | 0.76 | 0.58 | 0.37 | 1005 (23%) | 1078 (24%) | 0.96 | 0.99 |
| V9 | 464 | 4756 (2065) | 0.980 | 0.959 | 0.87 | 0.74 | 0.59 | 351 (7%) | 699 (15%) | 0.99 | 1.00 |
| V10 | 569 | 5710 (3780) | 0.942 | 0.930 | 0.37 | 0.19 | 0.07 | 918 (16%) | 518 (9%) | 0.82 | 0.95 |

| detector | mean AUC | V1..V10 |
|---|---|---|
| `videomae_zs` | 0.716 | 0.82 0.76 0.81 0.85 0.47 0.71 0.57 0.54 0.83 0.79 |
| `xclip_zs` | 0.726 | 0.84 0.63 0.88 0.70 0.69 0.68 0.72 0.57 0.75 0.78 |
| `videomae_probe` | **0.930** | 0.91 0.90 0.98 0.93 0.95 0.83 0.95 0.93 0.98 0.94 |

* **Ranking within a video is good; calibration across videos is only fair.** V6 and V10 punches score lower; V2 and V4 gaps score higher. Use per-video thresholds or ranks.
* **V6 is weak because of its labels.** It is annotated with hooks and uppercuts only, but the boxer also throws jabs and crosses, which sit in gap windows. Its labelled uppercuts are also small, tight-guard movements.
* **The gap counts are not false-alarm rates** (see §5 and §6). Workouts are dense with punching: V1 has 46% labelled-punch windows. Gap windows inside the annotated span score high more often than those outside it, so missed punches sit between labelled combos, not only at the edges.
* **Type guess (extra keys): balanced accuracy 0.236**, chance 0.167. Treat it as a weak hint and use the skeleton classifier for type.

## 5. Qualitative QC (`videodet/qc/`)
Each row shows the full frame with the crop drawn, plus 8 crop frames.

**`videomae_probe_top_gap.png`: highest-scoring gap windows**
* V1 f35556 / f35844: real bag punches after the last annotation (f35529).
* V2 f18808 / f18856: crosses outside the annotated span (6675 to 14577).
* V3: "SIT UP 1-2-1-2", punches thrown from a sit-up.
* V4 f5924 / f29120: hooks outside the span.
* V5 (scores 0.56 / 0.65): footwork with small hand moves. Plausible false alarms or feints.
* V6 f25736: a jab (V6 jabs and crosses are unlabelled by design).
* V7: bag hooks.
* V8 f624: an intro b-roll shot of **a different person and gym**, a real punch but the wrong person; f20696 is doubtful.
* V9: crosses.
* V10: possible small uppercuts.

**`videomae_probe_low_punch.png`: lowest-scoring labelled windows**
* V1 f9116: a crossfade transition. V2 f8624: a camera cut from a close-up.
* V3, V6, V9: small tight-guard punches, or punches hidden behind the bag.
* V4 (scores about 0.3): visible hooks, just under-scored.
* V7 f4976: the punch falls at the very end of the window.
* V8 f13588, V10 f4604 / f18772: the clip looks like walking, turning or squatting. Likely label or timing errors.

`missed_candidates.csv` merges gap windows above each video's t95 into 1,630 ranked events. Many are long unlabelled punching segments rather than single punches.

## 6. VLM second opinion (Qwen3-VL-8B-Instruct)
**Setup.**
* Same crops as the detector, at 336 px; 8 frames (every 2nd) given as an ordered image sequence.
* Question: does the person throw a punch (jab, cross, hook or uppercut, at a bag or in the air)? Guard, footwork, bouncing, slipping, blocking and resting do not count. Answer Yes or No.
* Score: P(Yes) from the Yes/No next-token logits (the Yes+No probability mass was 1.000 for every window).

**Selection** (`select_vlm.py`), per video: 25 top-scoring gap windows, 15 random gap windows, 10 lowest-scoring labelled windows, 10 random labelled windows.

| group | n | VLM Yes | mean detector score |
|---|---|---|---|
| random labelled punch | 100 | 56% | 0.58 |
| lowest-scoring labelled punch | 100 | 29% | 0.12 |
| **top-scoring gap** | 250 | **64%** | 0.80 |
| random gap | 150 | 13% | 0.12 |

Per video (top gap / random gap / random labelled / low labelled):

| video | top gap | random gap | random labelled | low labelled |
|---|---|---|---|---|
| V1 | 96% | 20% | 80% | 30% |
| V2 | 88% | 60% | 50% | 40% |
| V3 | 80% | 20% | 100% | 60% |
| V4 | 96% | 7% | 80% | 70% |
| V5 | 8% | 0% | 60% | 40% |
| V6 | 84% | 13% | 60% | 20% |
| V7 | 88% | 13% | 80% | 20% |
| V8 | 48% | 0% | 20% | 10% |
| V9 | 32% | 0% | 20% | 0% |
| V10 | 20% | 0% | 10% | 0% |

* The VLM itself separates random labelled from random gap windows with AUC 0.84. Among gap windows, the detector score predicts the VLM's Yes with AUC 0.85.
* **Conclusion.** In V1, V2, V3, V4, V6 and V7 the detector's top gap windows look like punches to an independent model as often as labelled punches do (80 to 96%). These are mostly **missed annotations**.
  * V2's random gap windows are 60% Yes: its labels cover only about 7% of frames.
  * V5's top gaps (8%) look like false alarms (footwork).
  * V8, V9 and V10 are inconclusive, because the VLM rarely says Yes even to labelled shadow-boxing punches there (10 to 20%).

## 7. Compute
| Job | Resource | Wall time |
|---|---|---|
| smoke test (V10, 2,000 windows) | 1 GPU | 1.4 min |
| `vd-ext-a` (V1, V6, V8, V9, V10) | 1 GPU | 32.3 min |
| `vd-ext-b` (V2, V3, V4, V5, V7) | 1 GPU | 20.7 min |
| `vd-vlm3` (600 windows) | 1 GPU | 132 min |
| 3 cancelled VLM attempts (OpenCV seeking; a batching test) | 1 GPU | about 10 min |
| `vd-probe-grid2` | 16 CPUs, no GPU | 23.8 min |
| scoring, probes, QC | Mac CPU | about 30 min |

**Total: about 3.3 GPU-hours**, at most 2 concurrent jobs.
* HPC directory: `/l/users/<user>/punch_videodet`. Videos were copied server-side from `punch_video3d` and SHA-256-checked.
* Conda env `punchvid`; gauge `~/.claude/train-progress-videodet.json`.
* `hpc.py` now excludes `videodet/raw/` and `videodet/cache/` from its `--delete` push.
* Incidents: no Lustre incident.
  * The login node was overloaded (ssh took about 3 minutes); jobs were unaffected.
  * A 32-CPU job hit `QOSMaxCpuPerUserLimit`; it was rerun with 16.
  * The first Qwen download stalled and was restarted.
  * Qwen3-VL ran at about 13 s/window. Time goes into the model forward pass, not decoding or preprocessing, and batching didn't help. The cause is undiagnosed.

## 8. Limitations
* **Labels cap every number here.** The probe was trained against noisy gap negatives, so it partly learns **the annotators' coverage**: it is pushed to score unlabelled punches low.
  * That makes its high-scoring gap windows more credible as missed punches.
  * But some missed punches are scored low and will not be found.
  * For a "no punch" class, use **low** scores plus agreement with the skeleton detector, not "gap" alone.
* Calibration across videos is imperfect, so choose thresholds per video or work from ranks.
* There is one crop per window. Windows spanning a camera cut (V2, V9) or a transition (V1) are scored on mixed content. Intro or b-roll shots of other people (V8) can score as punches.
* The main subject is the largest person box, with no identity tracking. Other people appear in V1 (68% of frames) and in V9/V10 (23 to 29%); spot checks found the right boxer.
* Punch type from pixels is weak (balanced accuracy 0.236).
* The VideoMAE weights are **CC-BY-NC-4.0**. A commercial app needs a different backbone, then a rerun of `extract.py` and `score.py`.

## 9. How to load the scores
```python
import numpy as np
d = np.load("detect/scores/videomae_probe/V3.npz")   # one file per dataset video V1..V10
d["start"]   # int32 (n,), window start frame (0-based, original video, WINDOWS.md); covers start..start+15
d["score"]   # float32 (n,), LOVO P(punch) from the binary probe (model never saw this video)
d["p_type"]  # float32 (n, 6), weak type probabilities, columns = d["type_classes"]
d["type"], d["hand"]   # argmax type; "lead"/"rear" derived from it
```
* `videomae_zs` and `xclip_zs` have only `start` and `score`. Their scores are monotone but are not calibrated probabilities.
* `detect/scores/<name>/meta.json` records the settings and pooled metrics.
* Window labels: `videodet/windows.py`.

**Reproduce:**
1. HPC: `HPC_REMOTE=/l/users/<user>/punch_videodet python hpc.py --env punchvid --sweep sweeps/videodet_extract.txt`
2. Copy `runs/vd-ext-*/feats` to `videodet/cache/feats/`.
3. `python videodet/score.py --kind large --C 3e-4 --head binary`
4. `python videodet/qc.py`, then `python videodet/candidates.py`
5. `python videodet/select_vlm.py`, then `hpc.py --env punchvid videodet/vlm_check.py videodet/vlm_selection.json`
6. Copy `vlm_answers.json` to `videodet/`, then run `python videodet/vlm_summary.py`.
