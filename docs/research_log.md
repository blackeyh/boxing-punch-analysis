# Boxing Punch Classifier: Project Log

A running record of goals, decisions, experiments, predictions vs. results, and open questions.
Newest experiments are appended at the bottom of **Experiment log**.

---

## 1. Goal

Classify a single boxing punch from a 2D skeleton clip into 6 classes, as the first step toward a **real-time punch-recognition app**.

**Success criterion (set 2026-11-12):** at least **60% balanced accuracy** under **leave-one-video-out** evaluation.
- *Balanced accuracy* = mean recall per class, so every class counts equally and a model can't score well by ignoring the rare Rear Hook.
- *Leave-one-video-out (LOVO)* simulates the real app: the model is always tested on a video (person, camera, lighting) it never saw.
- A random 80/20 split is reported only for comparison. It is optimistic because train and test clips share videos.

---

## 2. Data

| Item | Description |
|---|---|
| `Skeleton_data/V*.npy` | `(N, 25, 17, 2)`: one clip per labelled punch, padded to 25 frames, COCO-17 keypoints, (x, y) normalised to [0, 1]. Row *i* matches annotation row *i*. |
| `Skeleton_data/V6.npy` | Different format: `(46497, 1, 17, 3)`, a continuous frame stream with the frame number in channel 0. Clips are cut using the annotations. |
| `Annotation_files/V*.xlsx` | Start frame, end frame (inclusive), punch type. The layout differs per file. |
| `RGB_videos/Meta_data.ods` | YouTube links for V1–V10 only. No video files. |

**After cleaning (`punch_data.py`):** 5,428 clips from 10 videos.

| Class | Clips |
|---|---|
| Cross | 1,369 |
| Jab | 1,281 |
| Lead Hook | 1,031 |
| Lead Uppercut | 683 |
| Rear Uppercut | 676 |
| Rear Hook | 388 |

Clips per video: V1 1863, V2 232, V3 810, V4 559, V5 594, V6 677, V7 195, V8 199, V9 148, V10 151.
Median clip length is 10 real frames. V6 has only hooks and uppercuts.

**Data issues found and handled:**
- Inconsistent spreadsheet layouts. The loader auto-detects the start/end/label columns.
- Label spelling variants (`cross`, `Jab `, `Lead hook`, `Rear uppercut`). Normalised with strip + title-case.
- 20 broken annotations (zero or negative length, or longer than 25 frames). Dropped.
- Overlapping annotations (mostly V6 and V7). Kept, not yet investigated.

---

## 3. Code

| File | Purpose |
|---|---|
| `punch_data.py` | `load_dataset()` → `X (N,25,17,2)`, `y`, `video_id`, `lengths`. Run it directly for a summary. |
| `baselines.py` | `normalise()`, `mirror()`, `to_orthodox()`, `resample()`, `time_features()`, `evaluate()` harness, and all baseline experiments. Run `/opt/anaconda3/bin/python baselines.py`, or `... baselines.py B4` to run only experiments whose name contains "B4". |
| `mem_guard.py` | **Import before torch in any GPU script.** MPS memory cap + RSS watchdog. |
| `deep.py` | 1D-CNN and LSTM (manual, leak-free) on MPS, wrapped as sklearn estimators so `evaluate()` scores them like the RF. `python deep.py [cnn] [lstm] --seeds 0 1 2`. |
| `progress.py` | Training progress → `~/.claude/train-progress.json`, shown as a live gauge in the Claude Code status line (`~/.claude/statusline-train.sh`, refreshed every 2 s). `baselines.py` ticks per fit; `deep.py` ticks per epoch. **Every training script must use it.** |
| `hpc.py` | **Run any script on the HPC GPU from here.** Pushes code + data (one-way), submits Slurm GPU job(s), syncs `runs/<run>/` back every 5 s, feeds the status-line gauge. `python hpc.py deep.py cnn --seeds 0` or `python hpc.py --sweep sweeps/<file>.txt` (parallel jobs, one combined gauge). |
| `sweeps/*.txt` | Sweep definitions: one `name \| command` per line. |
| `runs/<run>/` | Complete local copy of every HPC run: `output.log`, `slurm.log`, `results.json`, `progress.json`, `job.sh`, `meta.json` (job id, node, state). |
| `lovo_predictions.py` | Per-clip leave-one-video-out predictions (CNN, seed-averaged) → `runs/lovo_predictions/preds.npz`. |
| `review/` | Error-review page: `template.html` + `export_review_data.py` → `punch_review.html` (published as an artifact). |
| Git / GitHub | **Private repo https://github.com/blackeyh/boxing-punch-recognition** (branch `main`, started 2026-11-12, first commit `bc17828`). Everything is versioned: code, data, `runs/` results, this log. Commit after every experiment or decision. `.gitignore` skips only `__pycache__/`, `*.tmp`, `.DS_Store`, `.claude/`. `hpc.py` never pushes `.git/` to the HPC. |
| `seed_check.py` | Reruns B2 and B3b across 5 seeds to measure model-randomness noise. |

---

## 4. Key decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-12 | Prove a simple classifier can work before building anything else | Feasibility first: if it can't separate punches, the app doesn't matter. |
| 2026-11-12 | Primary metric: LOVO balanced accuracy, target 60% | Honest about new users; robust to class imbalance. |
| 2026-11-12 | Start from the simplest baselines, changing one thing at a time | Each step is a controlled experiment; we learn *which* idea helps. |
| 2026-11-12 | Baselines ignore time order (flattened random forest) | Accepted limitation: a baseline only needs to be honest and comparable. |
| 2026-11-12 | Normalisation: hip midpoint → (0,0), divide by the clip's median torso length | Removes position and size; torso length stays stable when the fighter turns, unlike shoulder width. |
| 2026-11-12 | Use `per_frame` centring | Tied with `first` (57.1 vs 56.5); per_frame is simpler. |
| 2026-11-12 | **Orthodox only:** mirror southpaw video V4 to orthodox (stance canonicalisation) | The app can ask the user their stance once. Learning both stances (B3) hurt accuracy. |
| 2026-11-12 | **Parked:** automatic stance detection | Seen as a research problem to revisit later (see section 6). |

---

## 5. Experiment log

All experiments use a random forest (200 trees, `random_state=0`) on flattened clips (850 features), single seed.

### Summary

| ID | Change | Random 80/20 | **LOVO mean** | LOVO worst |
|---|---|---|---|---|
| B0 | Always predict the majority class | 16.7% | 15.7% | V6 0.0% |
| B1 | Raw image coordinates | 91.2% | 46.5% | V4 9.5% |
| B2 | + normalisation, per_frame | 91.6% | 57.1% | V4 2.0% |
| B2' | + normalisation, first | 90.6% | 56.5% | V4 1.8% |
| B3 | B2 + mirror augmentation (both stances) | 89.4% | 50.0% | V10 29.7% |
| B3b | B2 + V4 mirrored to orthodox | 90.9% | 63.1% ✅ | V4 38.2% |
| B4a | B3b + resample to 10 frames | 92.5% | 63.5% | V4 36.1% |
| B4b | B4a + duration | 92.3% | 62.8% | V4 37.5% |
| B4c | B4b + velocity | 91.4% | 66.8% ✅ (5 seeds: 66.6 ± 0.3) | V4 41.0% |
| **D-CNN** | 1D-CNN on B4c inputs as a sequence (MPS) | 94.9% | **71.8% ± 1.1** (3 seeds) | V4 44% |
| **D-LSTM** | LSTM, same inputs | 94.4% | **72.4% ± 1.6** (3 seeds) | V4 48% |
| S1-base | CNN on HPC (CUDA), 5 seeds | 94.9% | 72.7% ± 0.8 | V4 50% |
| S1 variants | + duration / − face / class-weight / augment | 95.1–95.3% | 71.8–73.0% (all within noise) | V4 ~50% |
| S3 ST-GCN | skeleton graph network (5 seeds) | 94.8% | 70.6% ± 1.0 | V4 52% |
| S3 ST-GCN + aug | + augmentation (5 seeds) | 94.6% | 73.0% ± 0.8 (ties CNN) | V4 51% |
| S4 CNN + view-aug | random horizontal squash (5 seeds) | 95.0% | 71.7% ± 1.2 | V4 49% |
| S4 CNN + frontal | frontalness input channel (5 seeds) | 95.2% | 72.1% ± 0.6 | V4 51% |

### B0/B1: first baselines
- **Prediction:** a large gap between random and LOVO.
- **Result:** 91.2% random vs 46.5% LOVO. Random splits mostly measure recognising the video. 46.5% is about 3× chance, so the task is feasible.
- V4 scored **below chance** (9.5%), meaning it was systematically wrong. Flagged for investigation.
- Rear Hook recall was 7% (fewest examples, confused with Rear Uppercut and Lead Hook).

### B2: normalisation
- **Predictions (user):** at least 60% LOVO; much less overfitting; `first` wins; V4 recovers.
- **Result:** 57.1% LOVO (+10.6). The gap shrank 45 → 34 points. `first` and `per_frame` tied. **V4 got worse (2%).** Without V4, the other 9 videos average 63.3%.
- **Lesson:** normalisation helps but removes only position and size. V4's problem lies in the body shape itself.

### V4 investigation
- Watching the videos (V8 best, V4 worst) revealed nothing visible.
- **V4 confusion matrix:** Jab → Cross 111/122, Cross → Jab 38/39, and rear punches → lead punches. A systematic lead/rear swap.
- **Wrist test** (which wrist travels more during jabs): left in every video except **V4 (right: 0.69 vs 0.44)**. V9 is ambiguous (0.29 vs 0.32).
- **Conclusion:** V4 is a southpaw fighter or a mirrored video. The fix is the same either way.
- **Lesson:** a better model can't fix this; it's a data problem. The data revealed what watching the video couldn't.

### B3: mirror augmentation
- **Predictions (user):** V4 above 50%; average above 60%.
- **Result:** V4 rose 2 → 39%, but the **average fell to 50.0%**. V9 dropped 33 points and V10 26.
- **Lesson:** with both stances in training, "left hand moved" no longer means lead. The model has to infer which hand is in front from subtle cues, which a flattened random forest can't. **Negative result: mirroring is the wrong fix for this model.**

### B3b: stance canonicalisation (orthodox only)
- **Result:** **63.1% LOVO, goal met.** Every video improved over B2; V9 rose 71 → 83%, V4 2 → 38%.
- **Caveats:** single seed, only 3 points above the target. V4 was identified as southpaw using labels (simulating "the user states their stance").
- **Remaining errors are about path, not hand:** Jab → Lead Hook 27%, Lead Uppercut → Lead Hook 34%, Rear Hook → Cross 41% / Rear Uppercut 33%. That's time-dependent information the random forest can't see.

### Seed check: is 63.1% real? (`seed_check.py`)
- Reran B2 and B3b with 5 random-forest seeds (0–4). The LOVO folds are fixed, so only model randomness changes.

| Config | Random 80/20 | **LOVO mean** | LOVO worst |
|---|---|---|---|
| B2 | 90.9% ± 1.2 | 57.3% ± 0.5 (56.5–57.8) | 2.2% ± 0.7 |
| **B3b** | 91.1% ± 0.9 | **63.2% ± 0.3 (62.8–63.4)** | 37.5% ± 0.6 |

- **Result:** every seed is above 60%. The B2 → B3b gain (+5.9) is about 12× the seed noise, so it's real.
- This also confirms that the B2 `first` vs `per_frame` gap (0.6) was noise.
- B3b per video (mean over seeds): V1 44, V2 58, V3 64, V4 38, V5 67, V6 58, V7 72, V8 90, V9 83, V10 58.
- **Important caveat:** seeds measure *model* randomness, not *data* uncertainty. Across the 10 videos, scores range 38–90% (std ≈ 16 points), so the uncertainty in the mean is about ±5 points (std/√10). A truthful statement: **"about 63%, plausibly anywhere from the low 50s to the low 70s on new videos."** The 60% goal is met on this data, but not by a comfortable margin. More videos would tighten this more than any model change.

### B4: time alignment (resampling, duration, velocity)
- **Why:** the random forest ignores time order, and punches of different lengths put the same phase in different columns.
- **What:** on orthodox data (B3b), after normalisation:
  - **resample** each clip's real frames to 10 frames by linear interpolation, so frame k = the same phase in every punch and padding disappears;
  - **duration** = the original real-frame count, added back as one feature;
  - **velocity** = differences between consecutive resampled frames.
- Tested separately, one addition at a time (B4a → B4b → B4c). Predictions were skipped this round.

| ID | Features | Random 80/20 | **LOVO mean** | LOVO worst |
|---|---|---|---|---|
| B3b | normalised, padded (reference) | 90.9% | 63.1% | V4 38.2% |
| B4a | + resample (340 features) | 92.5% | 63.5% | V4 36.1% |
| B4b | + resample + duration (341) | 92.3% | 62.8% | V4 37.5% |
| **B4c** | **+ resample + duration + velocity (647)** | 91.4% | **66.8%** | V4 41.0% |

- **Seed check, B4c:** **66.6% ± 0.3** over 5 seeds. The +3.4 gain over B3b (63.2 ± 0.3) is real.
- **Resampling alone ≈ no change on average (+0.4)**, but it reshuffled videos: V5 64 → 82, V1 44 → 54, while V9 83 → 68 and V10 58 → 46. Aligned positions help some videos and hurt others.
- **Duration ≈ no effect** (−0.7, noise level). Punch length doesn't carry much information beyond what the shape already shows.
- **Velocity gives the real gain (+4.0)**, with 8 of 10 videos at or above their B4a score (V2 −1, V5 −2). The biggest gains are V10 +10 and V9 +8. Explicit movement direction is what the forest was missing.
- Per class (B3b → B4c recall): Jab 63 → 61, Cross 72 → 72, Lead Hook 63 → 70, **Rear Hook 12 → 15**, Lead Uppercut 43 → 56, Rear Uppercut 59 → 72.
- **Remaining confusions:** Rear Hook → Cross 52% and → Rear Uppercut 29% (Rear Hook is now the main problem); Lead Uppercut → Lead Hook 33%; Jab → Lead Hook 28%.
- **Lesson:** aligning time is only useful once the model can see *change*. Positions at matched phases weren't enough; differences between phases were.

### D1: first deep-model run CRASHED THE MAC (2026-11-12, 17:37)
- Ran `deep.py` (1D-CNN + LSTM, 3 seeds, MPS) in the background.
- **The Mac kernel-panicked** (watchdog timeout). The panic report shows the training process (`python3.12`, pid 38763) using **46.7 GB resident memory on a 36 GB machine**, with the memory compressor at 100% of its segment limit and 40 swap files. The system froze until the watchdog rebooted it.
- The data is tiny (about 15 MB of features) and the models are small, so ~47 GB means **memory accumulated across fits** (MPS/GPU memory not released, or a leak). Root cause not yet confirmed.
- No results were saved; the output file was lost in the reboot. The project files were unaffected.
- **Rule from now on:** every GPU run uses a hard memory cap (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`) plus a memory monitor that kills the run above a threshold. Test memory behaviour on a few fits before any long run.

**Diagnosis (same day, with the guards in place):**
- Added `mem_guard.py`, imported before torch: MPS cap of 0.3 × 28 GB ≈ 8.4 GB (plus the required low watermark of 0.2), and a watchdog thread that kills the process if RSS > 10 GB.
- **1D-CNN:** memory flat at about 1.04 GB across fits. Not the cause.
- **`nn.LSTM` on MPS:** within the first 5-epoch fit, 7.5 GB of "other allocations" built up, and the cap stopped it with an error (the guard worked).
- Step-level test: `nn.LSTM` driver memory **grows in jumps, even during inference only** (1.0 → 2.8 GB over 200 steps), and is **never returned**, even by `torch.mps.empty_cache()`. Each test started where the previous one ended. A full run (~85k LSTM steps) explains 47 GB.
- **Root cause:** a driver-level memory leak in PyTorch's built-in LSTM on MPS (torch 2.9.0). It's invisible to `current_allocated_memory()`, which showed 0 GB.
- **Fix:** replaced `nn.LSTM` with `ManualLSTM` (same LSTM equations, built from `nn.Linear` plus a loop over the 10 frames). Also added `del` and `torch.mps.empty_cache()` after every fit.
- **Verified:** 6 × 20-epoch fits each of the LSTM and CNN → MPS driver memory flat at 1.05–1.07 GB, RSS around 0.7 GB.
- Speed with 60 epochs: LSTM ≈ 30 s per fit, CNN ≈ 6 s per fit, on MPS.
- **Lesson:** "GPU memory" on Apple Silicon is system RAM. A leak doesn't just crash the script; it can take down the whole machine. Cap first, measure on a short run, then go long.

### D2: 1D-CNN and LSTM on MPS (guarded rerun, `deep_run1.log`)
- Same data as B4c (orthodox, normalised, resampled to 10 frames), fed as a sequence: 10 frames × 68 channels (34 positions + 34 velocities).
- 60 epochs, AdamW, one-cycle LR, unweighted cross-entropy, 3 seeds. LSTM = `ManualLSTM` (leak-free). Memory stayed flat; no problems.

| Model | Random 80/20 | **LOVO mean (3 seeds)** | Seed 0 worst | Time per seed |
|---|---|---|---|---|
| B4c RF (reference) | 91.4% | 66.6% ± 0.3 | V4 41% | ~1 min (CPU) |
| **D-CNN** | 94.9% | **71.8% ± 1.1** | V4 44% | ~1 min (MPS) |
| **D-LSTM** | 94.4% | **72.4% ± 1.6** | V4 48% | ~5.3 min (MPS) |

- **Both deep models beat the RF by +5–6 points**, well above the seed noise. Reading frames in order helps.
- **CNN vs LSTM: a tie** (gap 0.6 points, stds 1.1–1.6). The CNN is about 5× faster, so it's the better default for iterating.
- Per video (seed 0, CNN / LSTM): V1 58/62, V2 58/66, V3 65/69, V4 44/48, V5 91/90, V6 74/67, V7 84/91, V8 91/91, V9 95/99, V10 59/58.
- Per class recall (RF B4c → CNN → LSTM, seed 0):
  - Jab 61 → 56 → 55
  - Cross 72 → 66 → 62
  - Lead Hook 70 → 77 → 82
  - **Rear Hook 15 → 41 → 40**
  - Lead Uppercut 56 → 71 → 71
  - Rear Uppercut 72 → 68 → 68
- **Rear Hook finally works partly** (15 → 41%), and Lead Uppercut gains 15 points. Jab and Cross dropped slightly.
- **Remaining confusions:** Jab → Lead Hook 29% (both lead-hand; the jab/hook split is the new main weakness), and Rear Hook → Cross 34–36%.
- Seed-to-seed spread is larger than for the RF (±1.1–1.6 vs ±0.3), as expected for neural nets on little data.

### HPC setup (2026-11-12)
- **Principle (user's rule):** this Mac folder is the single source of truth. The HPC is compute only: code and data are pushed one-way, nothing is edited there, and every output is synced back to `runs/`. No need to ever log in to the HPC.
- Cluster: MBZUAI HPC via `ssh hpc` (login node lo-01). Partition `ws-ia`: 116 workstations, each with 1× **NVIDIA RTX 5000 Ada**, 48 CPUs, ~230 GB RAM. Work dir `/l/users/<user>/punch`.
- **`punch` conda env** on the HPC, pinned to the Mac's versions: Python 3.12.7, numpy 1.26.4, pandas 2.2.2, scikit-learn 1.7.2, torch 2.9.0 (+cu128), openpyxl 3.1.5, psutil 5.9.0 (`hpc_setup.log`).
- `deep.py` picks the device automatically: cuda (HPC) > mps (Mac) > cpu.
- Cluster quirk: `sacct` (accounting DB) is down, so `hpc.py` reads the final state from `scontrol`, falling back to the `exit code:` line in `slurm.log`.
- **Smoke test** (`runs/smoketest`, job 295060): CNN, 2 epochs, 1 seed → ran on ws-l2-001 (CUDA), all files synced back. Pipeline verified.

### Step 1: targeted CNN input experiments (HPC sweep, running)
- `sweeps/step1_cnn_inputs.txt`: 5 parallel GPU jobs, CNN, **5 seeds** each (effects expected to be small, so more seeds).
- One change each vs the CNN baseline:
  - `--duration`: log(real frame count) as an extra channel. Targets Jab → Lead Hook (speed).
  - `--no-face`: drop nose/eyes/ears (68 → 48 channels). Removes likely noise.
  - `--class-weight`: balanced loss. Targets Rear Hook (smallest class).
  - `--augment`: random body scale 0.9–1.1 plus 0.02 joint jitter, velocities recomputed. Targets overfitting.
- Predictions skipped (user's choice).
- **Ran:** jobs 295066–295070 on ws-l3-014/021/004 and ws-l1-019, about 3 min each (~34 s per seed on an RTX 5000 Ada). Only ~2 jobs ran at once (likely a per-user limit). Results in `runs/s1-*/results.json`.

**Results (CNN, 5 seeds each, LOVO = mean of per-video balanced accuracy):**

| Variant | **LOVO mean ± std** | min–max | Random 80/20 |
|---|---|---|---|
| base | 72.7% ± 0.8 | 71.9–74.0 | 94.9% |
| + duration | 71.8% ± 1.0 | 70.8–73.1 | 95.1% |
| − face | 73.0% ± 0.5 | 72.4–73.6 | 95.2% |
| class-weight | 72.8% ± 0.7 | 72.1–73.8 | 95.1% |
| augment | 72.9% ± 0.6 | 72.1–73.8 | 95.3% |

- **None of the four changes moves the LOVO mean beyond seed noise** (all within ±1 point of base). Duration is slightly worse (−0.9, about 1 std). The HPC CNN base (72.7, 5 seeds) is consistent with the Mac MPS run (71.8 ± 1.1, 3 seeds).
- **Class weighting did not help Rear Hook** (42 → 43%). The problem is not class imbalance: Rear Hook genuinely looks like Cross (28%) and Rear Uppercut (20%) in this data.
- **Duration did not fix Jab → Lead Hook** (28 → 26%).
- **Removing face keypoints changes *which* errors happen without changing the average:**
  - pooled recall: Jab 55 → **69%**, Cross 65 → **73%**;
  - per video: V1 +6, V2 +9, V3 +5, but V10 −8, V7 −8, V6 −4.
  - Pooled recall is dominated by V1 (1,863 clips), so it looks like a big win there while the per-video average stays flat. **Lesson: always check the metric the goal is defined on.** Face keypoints help some videos (camera angles?) and hurt others.
- **Per-video spread dominates everything:** V4 ~50%, V10 ~55%, V1 ~60% vs V5/V8/V9 above 90%. That's a 40-point range between videos, against ±1-point effects from the input tweaks.
- **Conclusion: input tweaks have plateaued for this CNN at about 73%.** The remaining error is concentrated in specific videos and specific class pairs, which points to the data (labels, camera angles, stance) rather than the model. **Error analysis (step 2) is now the priority.**

### Step 2: error analysis (in progress, user reviewing)
- `lovo_predictions.py`: per-clip LOVO predictions from the CNN baseline, probabilities averaged over seeds 0–2 (MPS, ~3 min) → `runs/lovo_predictions/preds.npz`.
  - LOVO mean with seed-averaged probabilities: **73.2%** (ensembling 3 seeds ≈ +0.5 over single seeds).
  - Per video: V1 62, V2 56, V3 62, **V4 45**, V5 92, V6 74, V7 91, V8 93, V9 99, V10 58. **1,868 of 5,428 clips wrong.**
- **Review page:** https://claude.ai/artifact/4HgjSxV3DpoQHPhgrjninY (private).
  - Source: `review/template.html`; data is packed by `review/export_review_data.py` → `review/punch_review.html` (2.7 MB, all clips embedded).
  - Shows animated skeletons (lead side red, rear side blue, wrist paths traced), a confusion matrix filter per video, the most confident mistakes first, side-by-side confidently correct examples from the same video, and the model's probabilities.
  - Verdicts per clip ("label is wrong" + suggested label / "can't tell in 2D" / "model is wrong" + note) are saved in the artifact's database, collection `verdicts`, doc id `c<clip index>`. Claude reads them back with ArtifactData.
- **Why confident mistakes first:** a clip the model is very sure about yet "wrong" is the best candidate for a wrong label.
- **User's review (2026-11-12):** "mostly the model is the one wrong". 3 verdicts saved (clips 165, 211, 223: V1 Jab → Lead Hook at 100% confidence, all "model is wrong"). **So labels look mostly fine; the errors are the model's.**
- **Camera-angle check (quick, not followed up; user moved on to ST-GCN):** frontalness = shoulder width / torso length on the first frame.
  - Jab → Lead Hook rate by frontalness quintile (most side-on → most frontal): **47% → 34% → 30% → 21% → 9%**; Jab accuracy 33% → 78%.
  - V1 (frontalness 0.20): 61% of jabs called Lead Hook; V4 (0.21): 42%; the most frontal videos (V5/V9/V10, ~0.32+): 0–4%.
  - Likely mechanism: seen side-on, a jab moves across the image like a hook does from the front. **Viewpoint is a strong, unmodelled factor.**
  - Ideas for later: view-angle normalisation (rotate the 2D skeleton to frontal), a frontalness input feature, or view-mixing augmentation.

### Step 3: skeleton graph network, ST-GCN (HPC sweep, running)
- `deep.py stgcn`. Same 68-channel input as the CNN, regrouped per joint: (batch, 4 channels [x, y, vx, vy], 10 frames, 17 joints).
- 4 ST-GCN blocks (64, 64, 128, 128 channels). Each block:
  - **graph conv over joints:** a joint mixes its own features with its bone neighbours', using the fixed COCO skeleton graph (normalised A + I) **plus a learned joint-to-joint matrix B** (starts at 0), so links between unconnected joints (e.g. the two wrists) can be learned;
  - then a **temporal conv over 3 frames**, BatchNorm, dropout, and a residual connection.
- Then average over time and joints → linear layer. 194k parameters (CNN: 126k).
- Local check: joint mapping verified (left wrist → joint 9), MPS memory flat at 0.04 GB over 3 fits, ~0.7 s per epoch.
- `sweeps/step3_stgcn.txt`: `s3-stgcn` (plain) and `s3-stgcn-aug` (with augmentation), 5 seeds each. Reference: s1-base CNN, 72.7% ± 0.8.
- **Incident:** the local sync loop crashed mid-run. rsync exit 24 ("file has vanished": `progress.tmp` was replaced while being copied). The HPC jobs were unaffected.
  - Fixes in `hpc.py`: live pulls skip `*.tmp` and tolerate rsync codes 23/24; an ssh failure during the queue check is no longer mistaken for "jobs finished"; new `--attach RUN...` re-attaches the sync to submitted jobs.
  - Re-attached; both jobs COMPLETED (jobs 295092/295094, ~7.5 min each). All files recovered.

**Results (5 seeds each, same GPU type):**

| Model | **LOVO mean ± std** | Random 80/20 | Seconds per seed |
|---|---|---|---|
| CNN (s1-base) | 72.7% ± 0.8 | 94.9% | 34 |
| CNN + aug | 72.9% ± 0.6 | 95.3% | 37 |
| ST-GCN | 70.6% ± 1.0 | 94.8% | 87 |
| **ST-GCN + aug** | **73.0% ± 0.8** | 94.6% | 89 |

- **The plain ST-GCN is worse than the CNN (−2.1, ~2 std).** With augmentation it gains +2.4 and **ties the CNN** (73.0 vs 72.7–72.9). The graph network overfits more without augmentation (more parameters, little data).
- **The graph structure did not raise the ceiling.** Every model since D2 (CNN, LSTM, ST-GCN, all variants) lands at **71–73%**. That's a strong sign the limit is the data/representation (viewpoint, 10 videos), not the architecture.
- ST-GCN trades errors rather than reducing them: Jab 55 → 62% and Cross 65 → 68%, but Rear Hook 42 → 35% and Lead Hook 81 → 76% (CNN → ST-GCN + aug).
- Per video, V1 (55–60), V4 (~50) and V10 (~55) stay low for **every** model, and they're among the most side-on videos (V1 0.20, V4 0.21 frontalness). This supports the camera-angle finding.
- ST-GCN is ~2.5× slower per seed than the CNN for no gain. **The CNN stays the default.**

### Step 4: camera-angle experiments (HPC sweep, running)
- **Real-time question (user):** how would a live app know how side-on the fighter is?
  - Answer: from the pose itself. Frontalness = (left shoulder x − right shoulder x) / torso length, measured on the guard frame just before the punch (punches rotate the shoulders).
  - It's distance-independent, costs a few operations per frame, and the camera is usually fixed per session, so it could even be calibrated once.
  - Caveats: the CNN already receives the shoulder positions, so it could in principle compute this itself; 2D can't separate some angles; pose estimators get worse side-on.
- Median |frontalness| per video: V1 0.18, V2 0.18, V3 0.19, V4 0.19, V5 0.31, V6 0.28, V7 0.21, V8 0.25, V9 0.34, V10 0.32.
- `--frontal`: frontalness (signed, first frame) added as a 69th input channel.
- `--view-aug`: in half the training clips, all x-coordinates are squashed by a random factor 0.4–1.0 (positions are hip-centred), and velocities recomputed. A crude stand-in for a more side-on camera; true rotation needs depth, which 2D doesn't have.
- Not combined: squashing would contradict the frontalness channel.
- `sweeps/step4_viewpoint.txt`: CNN, 5 seeds each. Reference: s1-base 72.7% ± 0.8.
- **Ran:** both COMPLETED in ~3 min (ws-l5-008, ws-l5-004). No sync problems after the hpc.py fixes.

| Variant | **LOVO mean ± std** | Jab → Lead Hook | V1 | V4 |
|---|---|---|---|---|
| CNN (reference) | 72.7% ± 0.8 | 28% | 60% | 50% |
| + view-aug | 71.7% ± 1.2 | 31% | 53% | 49% |
| + frontal | 72.1% ± 0.6 | 25% | 57% | 51% |

- **Neither helps.** View-aug is slightly worse (−1.0; V1 −7). Frontal is within noise (−0.6; Jab → Lead Hook 28 → 25% is a small, unconfirmed hint).
- **Why view-aug fails:** squashing x doesn't simulate a side view. Turning the camera moves a jab's forward (depth) motion INTO the image's x-axis, which a squash can never create without depth. It only shrinks lateral motion, making the side-on/frontal mismatch no better.
- **Why frontal adds nothing:** the CNN already receives both shoulder positions; telling it the ratio adds no new information.
- **Caveat on the camera-angle finding itself:** the frontalness ↔ error link was measured pooled across videos, and frontalness is mostly a per-video property (V1–V4 side-on, V5/V9/V10 frontal). So it's partly confounded with everything else that differs between videos (fighter, style, annotator). The viewpoint is a plausible cause, not a proven one.
- **Conclusion:** cheap 2D tricks can't fix viewpoint. The real fixes need information we don't have in 2D:
  - **3D pose lifting:** estimate depth from the 2D skeleton (e.g. VideoPose3D / MotionBERT-style lifters), then rotate every skeleton to face the camera;
  - **more videos**, especially side-on ones;
  - **RGB video features.**
- The ~73% plateau now holds across 4 architectures and 9 input/training variants.

### Step 5: 3D (two parallel routes, started 2026-11-12)
**Route A: 3D from the original videos** (background agent, separate worktree `../Dataset_video3d`, branch `video3d`, HPC dir `punch_video3d`, own progress bar `train-progress-video3d.json`). Task: download the 10 videos, run a 3D pose estimator (MediaPipe by default), verify frame alignment with the existing annotations (V6's stored frame numbers as ground truth), and produce a drop-in 3D dataset plus a report. Results go to `video3d/REPORT.md` on that branch.

**Route B: 3D lifting of our existing 2D skeletons (here)**
- Tooling made safe for two parallel workstreams: `hpc.py` reads `HPC_REMOTE`/`GAUGE_PATH` (the code push mirrors with `--delete`, so a shared dir would wipe the other's files); the status line shows one bar per `~/.claude/train-progress*.json`.
- Model: VideoPose3D COCO-input checkpoint (`lift3d/SOURCES.md`). Re-implemented in `lift3d/vp3d_model.py` (running the cloned code was blocked by the safety classifier).
- `lift3d/lift.py`: lifts all 5,428 orthodox clips → `lift3d/lifted_h36m.npz` (N, 25, 17, 3).
- `lift3d/features.py`: root-centred, torso-scaled 3D; **`rotate_to_frontal`** turns each clip about the vertical axis so its first-frame hip line faces the camera; positions + velocities → 102 channels.
- `deep.py --input {2d, lift, lift-rot}`. `evaluate()` and `results.json` now also save **per-clip LOVO predictions** (`lovo_pred`), so error rates can be broken down by camera angle.
- Tests passed: pipeline runs end to end with random weights (receptive field 243, root = 0, padding stays 0); the rotation puts the hip line on +x for any heading, preserves distances and leaves height unchanged; the 3D input trains through the CNN; the 2D path still works.
- **BLOCKED on the checkpoint:** Meta's download server's TLS certificate expired 2026-11-11; the insecure download was blocked by the safety classifier; no mirror found. Waiting for the user's decision.
- Planned once unblocked: lift (minutes, on the Mac) → HPC sweep: CNN on `lift` vs `lift-rot`, 5 seeds, vs the 2D CNN (72.7%); then Jab → Lead Hook by frontalness quintile (was 47% side-on vs 9% frontal).

**Checkpoint obtained (2026-11-12 19:41):** the user ran the insecure download themselves. SHA-256 `d3219e005b50591f694da5cbaf6849f060d6b2cf895864a779f8a992ac63a232`, 67.9 MB. Loaded with `weights_only=True`; keys `epoch, lr, model_pos`; loads into our re-implementation with **strict=True** (16.95M params, receptive field 243), confirming both the file and the architecture.

**3D sanity checks:**
- Bone lengths are realistic and stable: upper arm 0.25 m, forearm 0.22 m, thigh 0.43 m, torso 0.47 m, under 3% variation within a clip.
- **Problem found: clip-mode lifting smooths punches away.** Jab lead-wrist travel was only 0.05–0.07 m. Cause: clips (3–25 frames) edge-padded into a 243-frame window are ~95% frozen pose.
- **Fix: per-frame mode** (`lift.py --mode frame`, now the default): each frame is repeated over the window. Travel becomes realistic (Jab 0.25–0.32 m, Lead Hook 0.41–0.46 m). Cost: frame-to-frame jitter (forearm length variation 10%). Lifting all 54k frames takes 1.7 min on MPS. The old output is kept as `lift3d/lifted_h36m_clipmode.npz`.
- **Warning sign:** after rotation, jabs look broadly similar side-on vs frontal, but **Lead Hooks don't** (frontal: mostly sideways x −0.37 m; side-on: mostly depth z −0.26 m). The lifter is imperfect side-on, so the rotation only partly removes the camera angle.
- **Step 6 sweep:** `sweeps/step6_lift3d.txt` (CNN, 5 seeds, `--input lift` vs `--input lift-rot`). Both COMPLETED (~3 min each).

| Input (CNN, 5 seeds) | **LOVO mean ± std** | Jab → L Hook | J → LH by frontalness quintile (side-on → frontal) | Rear Hook recall |
|---|---|---|---|---|
| 2D (reference) | 72.7% ± 0.8 | 28% | (47 / 34 / 30 / 21 / 9% from the ensemble model) | 42% |
| 2D + foreshort | 72.2% ± 0.9 | 26% | 39 / 33 / 27 / 18 / 13% | 38% |
| **3D lifted** | **70.0% ± 0.7** | 25% | 32 / 32 / 28 / 22 / 11% | 43% |
| **3D lifted, rotated to frontal** | **70.4% ± 0.6** | 28% | 43 / 34 / 29 / 21 / 11% | **54%** |

- **Lifted 3D is worse than 2D (−2.3 to −2.7 points), and the rotation adds nothing (+0.4, noise).** The side-on Jab → Lead Hook gradient is unchanged.
- One real signal: **Rear Hook 42 → 54% with rotated 3D** (Lead Uppercut fell 67 → 57%, Jab 55 → 51%). Depth seems to help separate the rear hook from the cross, while lifter noise hurts other classes.
- Per video: V1 60 → 64% (the most side-on video improves slightly), but V10 55 → 51%, V7 88 → 83%.
- **Why it underperforms:** (1) per-frame lifting adds jitter (10% bone-length variation) that 2D doesn't have; (2) the lifter is least accurate side-on, where depth is needed (the hook directions still differed side-on vs frontal after rotation); (3) the rotation angle comes from one noisy first-frame hip line; (4) the lifter never saw boxing.
- **Conclusion:** single-frame VideoPose3D lifting on these short clips does not beat 2D. Depth from a learned prior on 2D dots seems too unreliable here. The video route (real image evidence) remains the stronger bet.

**Step 7: follow-ups** (`sweeps/step7_2d_plus_3d.txt`, both COMPLETED ~3 min)

| Input (CNN, 5 seeds) | **LOVO mean ± std** | Jab → L Hook by frontalness quintile (side-on → frontal) | Jab | Rear Hook |
|---|---|---|---|---|
| 2D (reference) | 72.7% ± 0.8 | (rerun with per-clip predictions pending: step 7b) | 55% | 42% |
| 3D rotated | 70.4% ± 0.6 | 43 / 34 / 29 / 21 / 11% | 51% | 54% |
| 3D rotated, smoothed (3 frames) | 70.1% ± 0.9 | 38 / 26 / 25 / 21 / 13% | 56% | 56% |
| **2D + 3D rotated** | **72.5% ± 0.5** | **28 / 27 / 24 / 15 / 10%** | **60%** | 45% |

- **Smoothing doesn't help** (70.1 vs 70.4). Jitter isn't the main problem.
- **2D + 3D ties 2D on the average (72.5 vs 72.7)**, but it has the **flattest side-on gradient seen so far**: the most side-on jabs are misread as Lead Hook 28% of the time, vs 39–43% for 2D + foreshort / 3D alone. Jab recall 55 → 60%, V1 60 → 65%. Losses elsewhere (V5 −3, V10 −3, Lead Uppercut −6) cancel the gain.
- Needs a like-for-like 2D reference: s1-base predates per-clip predictions, so **step 7b** reruns the 2D CNN with them (`sweeps/step7b_2d_reference.txt`).

**Step 7b result: confirmed.** The 2D rerun with per-clip predictions (`s7-2d-ref`, job on ws-l1-007): LOVO **72.8% ± 0.7** (matches s1-base, 72.7).

| (CNN, 5 seeds) | LOVO | Jab → Lead Hook by frontalness quintile (side-on → frontal) | Most side-on quintile, per seed | Jab recall |
|---|---|---|---|---|
| 2D | 72.8% ± 0.7 | **47** / 36 / 29 / 17 / 11% | 49, 48, 49, 52, 38 | 55% |
| 2D + 3D rotated | 72.5% ± 0.5 | **28** / 27 / 24 / 15 / 10% | 34, 26, 23, 32, 23 | 60% |

- **Adding rotated lifted 3D cuts the side-on Jab → Lead Hook confusion from 47% to 28%.** Consistent across all 5 seeds (2D 38–52%, 2D + 3D 23–34%; no overlap). Frontal jabs are unaffected (11 → 10%). **This is the first change that actually fixes the camera-angle problem**, and it confirms the diagnosis: the side-on errors come from missing depth.
- **But the overall average doesn't move**, because the 3D channels cost elsewhere (Lead Uppercut 67 → 61%, V5 −3, V10 −3). Net: the gain is real and targeted, while the lifter's noise hurts classes that 2D already handled well.
- **Next ideas:** make the model rely on 3D only where it helps (e.g. a late-fusion / two-branch model, or 3D dropout in training), or use better 3D (the video route) so the noise penalty shrinks.

**Side test while waiting: foreshortening** (`--foreshort`, `sweeps/step5_foreshort.txt`, job on ws-l3-018, 3 min)
- 8 extra channels: each limb's (upper arms, forearms, thighs, shins) on-screen length ÷ its longest length in the clip. A limb pointing at the camera looks short, so this is a crude per-frame depth cue that costs nothing in real time.
- Result: **72.2% ± 0.9 vs 72.7% ± 0.8. No gain.** Jab → Lead Hook 28 → 26%, Rear Hook 42 → 38% (both within noise); per video, all within ±5 points.
- Jab → Lead Hook by frontalness quintile (side-on → frontal): 39 / 33 / 27 / 18 / 13%. The side-on gradient is still there. (The earlier 47 / 34 / 30 / 21 / 9% came from the seed-ensembled `lovo_predictions` model, so it's not an exact like-for-like comparison.)
- **Lesson:** hand-made 2D depth cues don't help. Like frontalness, the CNN can already derive limb lengths from the joint positions it sees. **Real depth has to come from outside the 2D skeleton** (a lifter's learned prior, or the video pixels).

### Video3D: 3D pose from the original videos (branch `video3d`, 2026-11-12)
- **Goal:** get real 3D (depth) for every clip to attack the viewpoint problem from step 4, without relabelling.
- **Done:** downloaded all 10 YouTube videos (720p), ran MediaPipe Pose Landmarker (heavy) on every frame on the HPC (CPU job, `punch3d` env, `HPC_REMOTE=/l/users/<user>/punch_video3d`), aligned to the existing 2D, and built `Skeleton3D_data/` + `punch_data3d.load_dataset_3d()` (rows asserted identical to `load_dataset()`).
- **Result:** **all 10 videos align at frame offset 0** (median 2D disagreement 7-12 px on 1280x720, sharp at +-1 frame, no drift). V6's stored frame numbers confirm the method. 99.9% of annotated frames usable.
- **Metadata errors found:** `Meta_data.ods` links for **V2 and V3 are swapped**, and the **V1 link is the wrong video**; the true source is `kbgkeTTSau8` (same channel/gym), found by scanning the channel.
- **V4 confirmed a genuine southpaw** (not a mirrored video).
- **Caveat:** side-on videos have many low-visibility wrists/elbows (V1 48% of wrists); 3D arm lengths vary 8-16% within a clip (depth noise on arms); legs/torso 2-5%.
- Full details: `video3d/REPORT.md`.
- **Main-thread verification before merging (2026-11-12):**
  - Branch pushed at `9c96f6b`. `load_dataset_3d()` returns all 5,428 rows (3D, re-extracted 2D, confidence) and its row-match assertion against `load_dataset()` passes. The 3D is finite and zero-padded correctly.
  - MediaPipe arm depth is noisier than the lifter's: forearm length varies 14.3% within a clip (lifted: 10%).
  - **Merged into `main`** (`f3465db`). Conflicts in `.gitignore` and this log were resolved by keeping both sides; `hpc.py` auto-merged (per-workstream env vars + `--attach` + the agent's `--no-gpu`/`--env`/raw exclusion).
  - The `Meta_data.ods` errors (V2/V3 swapped, wrong V1 link) are documented, not edited; the correct sources are in `video3d/manifest.json`.
- **Incident: sync loops freezing after the laptop slept (root cause found, 2026-11-12 21:43).**
  - `hpc.py` shares one ssh "master" connection (ControlMaster). The one created at wake-up (20:35) went dead but its socket stayed, so every later ssh/rsync joined it and hung forever. That's what froze the s8b sync loop: the job itself **completed normally** (68.4% ± 2.4); I first misread the stale local copy as a second hang.
  - The first s8 run (job 295389) really did hang on the node (process alive, 0% GPU for 30 min) and was cancelled. Its cause is unknown, but the identical rerun completed, so it was most likely a node-side glitch.
  - Fixes: every ssh/rsync call now has a time limit (600 s; 120 s for queue checks), and ssh uses keep-alives (`ServerAliveInterval=15`, `ServerAliveCountMax=3`, `ConnectTimeout=20`), so a dead connection fails in ~45 s instead of hanging. Stall detection (no progress for 10 min → warning + STALLED in the bar). Recovery: kill the `[mux]` master process and remove `~/.ssh/cm-*`.
  - Mistake to avoid: stopping a sweep's sync loop while it is still submitting leaves later jobs unsubmitted (it happened to step 9: 3 of 5 submitted; the rest resubmitted as step 9b). Check the log for all "submitted" lines first.
- **Incident 2: HPC storage lock (2026-11-12, ~21:40–22:15).**
  - Step-9 jobs s9-2d and s9-lift-rot sat "RUNNING" for 30+ min, frozen at 0% GPU. Kernel wait channels on the compute nodes: `cl_sync_io_wait` and `ldlm_completion_ast` (Lustre client I/O wait / distributed-lock grant). The jobs were blocked on Lustre locks in our `/l/.../punch/runs/` folders.
  - On login node lo-01: 9 leftover `rsync --server` / `tail` processes of ours from the dead sync connections. The user killed them; the lifted-3D job briefly moved (to seed 2), then froze again.
  - One process could not be killed: **PID 4070116, state D**, `rsync --server` writing into `runs/s9-mp3d-rot-fill-conf-20261112-214045/` (a run folder from the interrupted step-9 submission). An unkillable process stuck in Lustre needs the HPC admins.
  - **Action:** all 5 step-9 HPC jobs cancelled; step 9 rerun locally on the Mac (`sweeps/step9_local.sh` → `runs/s9-*-local/`).
  - **For HPC support:** lo-01, user <user>, PID 4070116 in D state since 21:41 on 2026-11-12, rsync into `/l/users/<user>/punch/runs/`; compute-node processes waiting in `ldlm_completion_ast` / `cl_sync_io_wait`.
  - **Planned prevention:** jobs write to node-local disk and copy to `/l` once at the end, so the sync never reads files a running job is writing; use a fresh `HPC_REMOTE` folder for new runs until support clears the stuck process.
- **Step 8:** `lift3d/features.py` now supports the COCO layout (MediaPipe); V4 is mirrored to orthodox in 3D (verified: its jabs then come from the left wrist, 0.47 vs 0.32 m). New `deep.py --input mp3d / mp3d-rot / 2d+mp3d-rot`. Sweep `sweeps/step8_mediapipe3d.txt` (CNN, 5 seeds each).

### Step 9: MediaPipe confidence handling + 2D/3D ensembles (run LOCALLY, MPS, 2026-11-12 22:14–22:39)
- `sweeps/step9_local.sh` (HPC unusable, see incident 2). CNN, 5 seeds each; per-clip probabilities saved. Analysis: `analysis/confidence_report.py` → `runs/step9_confidence_report.txt`.
- Ensembles average per-clip LOVO probabilities per seed (each probability is from a model that never saw that clip's video, so still a fair LOVO test).

| Model | **LOVO ± std** | Side-on jab → L Hook | Rear Hook |
|---|---|---|---|
| 2D (local rerun) | 72.0% ± 0.9 | 45.8% | 39.8% |
| lifted 3D rot | 70.5% ± 1.3 | 41.8% | 53.1% |
| MediaPipe 3D rot (step 8, no fill) | 57.9% ± 2.2 | 32.9% | 58.2% |
| 2D + MediaPipe 3D rot (step 8, no fill) | 68.4% ± 2.4 | 43.5% | 39.6% |
| **MP 3D rot + conf-fill** | **72.5% ± 0.4** | **21.6%** | 41.1% |
| MP 3D rot + conf-fill + conf input | 71.4% ± 0.7 | 21.2% | 41.3% |
| **2D + MP 3D rot + fill + conf** (best single model) | **73.7% ± 0.6** | 22.3% | 45.2% |
| Ensemble 2D + lifted 3D | 73.8% ± 1.0 | 48.1% | 48.0% |
| Ensemble 2D + MP 3D (fill + conf) | 74.8% ± 1.1 | 30.8% | 41.8% |
| **Ensemble 2D + lifted + MP 3D** (best overall) | **75.2% ± 0.7** | 42.6% | 48.9% |

- **Confidence fill rescued MediaPipe 3D: 57.9 → 72.5%**, and seed spread 2.2 → 0.4. Only 5.4% of joint-frames are replaced, and average arm jitter didn't drop, yet it fixed training. The low-confidence joints were rare but **extreme outliers** that derailed the CNN; averages hid this.
- **MediaPipe 3D (filled) is the best camera-angle fix so far:** side-on jab → Lead Hook 45.8% (2D) → **~22%**.
- **New best single model: 2D + filled MediaPipe 3D + confidence, 73.7%** (+1.7 over 2D in the same local setup).
- **New best overall: ensemble of 2D + lifted 3D + MediaPipe 3D, 75.2% ± 0.7** (+3.2 over 2D). The models' errors are complementary; averaging their probabilities beats every single model.
- Confidence as an extra input on its own slightly hurts (71.4 vs 72.5); combined with 2D it's part of the best single model.
- Note: the local 2D rerun scored 72.0% vs 72.8% on the HPC (CUDA vs MPS numerics plus seed noise); all step-9 comparisons use the local runs.

**Confidence of misclassified clips (MediaPipe visibility, 0–1; "wrong" = wrong in most seeds):**
- **Before** (2D): wrong clips have wrist confidence **0.60** vs **0.69** for right ones; arms 0.73 vs 0.78; all joints 0.87 vs 0.90.
- **After** (best models): **the same gap remains**: wrists 0.59–0.60 (wrong) vs 0.68 (right).
- Accuracy by wrist-confidence quintile (lowest → highest):
  - 2D: 54 / 60 / 61 / 67 / 83%
  - Ensemble 2D + lifted + MP 3D: 61 / 65 / 63 / 69 / 84%
  - Ensemble 2D + MP 3D: 57 / 65 / 67 / 73 / 84%

  Gains are mostly in the low/mid-confidence clips (+3 to +7 points); high-confidence clips were already ~83%.
- Share of misclassified clips with wrist confidence < 0.5: 2D 38.6% → best ensemble 36.7% (all clips: 29.9%).
- **Conclusion:** errors stay concentrated where wrists are hard to see. The fixes improve those clips too, but low visibility remains the main limit. Occluded wrists are an information problem a better classifier can't fully solve.

### Step 10: punch DETECTION (no-punch class), started 2026-11-13
- **Gap noticed by the user:** the classifier can't say "no punch". It only ever saw clips known to be punches. A real-time app needs detection (is a punch happening?) as well as classification.
- **User's plan:** use a detector offline to prepare the data, then add a "no punch" class.
- **User's concern:** the gaps between labelled punches surely contain **unlabelled punches**, and a simple rule-based detector (wrist speed) can't be trusted ("if a rule was enough…"; risk of calling every movement a punch). So: **two independent learned detectors** whose agreement builds the dataset, with the user reviewing disagreements and a random sample.
- **HPC hardened first:** jobs write only to node-local disk while running, publish small snapshots to `/l` every 30 s (copy + rename), and copy everything at the end; new HPC dir `punch2`. Verified (job 296531).
- **Shared window definition:** `detect/WINDOWS.md` (16-frame windows, stride 4, every frame of every video; output format).
- **Detector A: video pixels** (background agent, worktree `../Dataset_videodet`, branch `videodet`, HPC dir `punch_videodet`, own progress bar). Surveys video–text / vision-language / action models, scores all windows, validates against labels, reviews high-scoring gaps.
- **Detector B: skeleton PU detector (here).**
  - `detect/build_frames.py` → `detect/frames/V{v}.npz`: all 312k frames (tracked person, V4 mirrored, conf-filled 3D, per-frame labels). Cross-checked against the clip data: median difference 0.0001. Labelled punches cover only 6.4–40.3% of each video's frames (V2, V9, V10 under 7%).
  - `detect/pu_detector.py`: non-causal dilated TCN (receptive field ~65 frames) over the frame stream, 119 channels per frame (frontal-rotated 3D, velocities per second at 25 fps, visibility). **Positive–Unlabelled learning** (Elkan & Noto): train "labelled vs unlabelled", calibrate with c = mean score on held-out labelled frames, P(punch) = g / c, hidden punches ≈ sum of P over unlabelled frames. Leave-one-video-out: each video scored by a model that never saw it.
  - Caveat: Elkan–Noto assumes labelled punches are a random sample of all punches (SCAR). Annotators probably labelled whole segments, not random punches, so treat the hidden-punch estimate as rough.
  - Full run started (10 folds × 2,500 steps, local MPS) → `detect/scores/pu_skeleton/`, `runs/pu_detector/pu_summary.json`.
  - **Result (detector B):** AUC labelled-punch vs gap windows per video: V3 0.95, V5 0.94, V9 0.96, V7 0.90, V8 0.90, V4 0.86, V2 0.85, V6 0.85, **V1 0.70, V10 0.73**. At P(punch) ≥ 0.5 it catches 52–92% of labelled punch windows and flags 6–32% of gap windows. Labelled windows with no detected person: 0% (so misses aren't tracking gaps). The Elkan–Noto hidden-punch estimate (~3,000 vs 5,251 labelled) is **not trusted**: SCAR is violated (annotators labelled stretches, not random punches) and small c (V1 0.20, V10 0.31) inflates it.
  - **Verdict:** useful as one vote and as a ranking of which gaps to inspect, not as the judge.
- **Gap review page (published 2026-11-13):** https://claude.ai/artifact/Bx6xbiF7QKvsGxRp8y7Sr5 (private).
  - `detect/review/build_review.py` picks 28 non-overlapping moments per video (280 total): 12 highest detector-B gap windows, 6 nearest P = 0.5, 8 random gap windows, and 2 **blind control** labelled punches. Video crops (16 frames, 160 px, around the tracked fighter) → `detect/review/sprites/` (git-ignored, rebuildable). Moments are shuffled.
  - Visual check: control crops show the right fighter and the labelled punch; skeleton overlay aligned.
  - The page shows the looping video crop (slow motion, optional skeleton overlay). **Detector scores and groups are hidden from the reviewer.** Verdicts: 6 punch types / no punch / unclear / wrong person (keys 1–6, N, U, W) → artifact database collection `gap_verdicts`, doc id `V{v}_{start}`. When detector A arrives, rerun `build_review.py` to add its high-scoring and disagreement windows.
  - **v2 (user feedback):** (1) clips now include 8 context frames before and after (32 shown, the judged 16 highlighted on a timeline, context dimmed), so punches cut by the window edge can be judged; (2) **the user judges only punch / no punch** ("the AI is better than me at the type"): verdicts are now `punch` (Y), `no_punch` (N), `partial` = punch cut off at the edge (C), `unclear` (U), `bad_view` (W). `partial` windows will be excluded from both the punch and the no-punch training sets. Same 280 moments and order (verified), so any earlier verdicts still apply; an earlier typed verdict (e.g. `Jab`) counts as `punch`.
- **First review results (2026-11-13, 79 of 280 moments; copy in `detect/review/gap_verdicts.json`):**

| Group (blind to the reviewer) | Reviewed | Punch | No punch | Cut off | **Punch rate** |
|---|---|---|---|---|---|
| Known labelled punches (controls) | 7 | 7 | 0 | 0 | **100%** |
| Detector B confident punch | 32 | 19 | 11 | 2 | **59%** |
| Detector B unsure (~0.5) | 20 | 9 | 11 | 0 | 45% |
| Random unlabelled moments | 20 | 3 | 17 | 0 | **15%** |

  - **The review is reliable:** all 7 blind controls were called punches. Detector B scored 4 of those 7 below 0.5 (0.00, 0.05, 0.34, 0.44), i.e. it missed real punches.
  - **Detector B alone is not trustworthy, as the user suspected:** precision 59% (41% of its "hidden punch" calls are not punches), recall 71% on reviewed moments.
  - **Hidden punches are real:** about 15% of random unlabelled moments contain a punch (3/20; small sample, 95% CI roughly 3–38%). Raw gaps used as "no punch" would be ~85% clean, which is not good enough without filtering.
  - Next: more random moments to tighten the hidden-punch rate; detector A (video) to see whether it fixes B's false alarms.
- **Interim check of detector A's zero-shot scores against the user's verdicts** (77 reviewed moments with a clear punch/no-punch verdict: 38 / 39; detector A's trained probe not finished yet):

| Detector | AUC vs user verdicts | Top half flagged: precision / recall |
|---|---|---|
| B skeleton (PU) | 0.70 | 64% / 66% |
| A: VideoMAE zero-shot | 0.65 | 56% / 58% |
| A: X-CLIP zero-shot | 0.64 | 59% / 61% |
| **B + VideoMAE (rank average)** | **0.77** | |
| B + X-CLIP | 0.73 | |

  - **B and A are almost uncorrelated** (rank correlation 0.01–0.02): they make independent errors, so combining them helps (0.70 → 0.77). This supports the two-detector design.
  - Caveat: the reviewed moments were chosen using B's score (top / borderline / random), so this sample is not neutral between the detectors; the random group alone is too small to compare yet.
- **Detector A's trained probe vs the user's verdicts** (same 77 moments; the agent was still tuning probe settings, so its final scores may differ):

| Detector | AUC | At score ≥ 0.5: flagged / precision / recall |
|---|---|---|
| B skeleton (PU) | 0.70 | 46 / 59% / 71% |
| **A VideoMAE probe (LOVO)** | **0.86** | 20 / **95%** / 50% |
| A VideoMAE zero-shot | 0.65 | 13 / 54% / 18% |
| A X-CLIP zero-shot | 0.64 | 53 / 58% / 82% |
| B + A probe (rank average) | 0.84 | |

  - **A's probe is clearly the best detector so far:** when it says punch, the user agreed 19 of 20 times. It's conservative (catches half of the punches the user saw).
  - Combining with B doesn't beat A's probe alone (0.84 vs 0.86); rank correlation B vs A probe 0.39.
  - **Agreement:** both say punch → 14 moments, user said punch in 13 (93%). Neither says punch → 25 moments, but **5 (20%) were punches**: both detectors miss some hidden punches, so "both say no" is not yet clean enough for the no-punch class. Disagree → 38 moments, 20 punches.
  - Note: A's probe treats gap windows as negatives (no PU correction), yet scores well. Its precision on hidden punches suggests the pixels carry much stronger evidence than the skeleton.
- **No-punch contamination by threshold** (user's 77 verdicts): "no punch" = A < 0.5 still contained ~33% punches in the B-selected hard moments, but **on the random (unbiased) moments A was perfect**: the 3 real punches scored 0.92–0.99, the 17 non-punches ~0. A strict rule **A < 0.05 AND B < 0.2** → 1 of 15 (7%) punches even among hard moments.
- **Decision (user, 2026-11-13): the detectors are reliable enough; no more reviewing for now.** Build "no punch" with the strict rule; hidden punches = A flags (95% precision).
- **7-class dataset** (`detect/build_7class.py` → `detect/dataset7/`, using A's probe scores as of 18:21; the agent was still tuning; rerun with the final scores):
  - **All classes cut from the same MediaPipe frame data** (otherwise the model could learn the data source instead of punch vs no punch). The punch clips are identical to the existing MediaPipe clips (diff 0).
  - No-punch rule: gap window (≥ 8 frames from labelled punches), person present ≥ 90%, A < 0.05 and B < 0.2, ≥ 8 frames from any A or B flag (≥ 0.5); one clip per window with a length drawn from that video's punch lengths; at most as many as the video's punches.
  - Result: 5,428 punch clips + **3,468 no-punch clips** (fewer for V1: 414 and V5: 83, where little footage is clean). 30 clips had a missing-person frame, filled from the nearest frame.
  - **Hidden-punch windows (A ≥ 0.9, overlapping): 6,017**, concentrated in **V2 (2,157)** and **V10 (1,099)**, the least-labelled videos (< 7% of frames). Listed in `hidden_punches.json`, not trained on yet.
- **Training running** (`detect/train7.py`, local MPS): CNN, 5 seeds, LOVO, input MediaPipe 2D + MediaPipe 3D + visibility (187 ch). 7-class (with no punch) and a 6-class reference on the same MediaPipe-only input. Metrics: balanced accuracy, punch-type accuracy, miss rate (punch → no punch), false-alarm rate (no punch → punch).
- **Results (5 seeds each, LOVO, local MPS; `runs/train7-7class/`, `runs/train7-6class/`):**

| Model (MediaPipe 2D + 3D input) | Punch-type accuracy | Missed punches | False alarms |
|---|---|---|---|
| 6-class reference (punches only) | **73.6% ± 1.6** | – | – |
| **7-class (6 punches + no punch)** | **73.1% ± 0.9** | **12.0% ± 0.5** | **3.2% ± 0.3** |

  - **MediaPipe-only input is as good as the dataset's original 2D:** 73.6% vs 73.7% (step 9, dataset 2D + MediaPipe 3D). A live app can use MediaPipe end to end without losing accuracy.
  - **Adding "no punch" costs nothing on punch types** (73.1 vs 73.6, within noise).
  - Misses by video: V10 36%, V2 24%, V1 21%, V4 17%, V6 8%, V3 6%, V5 3%, V8 2%, V7 1%, V9 1%. The worst are the **least-labelled videos (V2, V10)** and the **most side-on ones (V1, V4)**.
  - Misses by punch type: Lead Uppercut 24%, Cross 14%, Rear Uppercut 14%, Rear Hook 13%, Jab 11%, Lead Hook 8%.
  - False alarms by video 1–7% (V1 highest at 7%).
  - Caveat: false alarms are measured on the strict-rule (clean) no-punch clips; on raw live footage (feints, footwork, guard resets) they will likely be higher.
  - Ideas for the misses: add the confirmed hidden punches (A ≥ 0.9) as extra punch training data, especially V2/V10; check whether V10's labelled punches are lighter/shorter than the others.
- **Hidden punches defined** (user asked): a gap stretch (≥ 8 frames from any labelled punch) where detector A scores ≥ 0.9. The 6,017 overlapping windows merge into **~1,335 events** (V10 341, V2 267, V7 205, V8 168, V4 162, V1 90, V9 78, others ≤ 13); V2/V4 events average ~43 frames, i.e. likely combinations. Type and exact boundaries are unknown (A's type guess is near chance).
- **Decision (user agreed): don't train on hidden punches** (risk of copying A's systematic errors, unknown boundaries, limited upside). Use them as a test set instead (not yet run).

### Step 11: real-time simulation with a rendered video (2026-11-13)
- `detect/realtime_demo.py`: trains the 7-class model **without the demo video**, then streams through it like a live camera. Every 2 frames it classifies the **last** 8, 12 and 16 frames (past only) and averages them. P(punch) = 1 − P(no punch), smoothed over 3 steps. A punch **starts above 0.6 and ends below 0.4** (at least 2 steps), and counts once with the type summed highest. Output: an annotated MP4 (skeleton, live call, per-type counters, 3-second probability trace, annotators' labels underneath, and a flag when a counted punch has no label). Rules fixed in advance, not tuned on the demo video.
- **Speed: 0.4 ms per decision** on the Mac (3 window sizes), against 12 decisions per second needed: **~200× faster than real time** for the classifier. Pose estimation is the real cost live.
- **First run (V9, frames 3400–7715, 3 min, 51 labelled punches):** 50 of 51 overlapped by a detection, but only **33 counts**. **Combinations were merged** (P(punch) stays high through a jab–cross), so most counts covered 2 labelled punches.
- **Fix: split a detection** where the punching hand switches (lead ↔ rear, sustained 2 steps) or at a clear dip (≥ 0.15 below both neighbouring peaks). Matching is now one-to-one.
- **Second run (V9):** **65 counts vs 51 labelled; 50 of 51 matched (1 missed); type correct 39/50 (78%); 15 extra counts.**
  - Extras: 8 are separate moments, of which 5 have strong detector-A support (0.69–0.99, likely hidden punches) and 3 weak (0.37–0.57, likely false alarms). 7 touch a matched punch with A 0.94–0.99, mostly alternating uppercuts: either over-splitting or a real second punch the annotators labelled together. **To be judged by watching the video.**
- Video: `runs/realtime_V9/realtime_V9_3400-7715.mp4` (15.6 MB, git-ignored, rebuildable).

### Step 12: two fighters and contact (planning, 2026-11-13)
- **Video inventory:** **no video has an opponent.** V1 hits hanging heavy bags; V2–V4 hit a freestanding bag; V5–V10 are shadowboxing. On-screen text: V3/V4 show combo notation (e.g. "1-4-ROLL-6B-3"), V10 shows punch names ("Jab + Cross") → a possible free label check.
- **Data rule (user):** no external data for training; **external footage is OK for testing only.**
- **MediaPipe does not keep identities:** in frames with 2+ people, the list order swaps between consecutive frames 3–18% of the time (V10 18%, V2 17%, V3 7%, V8 3%). Identity must come from our own tracker. That needs no training: position + motion prediction + body shape + colour appearance (`build_frames.py` already tracks by nearest hip).
- **Plan:** multi-person pose → training-free tracker → per-fighter stance from geometry (lead hand = the hand nearer the opponent) → per-fighter real-time counting → annotated video. Test on **external sparring footage** (fixed camera, different clothing, 2–5 min with clinches), with the user spot-checking crossings for identity swaps. Expect lower accuracy: sparring is usually filmed side-on, the hardest view for the model.
- Waiting on: the user's choice of test footage.
- **User extended the goal:** contact detection must work for **opponents, punching bags and BOB** (human-shaped bag). Design: target-agnostic. (1) The attacker's contact signature (abrupt deceleration before full extension) is the same for any target. (2) Target location: opponent skeleton / bag found by motion + background / BOB possibly detected by MediaPipe as a partial person (to test). (3) Target reaction: head snap / bag swing / standing bag & BOB wobble. On BOB and bags, the glove position at contact also gives head vs body / high vs low.
- **Self-labelling plan with our own data:** bag reaction after each labelled punch in V1–V4 → automatic contact / no-contact labels; V5–V10 shadowboxing = no contact. These train the attacker contact model (LOVO). External BOB and sparring footage are for testing only.
- **Test footage spec written:** `detect/TEST_FOOTAGE_SPEC.md` (sparring, BOB, hanging and standing bags; fixed side-on camera, ≥ 720p, ≥ 24 fps, real speed, ≥ 2 min continuous; report format). User is collecting footage.
- **Training on external footage (user asked):** advised not with the first batch (it would destroy the only honest generalisation test, it has no labels, and it would change the user's rule). Later, only with a strict video-level train/test split and model-proposed, human-confirmed labels.
- **Bag-contact feasibility** (`detect/contact/feasibility_bag_motion.jpg`): average pixel change from 2 frames before to 6 frames after the end of a labelled punch, fighter masked out by a dilated skeleton. **V2: the standing bag's outline lights up; V1: the bag being hit glows strongly, other bags barely.** Measurable. Issues to handle: on-screen overlays (timers/logos) also change, so contrast against random moments; in V1 the fighter moves between bags, so localise the bag per time window.
- **Bag reaction labels** (`detect/contact/bag_reaction.py` → `bag_reaction.json`, QC masks in `detect/contact/qc/`): 1/8-scale grey frames; bag mask per 2-minute window = pixels changing more after punches than at baseline moments, fighter masked by a dilated skeleton; reaction z-score per punch against baseline moments.
  - v1 bugs: (1) **V1 picked a small blob near the fighter** instead of the swinging bag (blob choice weighted distance too much); (2) **baseline contaminated by hidden punches**: random gap moments in V2 also hit the bag (≈270 hidden-punch events), so only 7/230 V2 punches came out as contact.
  - v2 fixes: baseline = only **clean no-punch moments** (A < 0.05 and B < 0.2, away from labels and flags); blob = largest reacting region within reach of the fighter.
  - v2 result: **V2 229/230 contact, V3 632 contact / 14 no-contact / 164 ambiguous, V4 560/560 contact.** Masks for V2–V4 accurately outline the standing bag. **V1 masks unreliable** (in different windows: the on-screen clock, part of a bag, or the fighter's surroundings), so V1 is excluded from contact labels for now.
  - **Consequence: almost no punches in our bag videos miss the bag**, so the planned within-video test (bag hits vs air punches by the same fighter) **is not possible with our data**. A trained contact model (bag videos = contact, shadowboxing = no contact) could learn "bag workout vs shadowboxing" instead of contact, and nothing internal could reveal it.
  - **Proposal:** a **training-free physics rule** on the attacker's arm (contact = sharp wrist deceleration before full extension; miss = full extension, smooth slowdown). It can't learn a shortcut. Sanity check on V2–V4 (mostly contact) vs V5–V10 (no contact); the real test is external BOB / bag footage with misses and feints. A trained model only as a comparison, judged on external footage.
- **Physics rule tested: it does NOT work** (`detect/contact/physics_rule.py` → `physics_rule.json`). Per punch: wrist speed relative to the shoulder (m/s), stop time = 80% → 30% of peak speed after the peak, extension at stop and max extension (3D, conf-filled). Rule: contact if stop ≤ 60 ms (and, for jab/cross, extension at stop ≤ 0.85); no contact if stop ≥ 130 ms or jab/cross extension ≥ 0.93.
  - Outcome: **bag punches (V1–V4) 22% contact / 45% no contact / 34% unclear; air punches (V5–V10) 23% / 45% / 32%. Identical.**
  - Why: (1) at 24–30 fps the fist stops within 1–3 frames, so stop time takes only a few values (42/67/83/125 ms) and the bag-vs-air difference is below one frame; (2) shadowboxers also stop short to protect the elbow (jab max extension is actually higher on the bag: 1.00 vs 0.97); (3) the only consistent difference is lower peak speed on the bag (e.g. cross 4.0 vs 5.4 m/s), which reflects workout style, not contact.
  - **Conclusion: contact cannot be read from the puncher's arm at normal video frame rates. It must come from the TARGET's reaction:** bag swing/wobble (pixels; works on V2–V4), BOB wobble (pixels), opponent head/torso snap (their skeleton), plus glove-to-target proximity at the moment of the punch.

### Videodet: independent pixel-only punch detector (branch `videodet`, 2026-11-13)
- **Goal:** a second opinion from the pixels (no skeleton input) on "is a punch thrown in this window?", to (a) find punches the annotators missed and (b) build a clean "no punch" class together with the skeleton detector. Windows exactly as `detect/WINDOWS.md` (16 frames, stride 4, 77,925 windows).
- **Setup:** person-centred square crop per window (torchvision Faster R-CNN person boxes; pixels only) → VideoMAE-B/L (Kinetics-400) logits + features and X-CLIP B/32 prompt logits (HPC, own dir `punch_videodet`, env `punchvid`, 2 GPU jobs, ~53 GPU-min). Models, revisions, licences, checksums: `videodet/SOURCES.md` (VideoMAE is CC-BY-NC).
- **Results (labelled-punch vs gap windows, mean per-video ROC-AUC):** zero-shot Kinetics "punching bag/person" 0.716, X-CLIP prompts 0.726; **LOVO logistic probe on VideoMAE-L features 0.930** (pooled 0.881). Strong L2 (C = 3e-4) was the key: a weakly regularised probe (0.888) learned video identity. Scores: `detect/scores/videomae_probe/` (+ `videomae_zs`, `xclip_zs`).
- **Gap windows are not clean negatives:** QC sheets (`videodet/qc/`) show most top-scoring gap windows are real unlabelled punches (bag work outside the annotated span in V1/V2/V4, jabs/crosses in V6, whose labels contain only hooks/uppercuts, "sit-up 1-2" punches in V3). Lowest-scoring labelled windows include scene transitions/cuts (V1, V2), tight-guard uppercuts (V6, V9) and apparent walking/squatting (V8, V10: likely label errors). **Qwen3-VL-8B second opinion** (600 selected windows, 2.2 GPU-h): it says "punch" for **64% of the top-scoring gap windows**, vs 56% of random labelled punches (its own sensitivity) and 13% of random gap windows (`videodet/vlm_summary.json`). So the top gap windows are largely real, unlabelled punches.
- Punch type from pixels is weak (6-way balanced accuracy 0.236); use it for detection only.
- Ops: login node overloaded (ssh ~3 min) but no Lustre incident; a 32-CPU job hit `QOSMaxCpuPerUserLimit` (rerun with 16).

---

## 6. Open questions / parked

- **[Parked] Automatic stance detection.** Make the app work for orthodox and southpaw fighters without being told. Ideas:
  - Mirroring creates labelled southpaw data for free.
  - Calibrate from the first punches (the jab is the most common punch).
  - A stance classifier from the guard pose, then canonicalise.
  - End-to-end learning with a sequence model.
  - Handle switch-hitters per clip (V9?).
  - Evaluate on V4 plus mirrored held-out videos.
- **V4 is still only 38% after flipping.** Stance wasn't its only problem. Its clips are the shortest (median 6 frames), and its camera angle may be unusual.
- **Overlapping annotations** in V6 and V7. Do they matter?
- **Rear Hook** remains the weakest class (12% recall in B3b).

---

## 7. Next steps

1. ~~Check that B3b's 63.1% holds across several random seeds.~~ Done: 63.2% ± 0.3.
2. ~~Time resampling, plus velocity features.~~ Done: B4c 66.6% ± 0.3 (new best).
3. Rear Hook (15% recall): why does it look like a Cross? Inspect examples; consider class weighting.
4. Sequence models (1D-CNN / LSTM / skeleton graph network) to fix the straight/hook/uppercut confusion.


---

## ⏸ RESUME POINT (laptop restart, 2026-11-13 ~19:40)
**Saved:** everything in `main` is committed and pushed (this commit). Agent branch `videodet` pushed at `deddbf4`. Branch `video3d` merged.

**Built since the last result entry:**
- `external/ingest.py` + `external/extract.py`: test-footage pipeline. Self-tested on our own V9 (reproduces our skeletons, median diff 0.0003). **YouTube currently answers 403** to yt-dlp 2026.06.09: either the user updates yt-dlp (`pip install -U yt-dlp`, user's decision), or puts local file paths in `external/clips.csv`.
- `external/track.py`: training-free identity tracker (position + velocity prediction, colour histograms, limb proportions, pose; clinch mode; retroactive swap-back after clinches) + `--synthetic` self-test (V8 + V9 fighters moved on crossing paths, shuffled order, 3% dropouts, known identities). **Self-test finished: 2,880 frames (2 min), 20 crossings, 0 identity swaps (0.0%) over 2,705 frames with both fighters; 0 retroactive fixes needed.** Caveats: an easy case (V8 dark clothes vs V9 white shirt; skeletons come from separate videos, so MediaPipe never merges them during a crossing, unlike real clinches). **Too slow: 18.5 min for 2 min of video** (full-frame HSV conversion per detection). Speed it up (downscaled frame, one HSV conversion per frame) before real footage.

**Running at restart time:**
- HPC job `vd-vlm3` (agent's Qwen3-VL check, ~81%, finishes on the HPC by itself). Recover its outputs: `cd ../Dataset_videodet && HPC_REMOTE=/l/users/<user>/punch_videodet GAUGE_PATH=$HOME/.claude/train-progress-videodet.json /opt/anaconda3/bin/python hpc.py --attach vd-vlm3`
- **The detector-A background agent stops at restart**, before its final report. To resume: re-launch an agent on worktree `../Dataset_videodet` (branch `videodet`), telling it to finish from the pushed state: attach vd-vlm3, analyse the VLM answers, finalise the probe settings, write `videodet/REPORT.md`, push.

**Next steps (agreed with the user):**
1. Tracker self-test → fix → commit.
2. When detector A is final: rebuild `detect/dataset7` with its final scores, retrain the 7-class model, compare (12.0% misses / 3.2% false alarms).
3. Contact: the physics rule failed. Contact comes from the target's reaction (bag motion works on V2–V4; V1 masks unreliable); BOB wobble and opponent head snap to be built and tested on external footage.
4. Waiting on the user: external test footage (spec in `detect/TEST_FOOTAGE_SPEC.md`).


### Session continued (no restart, 2026-11-13 evening)
- **Detector A final** (agent report merged, `videodet/REPORT.md`): VideoMAE-L LOVO probe, strong L2 (C = 3e-4), **mean per-video AUC 0.930** (snapshot was 0.888). Qwen3-VL-8B says punch for 64% of A's top gap windows (labelled punches 56%, random gaps 13%), so they're mostly missed annotations. V6's labels contain only hooks and uppercuts (its jabs and crosses are unlabelled). VideoMAE weights are CC-BY-NC-4.0.
- **Final A vs the user's 77 verdicts: AUC 0.96.** A ≥ 0.1: 33/34 agreed (97%); A ≥ 0.5: 17/17. "No punch" A < 0.02: 24 moments kept, **0 punches**.
- **No-punch rule rebuilt on final A:** no punch = A < 0.02, gap, person present ≥ 90%, ≥ 8 frames from A ≥ 0.1 or B ≥ 0.5 flags; hidden punches = A ≥ 0.1. → 2,981 no-punch clips (old snapshot dataset kept as `detect/dataset7_snapshotA`). 7-class retraining running (`runs/train7-7class-finalA`).
- **Tracker speed bug found:** NumPy `.npz` re-decompresses the whole array on every access, so per-frame lookups took ~0.2 s. Loading into memory once: **18 min → 9 s** for the 2-minute synthetic test, still **0 swaps / 20 crossings**. (The earlier HSV "speed-up" was the wrong target, but harmless.)
- **External test clips downloaded** (yt-dlp updated to 2026.10.26 by the user): A2 sparring (ring), A1 sparring (dark clothes, 60 fps → every 2nd frame), B1 BOB, C1 heavy bag, C2 standing bag. All pass the spec; contact sheets and a cut scan found no cuts. Skeleton extraction running; A2 done: 2 people in 54% of frames, 1 in 40%, 3+ in 5% (one fighter often missed, e.g. overlap or rope).
- **7-class retrained on the final-A no-punch rule** (`runs/train7-7class-finalA/`, 5 seeds): balanced acc 74.4% ± 0.8, **type 73.2% ± 0.7, missed punches 10.0% ± 1.1 (was 12.0), false alarms 2.8% ± 0.3 (was 3.2).**
  - Misses by video: V1 15% (was 21), V4 12% (was 17), V2 26%, V10 32% still worst; the others ≤ 7%.
  - False alarms by video: V9 9% (was 2%, to watch); the others 1–3%.
- **External skeleton extraction:** A2 sparring 2 people in 54% of frames; **A1 sparring (dark clothes, high camera) 2 people in only 30%**; **B1 BOB: 1 person in 100% of frames, so MediaPipe does NOT detect the BOB as a person** (locate it from pixels, like the bags); C1 heavy bag 1 person in 99%.
- **External extraction finished:** C2 standing bag has 1 person in 100% of frames.
- **Tracker fixes (external/track.py):**
  1. A1 is 60 fps with every 2nd frame kept, but the tracker read every video frame, so its colour checks drifted onto the wrong frames. It now skips to the right frame.
  2. **Real footage broke it:** fighters were held in only 1–6% of frames. A fighter who went undetected kept "coasting" at their last speed forever, so the predicted spot drifted off-screen and they were never re-found. The synthetic test only had 1-frame dropouts, so it missed this.
     - Fix: coasting slows to a stop; the search area widens while someone is unseen; after 30 frames unseen, a fighter is re-found mainly by clothing colour, and only on a good match, so bystanders aren't grabbed.
     - Result: A2 held 75% / 82% of frames, A1 65% / 60%, close to the ceiling, since MediaPipe sees both people in only 30–59% of frames. Synthetic test still has 0 swaps.
     - Visual check, A2 first 60 s: F1 (red headgear) and F2 (green headgear) keep their labels through a side swap, and spectators are ignored.
- **Stance rule (external/stance.py):** combines a 3D "which side is forward" clue (needs both legs visible) with a flat-image "facing direction + which foot is further that way" clue.
  - Own videos: whole-video call right on 10/10 (V4 southpaw), but 10-s stretches only 62–98% right.
  - **Fails on external C1 heavy bag:** called southpaw, but the on-screen text ("JAB-CROSS-L.HOOK-R.UPPERCUT") shows the boxer is orthodox. The side-on view hides the far leg, so the 3D clue is ~0.
  - Confirmed by running the model both ways on 30 s: the orthodox reading gives 29 punches including "jab, cross, lead hook…" sequences; the mirrored reading gives 17, almost all lead hooks.
  - Next: **stance by model**, i.e. try both readings and keep the one with more clear punches. LOVO check on own videos running (external/stance_by_model.py).
- **external/run_test.py:**
  - Final 7-class model = 3 seeds trained on all 10 videos (runs/final7/, safe-loadable).
  - Per fighter: stance, mirroring for southpaws, the same streaming counter as the real-time demo.
  - Outputs summary.json and an annotated video in runs/external/{id}/.
  - A2 first 60 s: F1 44 punches, F2 59. Likely over-counting (feints, guard moves) and many F1 "lead uppercuts" look odd. **Not verified, no labels.**
- **Stance by model, leave-one-video-out (external/stance_by_model.py, runs/stance_by_model.json):** run the skeleton as-is and mirrored, and the reading with more counted punches is the stance.
  - **Whole video: 10/10 right**, V4 southpaw included. Punches (right reading vs flipped): V1 1746/1484, V2 1269/246, V3 937/182, V4 1211/853, V5 721/294, V6 1703/701, V7 746/314, V8 781/449, V9 299/122, V10 386/306.
  - 1-minute stretches are 67–100% right (V1 68%, V4 67%, V10 72% weakest), so we use one call per fighter per clip; stance switching is not handled.
  - Now the stance method in run_test.py. Margin under 10% is flagged "stance uncertain"; the geometric rule is kept as a second opinion.
  - **C1 heavy bag is now orthodox (29 vs 17), matching the on-screen combo text.** 30-s counts: jab 10, cross 7, lead hook 4, rear hook 6, rear uppercut 2, which fits the drilled jab-cross-lead hook-rear uppercut (rear uppercut vs rear hook confusion).
- **Full-length external test runs, all 5 clips** (runs/external/{id}/summary.json + {id}_annotated.mp4; final 3-seed model, never trained on these). **Counts are UNVERIFIED (no labels yet).**

  | clip | fighter | stance (as-is vs mirrored punches) | counted | mix |
  |---|---|---|---|---|
  | A2 ring sparring, 5 min | F1 red headgear | orthodox (124 vs 58) | 124 | jab 27, cross 24, lead hook 29, rear hook 7, lead uppercut 23, rear uppercut 14 |
  | A2 | F2 green headgear | orthodox (151 vs 97) | 151 | jab 63, cross 32, lead hook 17, rear hook 17, lead uppercut 10, rear uppercut 12 |
  | A1 dark gym sparring, 5.5 min | F1 red gloves | orthodox (235 vs 117) | 235 | jab 59, cross 90, ... (likely over-counted) |
  | A1 | F2 blue gloves | orthodox (141 vs 62) | 141 | jab 45, cross 41, ... |
  | B1 BOB, 3 min | | orthodox, uncertain (314 vs 325) | 314 | jab 106, cross 75, lead hook 86, rear hook 1, lead uppercut 8, rear uppercut 38 |
  | C1 heavy bag, 3 min | | orthodox (132 vs 94) | 132 | jab 54, cross 36, lead hook 12, rear hook 15, lead uppercut 1, rear uppercut 14 |
  | C2 standing bag, 2.3 min | | orthodox, uncertain (91 vs 88) | 91 | jab 46, cross 41, lead hook 3, rear uppercut 1 |

- **Visual checks:**
  - A2 identities held, including a side swap.
  - **A1 identities not trustworthy:** the skeleton spans both bodies in a clinch, both wear black, often back to camera, often only one fighter visible.
  - **B1 on-screen text "DOUBLE JAB-CROSS-R.BODY HOOK" = orthodox.** Stance-by-model was a near-tie that went wrong (southpaw: 184 "lead hooks"). New rule: **near-tie (margin < 10%) -> orthodox.** This was added AFTER seeing B1, so B1 is no longer an independent stance test. Orthodox reading: jab is top (106), fitting "double jab"; the right body hook is not recognised (rear hook 1). Body shots are probably rare in our training videos.
  - **C2 contains a scene cut** (different shorts, other side of the bag at ~133 s; the spec check missed it) and talking/walking breaks.
- **Next:** review page so the user can mark counted punches real / not, giving the first true external accuracy; then contact detection.
- **Why sparring fails (user's diagnosis, confirmed by measurement).** Hand speed relative to hips, in torso lengths per second (median / top 10%):

  | footage | hand speed | body speed, top 10% |
  |---|---|---|
  | training punch frames | 4.1 / 11.1 | 1.3 |
  | training no-punch frames | 1.9 / 6.8 | 1.9 |
  | bags (C1, B1, C2) | 1.7–2.8 / 6.3–9.3 | |
  | A2 sparring | 3.1–3.3 / 12.8–15.2 | |
  | **A1 sparring** | **4.3 / 18.3** | 3.8–4.1 |

  - In training, speed nearly separates punch from no-punch, so the model likely learned "fast = punch".
  - In sparring the TYPICAL hand moves at training-punch speed, and the fastest moves exceed anything in training.
  - Tracking losses and skeleton jumps add fake speed.
  - Candidate fixes (no external data):
    1. speed augmentation, i.e. fast no-punch clips in training;
    2. remove physically impossible jumps;
    3. keep all 60 fps frames plus fresh per-frame detection for tracking.
- **Source paper: BoxingVI** (Kumar et al., arXiv 2511.16524; github.com/Bikudebug/BoxingVI). Read 2026-11-13.
  - **It is a dataset paper with NO model results** (no accuracy, F1 or mAP), so our LOVO numbers are effectively the first baselines on it.
  - Paper facts:
    - 20 videos, 18 athletes, 6,915 clips; split by subject, S1–S15 train and S16–S20 validation;
    - AlphaPose 2D in COCO format, 25-frame padding, 6 classes;
    - no no-punch class; stance/southpaw never mentioned;
    - claims "sparring", but its own Section 3 says shadow boxing, bag workouts and tutorials.
  - Future work, in their words: multi-person combat and opponent interactions, which is exactly our open problem.
  - **We have only V1–V10 (5,428 clips after cleaning).** The paper and Drive say 20 videos. The other 10 would be the SAME dataset, so not external data. Drive subfolders couldn't be listed automatically; the user is asked to check.
  - DeepStrike (Jabbr) is a commercial product with self-reported 98–99% detection, trained on "millions of punches from online fight footage"; there is no paper.
- **2026-11-13 late: sparring push (user goal: "I need to make it work on sparring").**
  - freshdet branch FAILED: fresh per-frame detection finds fewer fighters.
  - speedaug branch: the baseline is only 4% fooled by sped-up own movements, so speed is NOT the main cause; the aug trades misses for false alarms. Neither is merged.
  - Measuring stick: the user prefers NO hand labelling, so the plan is a VLM labeller. First validate Qwen3-VL-32B (4-bit, bitsandbytes in /l/users/<user>/pylibs_bnb, weights to /l/users/<user>/hf_cache) on our OWN labelled windows; pass bar >= 85% agreement, fixed in advance. If it passes, label A2's first 2 min per fighter and score with external/score_sparring.py.
  - Labelling page review/sparring_label/label.html stays as the fallback.
  - RESUME POINT: check the download (hf_cache/dl_qwen32b.log on HPC), then write the VLM benchmark script.
## Branch vlmlabel (2026-11-14 night): VLM as automatic sparring labeller
- Goal: labels for sparring WITHOUT user hand-labelling. A VLM must first pass a benchmark on our OWN labelled videos: external/vlm_bench.json, 200 windows (10 punches + 10 strict no-punch per video, 16 frames, skeleton-centred crop, crops checked visually). **Pass bar fixed in advance: balanced punch/no-punch agreement >= 85%** (external/vlm_score.py).
- Caveat: our no-punch windows are EASY (standing, squats); sparring's parries/feints are absent, so passing is necessary, not sufficient.
- **Slowness found:** Qwen3-VL-8B took ~51 s/window (16 frames, 448 px, 3.3k tokens). Time is linear in tokens and the GPU is at full clock. Profile: **vision tower 13.4 s vs language model 0.2 s** (8 frames). Suspect: Conv3d patch embedding (kernel == stride) slow in bf16 on torch 2.9 + cu128.
  - Fix in vlm_label.py: fast_patch_embed(), an identical linear map with the same weights. external/vlm_fastcheck.py checks equality and speed (job queued: HPC limit 2 jobs per user).
- 32B weights: two login-node downloads stalled; the user killed them. Now a CPU Slurm job (external/fetch_qwen32b.py -> /l/users/<user>/hf_cache3, xet off), ~14 min per 4.7 GB shard.
- HPC tips:
  - reuse HPC_REMOTE dirs so pushes are incremental (fresh dirs re-upload 258 MB);
  - Lustre was very slow tonight;
  - stale rsync PID 4070116 (D state, lo-01) needs an admin.
- RESUME: when vlm-bench-8b2 finishes, score with `python external/vlm_score.py runs/vlm-bench-8b2/vlm_8b.json`; check vlm-fastcheck; when the 32B download is done, run the 32B benchmark with the fast patch.
- **Qwen3-VL-8B benchmark result (200 windows, 51.8 s/window):**
  - At P(yes) > 0.5: balanced 82.5%, FAIL (yes on punches 67%, no on no-punch 98%); V8–V10 shadowboxing only 0–20% yes. **AUC 0.980.**
  - **Fair cut-off chosen leave-one-video-out** (threshold picked on the other 9 videos): **balanced 92.0%, PASS**. Punches found 97%, no-punch correct 87% (V6 50%, V3 70%). Chosen cut-offs are tiny (0.000–0.003): use P(yes) as a ranking, not a probability.
  - **Type is useless:** exact 29%, lead/rear 43% (chance 50%); it mostly answers "right straight". So the VLM labels punch PRESENCE/TIMING only; type stays with the skeleton model.
  - Next: speed fix check, then label A2's first 2 min per fighter (~900 windows) and compare with the 32B.
- **Speed fix confirmed (vlm-fastcheck):** patch embedding Conv3d took 53.73 s vs the identical linear map 0.0005 s. **Full forward 0.80 s (was ~52 s, 65x faster)**, 3,319 tokens. Max abs diff 0.0156 (bf16 rounding, values ~0.22); final logits diff 0.31; same top token. Benchmark is being re-run with the fix to confirm the scores.
- **Fast benchmark re-run (8B + fast patch):** 1.3 s/window (was 51.8). The slow and fast P(yes) have rank correlation 0.999, with the same Yes/No at 0.5 in 98.5% of windows. AUC 0.980; **fair LOVO cut-off 91.0% balanced (PASS)**. Equivalent, 40x faster.
- **Sparring scored with VLM labels** (A2 first 2 min, 750 fighter-windows of 16 frames, 0.8 s/window; external/vlm_score_sparring.py).
  - At the benchmark-optimal cut-off (0.0002): red VLM-yes 40% / model 29%, found 54%, real 74%; green 52% / 39%, found 62%, real 82%. **Strongly cut-off dependent:** from cut-off 0.0002 to 0.5, found rises 54->72% and real falls 74->31%. The tiny benchmark cut-off is inflated on sparring (movement everywhere).
  - **Visual check of 16 disagreement windows (cut-off 0.01):**
    - "model only": ~7/8 show NO punch. These are fake counts, mostly "uppercuts" while the fighter holds a high guard, and one where the OPPONENT's arm entered the crop.
    - "VLM only": ~1–2/8 clear real punches the model missed (e.g. a green jab). Several are VLM errors: low P, fighter hidden behind the other, or the punch credited to the wrong fighter in an overlap.
  - **Conclusion:** over-counting is real (rough estimate: about half of the sparring counts are not real punches); some misses too. The VLM is best used at a higher cut-off (~0.1–0.2) and is weak on occlusion and attribution. It does not judge type.
  - Next: "toward the opponent" gate, with settings fixed in advance; minute 1 and minute 2 reported separately.
- **"Toward the opponent" gate (external/opponent_gate.py; settings fixed in advance: approach >= 0.25 torso OR min distance <= 1.0 torso):** drops ~30% of counts (red 124->87, green 151->101).
  - Window scores, found/real:

    | cut-off | minute | red before | red after | green before | green after |
    |---|---|---|---|---|---|
    | 0.1 | 1 | 70/38 | 48/37 | 69/46 | 56/51 |
    | 0.1 | 2 | 73/73 | 73/73 | 82/27 | 82/33 |

  - Green's real rises 5–7 points; **red's real is flat while its found drops a lot.** Fake guard "uppercuts" happen at close range, so the hand is near the opponent anyway.
  - **Not a clear win; not adopted.** Large minute-to-minute swings show how noisy a 2-minute window score is.
- **Where this leaves sparring:** training-free filters (speed aug, fresh detection, opponent gate) have not fixed the fake counts. The evidence points to the model never having seen sparring guard/defence movement.
  - The likely real fix is sparring examples in training. The VLM can now produce punch/no-punch labels automatically (presence only, not type) on OTHER sparring footage, testing on A2.
  - **This bends the user's "no external training data" rule, so it is the user's decision.**

## 2026-11-14 — Qwen3-VL-32B (4-bit) as labeller: no better than 8B
- Benchmark (200 own-video windows): balanced 86.5% at 0.5, AUC 0.973; fair leave-one-video-out cut-off **91.5%**
  (8B: 91.0% fast / 92.0% slow). Punch type still useless (exact 24%, hand 44% ≈ chance). 2.9 s/window vs ~0.8 s for 8B.
- Sparring A2 first 2 min (750 windows), 32B cut-off 0.107 from own-video benchmark:
  - red: VLM punch 18%, our model 29% | found 66% | real 41% | kappa 0.36
  - green: VLM punch 33%, our model 39% | found 60% | real 50% | kappa 0.29
- 8B vs 32B on the same sparring windows: rank corr 0.89; yes/no agreement 78–89% (kappa 0.54–0.69).
  The two labellers only moderately agree on sparring, so VLM labels there are noisy (fine for "roughly half", not for fine scoring).
- Conclusion: confirms the earlier estimate — about half of our sparring counts are not real punches, and we find ~60–65% of real ones.
  Keep the 8B as labeller (same accuracy, 4x faster). Next real fix still awaits the user's decision on training with
  VLM-labelled other sparring footage (A2 held out).

## 2026-11-14: real bouts with human labels (Olympic set + BoxingWeb); MediaPipe is the sparring bottleneck
- **Merged `vlmlabel` into `main`** (fast-forward). A stray tracked symlink `external/raw` on that branch was untracked first: it would have replaced main's real footage folder.
- **Data rule (user, 2026-11-14): outside data may be used for training.** A1/A2 and their channels stay test-only.
- **Outside literature search (user's external chat) read.** Its two key claims were checked here; see below.

### Olympic set (Kaggle "Olympic Boxing Punch Classification Video Dataset"; Stefański, Kozak, Jach, Entropy 2024; non-commercial)
- Local copy: `~/Downloads/Olympic Boxing Punch Classification Video Dataset` (not in git). 29 bouts, **104 min**, 1920x1080, 50 fps, two venues/cameras (`kam2`: 15 bouts, `kam4`: 14; different gyms, so no bout is shared between cameras).
- **Labels are richer than the status brief assumed** (CVAT tracks with boxes, Polish names): **4,605 punches** = head 2,513 / body 372 / block 538 / miss 1,182, each with **left or right hand**. So thrown-but-not-landed punches ARE labelled.
- The label marks the **contact moment** (median 4 frames = 80 ms), with a box at the hit location. It does **not** say which boxer threw the punch; that has to come from whose glove is at the box.
- About 44 labelled punches per minute for the two boxers together.
- Scene: small fighters (~250 px tall), a referee in white, many spectators, ropes. Much harder than A1/A2.
- Code: `olympic/parse.py` -> `olympic/events.json`, `olympic/videos.json`; `olympic/register.py` adds the bouts to `external/manifest.json` as `O_<camera>_<clip>` (category `O`) with symlinks in `external/raw/`.

### BoxingWeb (BoxMind; Wang et al., arXiv 2601.11492; github.com/gouba2333/BoxingWeb; no licence stated)
- Downloaded from Tsinghua Cloud (5.19 GB) to `~/Downloads/BoxingWeb/` with a resumable script (`download.sh` there). Not in git.
- **50 rounds of elite bouts, 134 min, 6,872 punches** (data_train 40 rounds / 5,537; data_test 10 rounds / 1,335). Mixed formats: mostly 1280x720 at 24-30 fps, 11 rounds 1920x1080 at 60 fps.
- Per punch: `frame_begin`/`frame_end` (median 12 frames), **side** (red/blue = which boxer), **technique** (l/r straight 1,903/1,547, l/r hook 1,268/1,097, l/r uppercut 538/519), **target** (head 5,389, chest 769, abdomen 714), **effect** (effective 1,780 = 26%, ineffective 5,092), distance.
- This is the first data with type + fighter + landed on real fights. Technique is by LEFT/RIGHT hand, so lead/rear needs each boxer's stance.
- Each round has a `*_pose_gt.pkl`. **Not opened: pickle can run code on load**; inspect with a restricted unpickler first.
- Not yet checked: camera cuts/replays in the broadcast footage, label completeness.

### Step 3: pose ceiling at the human-labelled punches (`olympic/pose_ceiling.py`, `runs/olympic_pose_ceiling.txt`)
- Second tool: YOLOX-m person detector + **RTMPose-m** via rtmlib/onnxruntime (`olympic/extract_rtm.py`, conda env `rtmpose`), dumps in `external/pose_rtm/` (git-ignored). First 90 s of 4 bouts (2 per venue), 231 labelled punches.
- Test: at each labelled punch, does some skeleton have a wrist inside the referee's box (+30 px), and does a second skeleton have a head/torso point in it?

| | MediaPipe (5 poses) | RTMPose |
|---|---|---|
| puncher seen | **45.9%** | **99.6%** |
| both boxers seen (contact punches) | **15.3%** | **96.2%** |
| chance (labels shifted 7-21 s, one bout) | 10-17% / 2-7% | 50-62% / 30-33% |

- RTMPose's chance level is high because it also poses the crowd behind the ring (5-12 people per frame). Still far above its own chance; MediaPipe is not.
- Visual check: MediaPipe usually returns 1 person, often the **referee** or half a boxer, and the tracker then follows the referee. RTMPose gives both boxers clean skeletons in 6/6 sampled punch frames.
- Speed: MediaPipe ~30 video fps per worker here. RTMPose 4.3 fps on a busy CPU, ~1 fps when 4 ran beside MediaPipe; CoreML crashed on the detector. A full run belongs on the HPC GPU. Not a real-time measurement (all people posed, detector every frame).
- **Conclusion: on real bouts the failure is in the pose step.** RTMPose is 2D only; the model must be retrained on 2D (own videos: 72.7% vs 73.7% with MediaPipe 3D) or get a lifter. The tracker must pick the 2 boxers among everyone (kit colour, inside the ring).

### Step 4: the CURRENT counter, unchanged, on all 29 bouts (`olympic/score_counter.py`, `runs/olympic_baseline.log`, `runs/olympic_score_baseline.json`)
- Pipeline exactly as for A1/A2: MediaPipe (5 poses) -> `external/track.py` -> `external/run_test.py --no-video` (new flag) -> final 3-seed model.
- Matching: a label is found by a counted punch whose span (+-0.2 s) contains it, one-to-one, either fighter.

| measure | result |
|---|---|
| labelled punches found | **24.8%** (chance, labels shifted 7 s: 15.8%) |
| counts matching a label | **47.7%** (the rest are fake OR unlabelled) |
| hand right, among found | 60.6% (coin flip 50%) |
| counted by the fighter whose wrist is nearest the box | 87.8% |
| a tracked fighter's wrist within 80 px of the box | 45.1% of labelled punches |
| found by outcome | head 24%, body 23%, block 29%, miss 25% |

- Counted 23/min against 44/min labelled. Stance calls are mostly "uncertain".
- **This is the baseline every fix is judged against.** It is barely above chance, and the pose step alone explains most of it (a tracked wrist reaches the box in only 45% of punches).
- **Caution (user):** "unlabelled = no punch" was wrong for BoxingVI (~15% hidden punches). Before using unlabelled moments as negatives here, check ~50 moments where the counter fires with no label.

### Next
1. RTMPose on all Olympic bouts and BoxingWeb (HPC GPU), plus a boxer selector (2 boxers among referee/crowd).
2. Re-grade with RTMPose skeletons (2D-only model) against this baseline.
3. Train: thrown-punch detection with real-bout positives and mined hard negatives; landed/blocked/missed head using both boxers; type on real bouts from BoxingWeb.

### Boxer selector v1 (`olympic/select_boxers.py`, 2026-11-14) — before any RTMPose skeleton reaches the punch model
- **Order (user):** build the selector BEFORE switching the model to RTMPose, or the referee/crowd skeletons would be fed to it. RTMPose extraction only saves skeletons; the selector decides which 2 are passed on.
- Method: tracklets (Hungarian on keypoint distance) -> 9 kit-independent features per person over +-1 s (wrist height = guard, hip mobility, arm speed, leg extent, keypoint score, size, distance to nearest similar-sized person, tracklet length) -> logistic regression -> smoothed P(boxer), top 2 per frame, with hysteresis for whole-tracklet boxers.
- **Free labels from the referee:** at a contact punch, the person with a wrist in the box and the second person at the box are boxers; everyone else in that frame is not. 4 sample bouts x 90 s: 230 boxer / 761 other person-frames.
- **Leave-one-venue-out** (train on 2 bouts of one gym, test on the other gym):

| held-out bout | puncher among the 2 selected | both selected at the box (contact) | frames with 2 selected |
|---|---|---|---|
| kam2_gh088416 | 88% | 40% | 45% |
| kam2_gh218416 | 100% | 86% | 84% |
| kam4_gh069681 | 96% | 90% | 88% |
| kam4_gh199681 | 98% | 71% | 99% |

  - Person-frame accuracy 83.5% (kam2 held out) / 90.5% (kam4 held out). Strongest cues: low wrist held high (guard), arm speed.
  - With ALL people RTMPose gave 99.6% / 96.2%, so the selector still loses boxers, mainly in kam2_gh088416.
  - Visual check, 8 frames of kam2_gh088416: crowd never selected; referee selected in 1 of 8 (when a boxer had no skeleton). Misses = a boxer hidden behind the referee (no skeleton at all), or a boxer just under the threshold.
  - "Shifted labels" is not a clean chance level for the selector (boxers often stand in the same spot seconds later), so the referee rate needs its own check.
- **Not good enough yet to feed the model.** It was trained on 2 bouts per venue; the fix is more data (RTMPose on all 29 bouts -> ~4,600 punches of free labels), then kit colour as an optional extra cue and a direct referee-selected measurement.

### RTMPose on all bouts, HPC GPU (2026-11-14 13:55, RUNNING)
- HPC dir `/l/users/<user>/punch_rtm` (own dir, not touched by `hpc.py` pushes): `videos/` (29 Olympic `O_*` + 50 BoxingWeb `W_{train,test}_*`, 8.18 GB, sizes verified identical), `code/`, `lists/`, `out/`, `logs/`. Local copies of the code: `hpc_rtm/`.
- Env `punchrtm`: python 3.11, rtmlib 0.0.16, **onnxruntime-gpu 1.22.0 with the cu12 nvidia wheels**. Two traps: (1) rtmlib pulls the CPU `onnxruntime`, which hides the GPU build: uninstall both, reinstall the GPU one; (2) onnxruntime-gpu 1.32 needs CUDA 13, but the nodes' driver is 570.195 (CUDA 12.x only), so it silently fell back to CPU.
- Jobs 298552 (`rtm-part0`, 40 videos) and 298553 (`rtm-part1`, 39), Olympic bouts first; ~40 processed frames/s each on an RTX 5000 Ada (laptop: 1-4). K = 20 people per frame. Results: `out/{id}.npz`, written on node-local disk and copied once. Jobs skip finished videos, so they can be resubmitted.
- Fetch: `rsync -a hpc:/l/users/<user>/punch_rtm/out/ external/pose_rtm/` (the 4 local 90-s samples there will be replaced by full-length dumps; delete their `.feat.npz` caches first).
- BoxingWeb clips are linked as `external/raw/W_*.mp4` (not yet in `external/manifest.json`).

### Boxer selector v2: trained on ALL 29 Olympic bouts (RTMPose from the HPC), 2026-11-14 afternoon
- All 29 Olympic RTMPose dumps fetched to `external/pose_rtm/` (full length, K = 20); the old 90-s samples are in `external/pose_rtm_samples90s/`. BoxingWeb extraction still running on the HPC.
- Same method as v1 (`olympic/select_boxers.py fit`, log `runs/olympic_selector_fit.log`), now ~4,600 labelled punches of free labels. **Leave-one-gym-out:**

| held-out gym | puncher among the 2 selected | ceiling (all people) | both selected "at the box" | ceiling | white shirt (referee) among selected | frames with 2 selected |
|---|---|---|---|---|---|---|
| kam2 (15 bouts, 1,810 punches) | **97.5%** | 99.1% | 80.3% | 95.5% | 1.6% (18/1,142) | 92.5% |
| kam4 (14 bouts, 2,795 punches) | **95.3%** | 99.5% | 65.6% | 98.4% | 0.2% (2/1,045) | 90.0% |

- **The "both at the box" number understates the selector.** Diagnosis on kam4: in 34% of contact punches one selected person had no wrist/head point in the referee's box; looking at 4 such frames, 3 had BOTH boxers correctly selected (the box is small, the opponent's keypoints fall outside it) and 1 was a real miss (boxer at P = 0.42). The ceiling is also inflated by crowd skeletons behind the box. So this metric is not a clean judge; a better one is needed (BoxingWeb says which boxer threw each punch).
- **Kit-colour check FAILED as a method** (`olympic/selector_colour_check.py`): torso colour classes are unreliable here (dark navy vests read as "other"; in some bouts both boxers wear red vests and differ only by headgear). Its numbers are not used.
- **Visual check on held-out clips** (`olympic/render_selector.py` -> `runs/rtm_samples/*_selector_*.mp4`, local only), 8 frames: both boxers correct in 5; referee never selected. Failure modes:
  1. when one boxer has no skeleton (at the frame edge, or hidden), the second slot is filled with a **standing spectator** at P ~0.5-0.6;
  2. **after the bell** (referee raising a hand), the boxers drop their guard and are rejected, and spectators get picked;
  3. a boxer walking with hands down scores low (0.09).
- Next for the selector: a pair rule (the two picks must be near each other and of similar size; otherwise keep one), a higher bar for the second pick, and an honest metric from BoxingWeb's per-boxer labels.

### Selector: the 2D-distance objection, two depth cues tested (2026-11-14)
- **User's objection (correct):** "the two picks must be near each other" is meaningless in a flat image; a spectator metres behind the ring can sit next to a boxer on screen.
- **Judged by eye, 24 random held-out frames** (12 per gym, picks only; `olympic/render_random_frames.py`, same seed for every method). Top-2-independently (v2): **14/24 fully right** (kam2 9/12, kam4 5/12); **10/24 include a non-boxer**. In kam4, 6 of 7 wrong frames = a spectator standing behind the far ropes, often with raised arms (filming, hands on head = looks like a guard). So the earlier numeric metrics flattered v2.
- **Depth cue 1, feet on the ring floor (`olympic/selector_ring.py`): FAILED.** Ring region self-calibrated per bout from confident picks' ankles. Raw separation looked good (boxers' foot in the region 91-94%, others 30%), but adding it to the classifier or as a hard gate changed nothing (wrong picks at punches 8.9 -> 8.2-8.6% kam2, 21.3 -> 21.9-22.3% kam4). 50-90% of wrong picks already have a "foot in the ring": legs behind the far ropes are hidden, RTMPose guesses the ankles, and the guesses land on the canvas (and wrong picks contaminate the self-calibrated region).
- **Depth cue 2, same apparent size (`olympic/selector_pair.py`): HELPS.** Boxers exchanging punches are at the same distance from the camera, so they look the same size; choose the pair jointly: size ratio within 1.5, hips within 4 body-scales, highest P_i + P_j; otherwise only the single best.

| held-out gym | picks nowhere near the punch: top-2 | pair rule | puncher kept: top-2 | pair rule |
|---|---|---|---|---|
| kam2 | 8.9% | **4.2%** | 95.9% | 97.1% |
| kam4 | 21.3% | **11.8%** | 87.6% | 87.9% |

  - Same 24 frames by eye: **17/24 fully right** (kam2 9, kam4 8), **4/24 include a non-boxer** (was 10), 3/24 one boxer only (nobody wrong).
  - Remaining errors: referee overlapping a boxer; referee picked when a boxer stands hands-down; a boxer-sized spectator; **two spectators of the same size picked as a pair**.
- **Still not good enough to feed the model.** The per-frame approach has hit its limit: it re-decides who the boxers are in every frame from momentary pose. **Next: decide ONCE per bout.** Link tracklets into identities by appearance (kit/headgear colour, body proportions), sum the boxer evidence over the whole bout per identity, and keep the two identities with the most; this also answers "which boxer is which". `external/track.py` has the appearance pieces.
- 24 hand-judged frames is a small sample (roughly +-20 points); treat the eye counts as direction, not precise rates.

### BoxingWeb ships per-frame RED/BLUE boxer poses: supervised data for the selector and for identity (2026-11-14 16:30)
- HPC RTMPose run finished 15:43 (jobs 298552/298553, exit 0): all 29 Olympic + 50 BoxingWeb dumps in `external/pose_rtm/` (BoxingWeb: 261,372 frames, median 5 people per frame).
- **User's push:** appearance fails when fighters dress alike; "we have a lot of data, so there must be an efficient solution". Led to opening the `*_pose_gt.pkl` files.
- **Safety:** disassembled with `pickletools` first: the 50 files reference only `numpy.core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype`. Loaded with a restricted unpickler that allows exactly those (`boxingweb/safeload.py`).
- **Contents, per round:** `pose_red_2d`, `pose_blue_2d` (T, 45, 2; normalised by width, height), `pose_red_3d`, `pose_blue_3d` (T, 45, 3), for EVERY video frame (T = video frame count, all 50 rounds). 45-point layout is not COCO (torso matches: their 34/33/12/9 = COCO shoulders 5/6, hips 11/12; points stop around the knees).
- Visual check (1 round, 4 frames incl. a clinch): the red points sit on the red boxer and the blue on the blue boxer in all 4. How the authors produced them is not stated in the repo (the paper reports tracking IDF1 0.985), so treat as high-quality tracks, not hand labels.
- `boxingweb/build_labels.py` -> `external/boxers_gt/{id}.npz` (git-ignored): every RTMPose person in every frame gets a role, red boxer / blue boxer / other, by Hungarian matching of torso points (gate 0.9 torso lengths); also `boxingweb/events.json` (6,872 punches: frames, side, technique, hand, target, landed, distance).
  - **Both boxers matched to an RTMPose person in 86.7% of frames**, one in 10.9%, neither in 2.4%; 5.1 other people per frame.
  - One outlier round (train_2024-00-00_LENUTA-...: 30%) to inspect.
  - cv2 reports <= 30 fps for all 50 rounds (the 11 "60/1" ones too), so RTMPose frames and pose-file frames are one-to-one.
- **What this unlocks:**
  1. Selector trained with real supervision: ~450k boxer and ~1.3M non-boxer person-frames with identity, instead of 4,600 punch moments and hand-made rules. Test on the 10 test rounds (exact per-frame truth), then cross-domain on the Olympic bouts and A1/A2.
  2. Identity (which boxer is which) can be MEASURED per frame, in colour and in greyscale, and a learned association model trained on it.
  3. 3D poses of both boxers for every frame, aligned with typed + landed punch labels: training data for the punch model itself.
- Not yet checked: camera cuts / replays inside rounds; domain gap (broadcast, moving camera, close view) to gym footage.

### Learned boxer selector, skeleton-only, trained on BoxingWeb's per-frame truth (2026-11-14 evening)
- **Inputs (user asked): skeletons only.** No pixels, no colour, no absolute image position. 65 features per person-frame from a +-1 s tracklet window (`boxingweb/feats.py`): mean/spread of 12 joints relative to the hips, wrist/hip speeds, mobility, keypoint confidence, tracklet length, size relative to the others in the frame, and the nearest person of similar apparent size (distance, size ratio, their wrist heights and arm speed).
- **Training (`boxingweb/train_selector.py`):** gradient-boosted trees on 425,424 person-frames (28.1% boxers) from the 40 `data_train` rounds, frames where both boxers were matched. Picks: smoothed P(boxer) along the tracklet + pair rule. Model: `runs/selector_learned/gbm_boxingweb_train.joblib` (git-ignored), picks in `external/boxers_rtm/`.
- **Test 1, exact per-frame truth, BoxingWeb `data_test` (10 rounds, 47,527 frames with both boxers present):** person-frame accuracy 95.5% (boxers recognised 94.2%, others rejected 96.1%).

| selector | exactly the 2 boxers | a non-boxer picked | one boxer only | nobody |
|---|---|---|---|---|
| old 9-feature (Olympic-trained) + pair rule | 61.4% | 29.6% | 4.1% | 4.9% |
| **learned, top 2** | **88.5%** | **5.3%** | 5.7% | 0.5% |
| learned + pair rule | 85.9% | 6.7% | 6.9% | 0.5% |

  - The pair rule slightly hurts on broadcast footage (close camera: the two boxers can differ a lot in apparent size); it was designed for the gym view.
- **Test 2, Olympic gym bouts, NEVER trained on (different camera, gyms, crowd):**

| gym | puncher kept | picks nowhere near the punch | (hand-built pair rule, leave-one-gym-out) |
|---|---|---|---|
| kam2 | 98.8% | **0.7%** | 97.1% / 4.2% |
| kam4 | 96.2% | **1.3%** | 87.9% / 11.8% |

  - Frames with 2 selected: 79-81% (lower than before: it now prefers one pick to a doubtful second).
- **Same 24 Olympic frames by eye** (seed 7): exact 17/24 (kam2 9, kam4 8), a non-boxer picked in 2 (+1 ambiguous: a skeleton on a boxer hidden behind the referee), one boxer only in 4. History on these frames: top-2 hand-built 14 exact / 10 wrong; hand-built pair rule 17 / 4; learned 17 / 2-3.
  - Remaining: a spectator behind the ropes (1), referee next to a hands-down boxer (1), and a visible boxer left unpicked (2 of the 4 one-only frames).
- Bug fixed on the way: file filters in `olympic/*.py` matched the new `.f2.npz` caches; now they keep only names with a single dot.
- Not done: adding the Olympic weak labels to training (domain mix), once-per-bout identity, identity measurement (colour vs none).

### Selector with gym labels added; first identity measurements (2026-11-14 late)
- **Baseline reminder:** "24.8%" = share of the 4,605 referee-labelled Olympic punches found by the OLD pipeline (MediaPipe + old tracker + model trained on our 10 videos); chance by shifted labels 15.8%.
- Stale status-line gauge "pose O_kam2_gh178416" = the crashed first MediaPipe run (12:14); cleared.
- **Selector + Olympic gym labels (`boxingweb/train_selector_mix.py`): no change.** Leave-one-gym-out, gym rows up-weighted to 30%: wrong picks 0.6% / 1.3% (was 0.7 / 1.3), frames with 2 selected 77.8% / 81.1% (was 78.9 / 81.4), BoxingWeb test exact 88.6% / 87.8% (was 88.5). Keep the BoxingWeb-only model.
  - Two boxers are selected AT the labelled punch frame in 88.8% (kam2) / 90.1% (kam4); within 1 s of a punch 84-85%; elsewhere 71-73% (breaks, boxers apart).
- **Identity ("which boxer is which"), skeleton-only, BoxingWeb test rounds (`boxingweb/identity_baseline.py`, `runs/identity_baseline.txt`).** The two boxers are given (truth roles), so only identity is tested.
  - **Motion continuity on RTMPose detections: right identity in 51.8% of boxer-frames = a coin flip.** With identity reset at detected camera cuts (38 cuts, 1.3/min): 55.9%, **9.8 swaps per minute**, 196 of 298 swaps while the boxers' boxes overlap. (Simple tracker: last pose + velocity, 2x2 assignment.)
  - **Body signature (limb-length ratios, 1 s medians; learn on the first half of a round, test on the second): 68.4% mean, 53-85%.** Weak: bodies in one weight class look alike in 2D.
  - **The provided red/blue tracks are themselves smooth:** a swap would be the cheaper continuation in only 24 of 53,392 test frames (0.8/min). So continuity works on clean, complete poses; the swaps come from RTMPose's per-frame output in overlaps (a boxer missing, a skeleton jumping onto the other body).
  - **"Keep the left/right order through an overlap" does NOT hold:** over all 50 rounds (145 min), 2,674 horizontal-overlap episodes (18.4/min, median 0.47 s); the order differs afterwards in 28.3% of them (5.2/min).
- **Conclusion so far:** the user's doubt is confirmed by measurement: with skeletons from a per-frame detector, neither continuity, body proportions nor side order keeps identity. Open options: (a) a learned two-person tracker trained on BoxingWeb's tracks (predict both boxers' poses through the overlap), (b) pixel appearance per fighter learned within the video, (c) accept per-bout totals + per-fighter split only where confident. Not yet tried.

### Identity option 1: learned re-association at each separation, skeleton-only — NO GAIN over continuity (2026-11-14 night)
- `boxingweb/identity_episodes.py`: a round = "apart" stretches (the two boxers' boxes overlap < 0.15; following by pose distance is reliable: track purity 99.8%) + overlap episodes between them. One decision per episode: same way round or swapped. 12 skeleton cues (continuity vote through the overlap, extrapolated/held position, apparent size, feet height, limb signature, stance, left/right and front/back order, episode length). Train on data_train (1,513 episodes), test on data_test (363 episodes, 12/min). Boxers given by truth roles.
- **Broadcast camera cuts were inflating the errors.** By eye, 5 of 6 "wrong" episodes were the TV director switching camera; my skeleton-jump cut detector found 1.3 cuts/min, the pixel detector (`boxingweb/detect_cuts.py`, thumbnail difference spikes -> `external/cuts/`) finds **4.9/min** (710 in 50 rounds). Stretches now never span a cut and identity is re-set at cuts (a phone on a tripod has none).
- **Results, cuts excluded (test rounds, 30 min):**

| method | separations decided right | wrong per minute | right identity, apart frames |
|---|---|---|---|
| always "same way round" | 58.7% | 4.95 | 70.5% |
| left/right order kept | 76.6% | 2.80 | 79.7% |
| held position | 79.1% | 2.51 | 80.5% |
| **continuity through the overlap** | **92.0%** | **0.96** | **84.1%** |
| learned, logistic | 90.9% | 1.09 | 83.0% |
| learned, boosted trees | 90.6% | 1.12 | 82.1% |

  - Cues alone: continuity 92%, position 79%, limb signature 76%, feet height 72%, apparent size 64%, stance 61%. Combining them does not beat continuity: **the skeleton carries no extra identity information at a separation.**
  - **By eye, 10 of the 29 remaining errors:** ~4 real (same camera, the boxers genuinely exchanged sides during a multi-second overlap), ~5 still camera changes the pixel detector missed (zooms / angle changes), 1 truth error. So the real rate for one continuous camera is roughly **0.4-0.5 wrong identity decisions per minute** (small sample), i.e. 1-2 swaps per 3-minute round. In the real ones the truth labels match the kit colours, so pixels would have solved them in these bouts.
- **Flagging instead of guessing (`boxingweb/identity_flags.py`):** overlap length, percentiles 50/75/90 = 0.67 / 1.73 / 3.75 s.

| flag rule | separations flagged | errors caught | silent errors left |
|---|---|---|---|
| overlap lasted >= 1 s | 38% (4.6/min) | 22 of 29 | 0.23/min |
| overlap lasted >= 2 s | 21% (2.5/min) | 20 of 29 | 0.30/min |
| >= 1 s OR model confidence < 0.6 | 42% (5.0/min) | 25 of 29 | 0.13/min |

  - Errors concentrate in long overlaps, so they are flaggable, but the flag rate (2.5-5 per minute) is far too high for user taps; it needs a second cue that resolves most flagged cases automatically.
- **Verdict on the user's question:** skeleton-only cannot keep identity through long overlaps (~0.5 real swaps/min), and learning from 104 min of truth does not change that. For look-alike fighters the options left are pixel-level appearance learned inside the video (gloves, skin, hair, shorts trim; not shirt colour), a pixel-following tracker through the overlap, or flag + tap. Untested.
- Housekeeping: two inline scripts using multiprocessing had hung (macOS spawn cannot re-import stdin); killed. Use script files for Pool.

### Identity by RECOGNITION instead of tracking (user's idea), ready-made models, no training (2026-11-14 night)
- **User's idea:** a separate model whose only job is "A is A, B is B", asked when it matters (e.g. at each punch), likely available off the shelf (person re-identification).
- `boxingweb/identity_reid.py`: crops of each boxer from the apart stretches (6 per boxer per stretch, 6,156 crops, test rounds), mean embedding per stretch; DINOv2-small and ImageNet ResNet-50 features, cosine similarity. Truth roles pick the boxers.

| features | A. separations right (no cut) | B. at camera cuts | C. whole round, NEVER corrected, cuts included: fixed gallery / updated gallery |
|---|---|---|---|
| motion continuity (for reference) | 92.0% | nothing to go on | 84.1% WITH truth resets at cuts |
| **ResNet-50, colour** | **93.7%** | **85.7%** | **92.8% / 94.1%** of apart frames |
| DINOv2-small, colour | 89.3% | 85.0% | 85.0% / 84.3% |
| ResNet-50, greyscale | 82.6% | 69.3% | 70.8% / 64.1% |
| DINOv2-small, greyscale | 83.2% | 77.9% | 75.4% / 69.6% |

  - **In colour, recognition beats tracking and is self-correcting**: a mistake on one stretch does not carry over, so the whole round stays 93-94% right with no resets and through 140 camera cuts.
  - **Without hue it drops to ~70-75% over a round** (50% = chance). So most of the power here is the red vs blue kit; these generic features are not built to tell people apart. Greyscale still keeps red/blue brightness differences, so truly identical kit would be harder still.
- **Recognition and continuity make different mistakes** (`boxingweb/identity_combine.py`, 363 separations):
  - Colour: they agree in 87.9%, and are then right 98.7% (0.13 silent errors/min); they disagree 1.45 times/min, recognition right in 57% of those.
  - Greyscale: agree 80.7%, then right 96.2% (0.36 silent errors/min); disagree 2.31/min, continuity right in 74%.
  - So "trust agreement, flag disagreement" cuts silent errors from ~1/min to 0.13/min (colour) at 1.45 flags/min.
- **Not yet tried:** a real person re-identification model (trained to tell people apart in similar clothes), a small A-vs-B classifier fitted inside each video, asking at punch time, and the both-in-black clip A1.

### Identity by recognition, round 2: real person re-identification models (2026-11-14 night)
- "Wrong kind of model" in round 1 = ResNet-50 (ImageNet object classes) and DINOv2-small (general similarity): neither was trained to tell individual people apart.
- Now: **OSNet x1.0** and **OSNet-AIN x1.0**, trained for person re-identification on MSMT17 (4,101 people, 15 cameras). Weights `huggingface.co/kaiyangzhou/osnet` (MIT, loaded with `weights_only=True`); model code `third_party/osnet/osnet.py`, `osnet_ain.py` from github.com/KaiyangZhou/deep-person-reid (read first: plain module definitions; their gdown download path is not used). Tight 256x128 crops.
- `boxingweb/identity_reid2.py`, BoxingWeb test rounds: the first 5 apart stretches of each round are "the pictures we were shown" (60 crops, truth-labelled = a clean start); every later stretch is labelled with no correction, camera cuts included. Score = share of apart frames with the right boxer.

| model | colour: gallery / A-vs-B classifier / self-updating classifier | greyscale: gallery / classifier / self-updating |
|---|---|---|
| ResNet-50 (generic) | 92.7% / 93.6% / 94.9% | 74.3% / 75.7% / 76.6% |
| OSNet (re-identification) | 99.8% / 100.0% / 99.8% | 79.9% / 82.8% / 84.0% |
| **OSNet-AIN (re-identification)** | **99.9% / 99.8% / 99.8%** | **88.0% / 89.4% / 94.9%** |

  - **In colour the identity problem is essentially solved on these bouts: 99.8-100% of a whole round, never corrected, through camera cuts.** (Motion continuity: 84.1% even with truth resets at every cut.)
  - **Greyscale: 94.9%** with OSNet-AIN + a self-updating A-vs-B classifier (generic features: 77%). The per-video classifier and its self-updating both help most where it is hard.
  - Caveats: (1) the boxers are still handed over by the truth roles; (2) greyscale is an imperfect stand-in for look-alike fighters (it keeps the red/blue brightness difference, but is also an input the model never saw in training); (3) broadcast footage, large well-lit boxers; gym footage has smaller, blurrier people; (4) "apart" frames only: identity at the punch moment itself is not scored yet.
- Next: credit at punch time (BoxingWeb says which boxer threw each punch), boxers from the learned selector instead of the truth, then A1 (both in black) by eye.

### Identity, the REAL pipeline, scored at the punch (`boxingweb/identity_pipeline.py`, 2026-11-14 night)
- User: greyscale is too harsh (real look-alikes still differ in skin, hair, clothing colour); from here identity is scored in colour, greyscale = worst case only.
- No truth handed over: boxers = learned skeleton-only selector (top 2); apart stretches + pixel cuts; OSNet-AIN embeddings; stretch 0 defines A and B; every later stretch labelled by an A-vs-B classifier that keeps learning from its own answers. Truth only scores (which RTMPose person is the labelled puncher; one global A/B = red/blue choice per round, so chance = 50%).
- **BoxingWeb test rounds, 1,335 labelled punches:**
  - identity in the stretch frames: **99.5% right** (with truth-picked boxers it was 99.8%).
  - only **15.6% of punches happen inside an "apart" stretch**; the rest land while the boxes overlap, so identity has to be carried or asked at the punch.
  - **credit to the right boxer:** carry (stretch, else follow by motion from the last stretch) **94.2%** (covers 76% of punches); direct (crop the puncher at the punch frame, ask the classifier: the user's idea) **90.0%** (85%); **both: 94.4%** (85% of punches).
  - not creditable: 11.6% of punches where the puncher has no RTMPose skeleton matched to the truth pose at the punch frame (a boxer lost or merged in the overlap, or the matching gate), 3.4% where the selector did not pick the puncher.
  - per round 87.5-100%. Part of the remaining 5.6% is scoring noise: deciding which RTMPose person is the puncher inside an overlap is itself uncertain.
- HPC job 298944 (`rtm-sparring`): RTMPose on A1 (both in black) and A2, to run this pipeline on real sparring and judge by eye.

### Identity pipeline on real sparring, A1 (both in black) and A2, judged by eye (2026-11-14 night)
- Scope reminder (user asked): what is tested so far is boxer selection + identity (credit), with punches taken from human labels; punch DETECTION, TYPE and LANDED are not rebuilt on RTMPose yet. Publication question parked until the end (user).
- RTMPose on A1/A2 via HPC job 298944. **RTMPose sees >= 2 people in 94.2% of A1's frames (MediaPipe: 30%) and 100% of A2's (MediaPipe: 54%).**
- `boxingweb/identity_sparring.py` (selector with pair rule -> apart stretches -> OSNet-AIN -> self-updating A-vs-B classifier -> carry by motion in overlaps); outputs in `runs/identity_sparring/` (sheets and clips local only).

| clip | two boxers selected | stretches | frames with an identity | classifier margin (median; share < 1.0) |
|---|---|---|---|---|
| A1, 9 min, both in black | 83.0% | 186 | 98.9% | 3.9; 3% |
| A2, 5.3 min, ring + spectators | 71.9% | 97 | 85.5% | 5.3; 1% |

- **By eye, 24 evenly spaced stretch frames per clip:**
  - **A1: 24/24 consistent.** A is always the fighter with the dark headgear and red gloves, B the one with the white headgear and blue gloves, through side changes over 9 minutes. Both wear black shirts and shorts; the model separates them anyway (headgear, gloves, shoes), as the user predicted.
  - **A2: identity never swaps** (A = red headgear, B = green headgear) in all ~19 frames with both boxers picked. **5 frames are selector errors, not identity errors:** before the start, during breaks and after the bell, a boxer standing idle is not picked and a spectator is. Stretch 0 itself was boxer + spectator; the self-updating classifier recovered from that.
- Next for the selector: use the identity model to reject bystanders (someone who matches neither A nor B), and choose A and B once per clip from the two most frequent appearance clusters instead of trusting stretch 0.
- Overlap frames (identity carried by motion) were not judged by eye here; clips: `runs/identity_sparring/A1_sparring_gym_dark_clip.mp4` (120-160 s), `A2_sparring_ring_clip.mp4` (60-100 s).

### Swaps measured, including inside overlaps (`boxingweb/identity_swaps.py`, 2026-11-14 night)
- User saw a tiny swap in the A1/A2 clips and asked for a number. User also set the bar: once a part is usable, move on instead of polishing it.
- **Measured, BoxingWeb test rounds (truth in every frame), 30 min, 1,124 overlaps (37/min):**

| carry | right boxer, apart frames | right boxer, overlap frames | wrong episodes in overlaps | time with the wrong boxer | punch credit |
|---|---|---|---|---|---|
| forward (current) | 99.48% | 86.31% | 16.3/min, median 0.07 s (2 frames) | 6.87% | 94.2% of 1,012 |
| two-sided (tested fix) | 99.48% | 88.55% | 18.1/min, median 0.07 s | 6.49% | 93.9% of 1,103 |

  - Swaps are confined to overlaps, are short (typically 2 frames) and end at the next apart stretch, where recognition re-decides. Part of the overlap error is scoring noise (which RTMPose person is which boxer inside an overlap is itself uncertain).
  - **Two-sided carry (also follow backwards from the stretch after the overlap; where the two disagree, each frame takes the label from the nearer stretch): no real gain** (overlap frames +2 points, punch credit unchanged, it only covers more punches). Not adopted as a fix; kept in the script.
- **Estimated on the sparring clips** with an independent per-clip head-colour (headgear) A-vs-B check (agrees with the stretch labels on held-out stretches: 92%):

| clip | overlap frames flagged as swapped | swap episodes >= 3 frames | time swapped | false-alarm level (apart frames) |
|---|---|---|---|---|
| A1 (both in black) | 6.2% | 3.9/min, median 0.13 s | 1.0% of the clip | 1.1% of frames, 0.45 episodes/min |
| A2 | 11.6% (7.7% two-sided) | 4.2/min, median 0.18 s | 2.1% of the clip (1.2% two-sided) | 1.8%, 0.95 episodes/min |

  - The proxy is only 92% reliable, so these are rough upper estimates; its own false-alarm level in apart frames (where recognition is ~99.5% right) is 1-2%.
- **Decision: identity is usable** (about 94% of punches credited to the right boxer; brief swaps inside overlaps). Move on to punch detection on RTMPose skeletons. Selector idle-time errors (bystander picked during breaks) stay on the to-do list, not blocking.

### Punch detection + type rebuilt on RTMPose skeletons (BoxingWeb), first results (2026-11-14 night)
- **Data (`boxingweb/punch_data.py` -> `external/punch_ds/`):** per boxer and frame, 157 channels: own pose (hip-centred, body-scale units) + change + hip speed, the OPPONENT's pose in the same coordinates + change, own keypoint confidences, presence flags. Gaps <= 8 frames interpolated; boxer skeleton present in 94.7% of boxer-frames. Labels per frame: none / l-r straight / l-r hook / l-r uppercut (physical hand, no stance needed). **324 of 6,872 punch labels (4.7%) are unusable** (shorter than 2 frames, longer than 1.5 s, or out of range) and dropped; 2,036 consecutive same-boxer labels overlap (the later one takes over). Mirrored copies for training. Boxers = truth-matched skeletons (detection only).
- **Model (`boxingweb/punch_tcn.py`):** dilated residual 1D convolutions over time (about +-1 s), per-frame 7-way output; events from smoothed P(punch) with a threshold chosen on 5 held-out training rounds; split where the punching hand changes. 35 rounds train, 5 validation, 10 test (the authors' `data_test`). ~30 s per run on MPS. `runs/punch-tcn*/`.
- **Test, 1,178 usable labelled punches, 3 seeds (`boxingweb/punch_eval.py`, `runs/punch_eval.txt`):**

| model | matching | precision | recall | F1 | technique (3 classes) macro-F1 | hand right |
|---|---|---|---|---|---|---|
| two-person input | loose (any overlap, +-3 frames) | 0.769 | 0.772 | **0.771 +- 0.006** | 0.644 | 85.2% |
| two-person input | strict (temporal IoU >= 0.5) | 0.667 | 0.604 | **0.634 +- 0.008** | 0.727 | 97.3% |
| own skeleton only | loose | 0.762 | 0.768 | 0.764 +- 0.002 | 0.647 | 85.1% |
| own skeleton only | strict | 0.665 | 0.594 | 0.627 +- 0.012 | 0.717 | 97.5% |
| 3-seed ensemble, two-person | loose / strict | | | 0.786 / 0.637 | 0.670 / 0.745 | |

  - Chance (labels shifted 7 s): F1 ~0.20. Six-way type right among loose matches: 63.5%.
  - **The opponent's skeleton adds little** (+0.7 F1 points, about one standard deviation): unlike the old model's failure on sparring, a detector trained on real bouts does not need the opponent to reject fake punches.
- **BoxMind's reported numbers, read from the paper (arXiv 2601.11492, B.2, Table S3):** detection precision 0.806, recall 0.763, **F1 0.783** on the BoxingWeb test set; a temporal IoU threshold of 0.5 is mentioned in the same sentence as their NMS step and the matching rule is not stated; no baselines for detection. Technique F1 (straight/hook/uppercut): 0.692 pose-only, 0.725 best (video + pose). Their detector uses 2D + 3D poses from 4D-Humans and is trained on 40 BoxingWeb rounds + 30 private BoxingStudio rounds; tracking IDF1 0.985 with a manual first-frame initialisation.
- **Comparison (not like-for-like):** loose matching: ours 0.771 (ensemble 0.786) vs their 0.783, about equal; strict IoU >= 0.5: ours 0.634, clearly below IF that is their rule. Our event boundaries are the weak part (hand right jumps to 97% under strict matching, so the well-localised detections are also the well-classified ones). Technique macro-F1 0.727-0.745 under strict matching vs their 0.692 (pose) / 0.725 (best), but theirs may be measured on true punch segments. We use 2D only, less data, and drop 4.7% bad labels.
- Against our own baseline: old pipeline on real bouts found 24.8% of punches (Olympic set, chance 15.8%); the new detector finds 77% with 77% of counts real on BoxingWeb test (chance ~20%). Different dataset, so not the final comparison: the Olympic re-grade is still to do.

### FULL CHAIN end to end on BoxingWeb test rounds, no truth handed over (`boxingweb/punch_fullchain.py`, 2026-11-14 night)
- User: tightening punch start/end is not a priority; first confirm the whole chain. Status given: detector + type retrained as ONE model (done); landed not started; chain not yet run together.
- Chain: RTMPose -> learned selector (top 2) -> apart stretches + pixel cuts -> OSNet-AIN identity (stretch 0 = A/B, self-updating classifier, forward carry) -> per-boxer skeleton streams -> punch TCN, 3-seed ensemble, threshold 0.40 from validation. Truth only scores (A/B = red/blue per round; loose matching per boxer).

| boxers from | counted | found AND credited to the right boxer | counts that are real | F1 | type right (6) | hand right | boxer skeleton present |
|---|---|---|---|---|---|---|---|
| the truth (reference) | 1,171 | 78.4% | 78.8% | 0.786 | 66.0% | 86.6% | ~95% of boxer-frames |
| **full chain** | 1,063 | **69.2%** | **76.7%** | **0.727** | 64.9% | 85.4% | ~86% |

  - 1,178 usable labelled punches. The chain loses ~9 points of "found", almost all because a boxer's skeleton stream has gaps (86% vs 95% present): the selector leaves one boxer unpicked, or the two picks overlap and no identity is assigned. Precision, type and hand hardly move.
  - "Counts that match only the OTHER boxer's labels": 10.2% with truth boxers, 11.6% in the chain, so the chain adds ~1.4 points of wrong-fighter credit; the 10% floor is mostly simultaneous exchanges where both boxers punch.
- **End-to-end, per fighter, by type, on real bouts: F1 0.727** (old pipeline on real bouts: found 24.8% vs chance 15.8%, Olympic set, not the same test).
- Still open: landed / not landed (BoxingWeb "effective", Olympic head/body/block/miss); Olympic re-grade against 24.8%; selector gaps (the main loss); punch boundaries.

### LANDED or not: definition, first attempts, plan (2026-11-14 night)
- **BoxingWeb "effective" from skeletons (`boxingweb/punch_attr.py`, true punch segments + hand given):** landed macro-F1 0.560-0.571, AUC 0.62-0.63 (BoxMind Table S3: 0.621 pose-only, 0.645 best); target head/torso macro-F1 0.68-0.71 (theirs 0.747 / 0.764). Without the opponent's skeleton: landed F1 0.510, AUC 0.558. So the opponent matters here (unlike for detection), but skeletons carry little.
- **User: "we still have no concrete plan for landed; skeleton results are barely better than guessing." Correct.** Two problems were mixed: (1) BoxingWeb's label is a coach's QUALITY judgement (clean scoring blow, 26%), not contact; (2) skeletons cannot see contact (wrist, not glove; flat image).
- **DEFINITION AGREED WITH THE USER: landed = made contact with the head or body, as opposed to blocked or missed** = the Olympic set's referee labels (head/body 63%, block 12%, miss 26%, box at the contact point). BoxingWeb "effective" stays a secondary, stricter label.
- Plan: (1) Olympic labels, 3-way landed/blocked/missed, split by gym; (2) cheap physical skeleton cues first; (3) pixels at the contact point with a ready-made video model; (4) sound later (BoxingWeb videos have audio; the Olympic set has none; A1/A2/C1 were downloaded video-only).
- **Step 2 result (`olympic/landed_cues.py`): skeleton cues do NOT separate the outcomes.** 4,105 of 4,605 punches usable (both boxers picked and followed around contact). 19 cues (glove reach to head/torso, opponent guard in the way, guard height, opponent head/torso movement after contact and along the punch, arm extension, glove speed and slow-down, spacing). Per-cue AUC between 0.32 and 0.64 (0.5 = nothing); best: guard-minus-reach 0.64 for landed-vs-blocked, extension 0.33 for blocked-vs-missed, reach 0.40 for landed-vs-missed. All cues in boosted trees, train one gym / test the other: **3-way accuracy 57.1% vs 62.1% for always "landed"; macro-F1 0.433; landed-vs-not AUC 0.610.**
  - Caveats: small figures (boxers ~250 px tall, so 10 cm is ~15 px, near RTMPose's jitter), 25 fps skeletons (every 2nd frame of 50 fps), puncher chosen with the referee's box.
- **Step 3 (running next): pixels.** 16 frames at 50 fps around contact, square crop centred on the puncher's glove (side 2.4 body scales), VideoMAE-L/B features on the HPC (`punchvid` env, weights cached), probe trained on one gym and tested on the other.

### Landed step 3: pixels at the contact point, frozen video features (`olympic/landed_probe.py`, 2026-11-14 night)
- 4,105 contact clips (`olympic/landed_clips.py`): 16 frames at 50 fps around contact, square crop of 2.4 body scales centred on the puncher's glove ("glove", run-time available) or on the referee's box ("box"). VideoMAE-L / -B (Kinetics-400, frozen) features via HPC jobs 299059/299060 (`hpc_rtm/landed_feats.py`, env `punchvid`, ~7 min) -> `external/landed_feats/`. Linear probe.
- **Train on one gym, test on the other (always "landed" = 62.1% accuracy):**

| input | 3-way accuracy | macro-F1 (landed / blocked / missed) | landed-vs-not AUC |
|---|---|---|---|
| skeleton cues | 57.1% | 0.433 (0.70 / 0.23 / 0.37) | 0.610 |
| pixels, VideoMAE-L, glove crop (class-balanced) | 55.2% | 0.482 (0.66 / 0.32 / 0.47) | 0.685 |
| pixels, VideoMAE-B, glove crop | 52.7% | 0.466 | 0.671 |
| pixels, VideoMAE-L, referee-box crop | 49.6% | 0.457 | 0.685 |
| pixels, VideoMAE-B, referee-box crop | 58.4% | 0.496 | 0.687 |
| pixels L + B averaged, glove crop | 55.6% | 0.494 (0.67 / 0.36 / 0.45) | 0.694 |
| pixels L + skeleton cues | 60.6% | 0.472 | 0.649 |
| pixels L+B concatenated, no class weighting | **65.7%** | 0.449 | **0.710** |

  - Leave-bouts-out with the gyms mixed: accuracy 56.6% (balanced), macro-F1 0.511, AUC 0.716: the cross-gym split is NOT the main limit.
  - Centring the crop on the referee's box is no better than on the glove: knowing the exact contact point is not the limit either.
  - As a confidence filter (no class weighting): calling "landed" only when P >= 0.7 keeps 50% of punches, 76.1% of which landed; P >= 0.8 keeps 29%, 80.3% landed (base rate 62.1%).
- **Verdict:** pixels help (AUC 0.61 -> ~0.70; blocked and missed F1 up), but frozen general-purpose video features with a linear probe are far from a usable landed/blocked/missed decision. The model has never been trained to look at glove-on-head contact.
- Not tried: fine-tuning the video model on these clips (the obvious next step; GPU job), higher-resolution / tighter crops, optical-flow or head-region features, sound, joint training with BoxingWeb "effective". FACTS (per the outside literature list, unverified) reports 83% on 8 classes from fine-tuned video on this data; its split is unknown.
- BoxingWeb "effective" from skeletons, final 3-seed averages (`runs/punch-attr*/`): with the opponent, landed macro-F1 0.559 +- 0.010, AUC 0.617; target head/torso macro-F1 0.688 +- 0.012, AUC 0.784. Without the opponent: landed 0.520 +- 0.011, AUC 0.563; target 0.670 +- 0.004, AUC 0.737. (BoxMind: effect 0.621 pose-only / 0.645 best; target 0.747 / 0.764.)

### Landed step 3b: FINE-TUNING the video model on the contact clips (2026-11-14 late night)
- User chose to invest in landed. Real-time note given: the video model runs only per detected punch on one small clip (not per frame), answers ~0.3 s after contact; a phone would need a small model (to be checked once the approach works).
- HPC: `hpc_rtm/landed_extract_clips.py` (job 299106) -> `/l/users/<user>/punch_rtm/clips/landed_glove_20x224.npy` (4,105 clips x 20 frames x 224x224, 12.4 GB; 3 clips had frames past a video end, filled). `hpc_rtm/landed_finetune.py` (jobs 299107/299108): **VideoMAE-Base** (MCG-NJU, Kinetics-400 checkpoint, 86M parameters, CC-BY-NC), all weights trained, new 3-way head; 16 of 20 frames, flip + zoom/shift + brightness/contrast; AdamW (backbone 3e-5, head 1e-3), 12 epochs, label smoothing, sqrt class weights, mixed precision; ~45 s per epoch, ~10 min per fold. Train on one gym (2 bouts held back to pick the epoch), test on the other. Outputs `external/landed_ft/` (git-ignored), scored by `olympic/landed_ft_score.py`.
- **Result, 4,105 punches, both folds pooled (always "landed" = 62.1%):**

| input | 3-way accuracy | macro-F1 (landed / blocked / missed) | landed-vs-not AUC |
|---|---|---|---|
| skeleton cues | 57.1% | 0.433 (0.70 / 0.23 / 0.37) | 0.610 |
| frozen video features + probe | 65.7% | 0.449 (0.78 / 0.22 / 0.34) | 0.710 |
| **fine-tuned VideoMAE-Base (best-validation epoch)** | **71.3%** | **0.596 (0.81 / 0.40 / 0.58)** | **0.784** |
| fine-tuned VideoMAE-Base (last epoch) | 72.0% | 0.612 (0.81 / 0.42 / 0.61) | 0.793 |

  - Confusion (truth rows landed/blocked/missed; predicted columns): [[2227, 159, 162], [259, 185, 46], [471, 81, 515]]. Of punches called landed, 75.3% landed; 87.4% of landed punches are caught. Blocked is the weak class (38% recall: more than half are called landed); 48% of misses are caught.
  - As a filter: "landed" when P >= 0.7 keeps 54% of punches, 81.1% landed, catching 71% of landed; P >= 0.8: 39%, 83.6%, 52%.
  - Test accuracy per fold peaked around epochs 4-9 and then plateaued while training loss kept falling (0.43-0.46): overfitting on ~1,400-2,100 training clips.
- VideoMAE-Large fine-tuning queued (jobs 299182/299183, bs 6, lr 2e-5, 10 epochs).
- Punch segments, puncher and hand still come from the labels here; not yet chained behind the punch detector.

### Landed: VideoMAE-Large fine-tuned (jobs 299182/299183) and laptop speed (2026-11-14 late night)
- Same recipe as Base, bs 6, lr 2e-5, 10 epochs; ~2-2.4 min per epoch, ~20-24 min per fold.

| fine-tuned model | 3-way accuracy | macro-F1 (landed / blocked / missed) | landed-vs-not AUC | called landed that landed | landed caught |
|---|---|---|---|---|---|
| VideoMAE-Base, best-validation epoch | 71.3% | 0.596 (0.81 / 0.40 / 0.58) | 0.784 | 75.3% | 87.4% |
| **VideoMAE-Large, best-validation epoch** | **74.2%** | **0.639 (0.83 / 0.45 / 0.65)** | **0.831** | 79.0% | 86.4% |
| VideoMAE-Large, last epoch | 76.0% | 0.656 (0.84 / 0.48 / 0.65) | 0.832 | 78.5% | 90.5% |

  - Large, confusion (truth rows landed/blocked/missed): [[2201, 108, 239], [236, 191, 63], [348, 67, 652]]: missed recall 61% (Base 48%), blocked recall 39% (Base 38%; 48% of blocked are still called landed).
  - Large as a filter: P >= 0.7 keeps 57% of punches, 83.5% landed, catching 76% of landed; P >= 0.8: 47%, 87.0%, 66%.
  - Always "landed" = 62.1%; skeleton cues 57.1%; frozen features 65.7%.
- **Speed measured on the user's MacBook, one 16-frame punch clip (random weights, batch 1):** Base 81 ms on MPS / 246 ms CPU; Large 220 ms MPS / 763 ms CPU. The model runs once per detected punch (at most ~1/s), so both keep up on a laptop GPU; neither is suitable for a phone (a small mobile video model would have to be trained the same way; not measured).
- User asked why a vision model again: only for landed, because skeletons carry no contact information (two failed attempts above); the rest of the chain stays skeleton-based.
- A pending HPC job "interactive" (299187) in the queue is the user's, not submitted by this session.
- Open: blocked vs landed; chain behind the punch detector (segments, puncher and hand still from labels); more training clips (overfitting after a few epochs); BoxingWeb "effective" with pixels; sound.

### NEW CHAIN on the Olympic bouts vs this morning's baseline, same footage, same scoring (`olympic/fullchain.py`, 2026-11-14 late night)
- RTMPose -> learned selector (pair rule) -> OSNet-AIN identity + carry -> per-boxer streams -> punch TCN trained on BoxingWeb only (3-seed ensemble, threshold 0.40 fixed from BoxingWeb validation). Nothing in the chain was trained on the Olympic bouts. Olympic streams are 25 fps; the detector was trained at 30 fps.

| pipeline | counted | found | chance (labels shifted 7 s) | counts matching a label | F1 | hand right |
|---|---|---|---|---|---|---|
| OLD (MediaPipe + old tracker + model from our 10 videos) | 2,390 (23/min) | 24.8% | 15.8% | 47.7% | 0.326 | 60.6% |
| **NEW chain** | 5,198 (50/min) | **79.4%** | 37.0% | **70.3%** | **0.746** | **79.7%** |

  - 4,605 labelled punches (44/min), 29 bouts, 104 min. Found by outcome: head 80%, body 79%, block 83%, miss 77%. Boxer skeleton present in ~81-96% of frames per bout.
  - Chance is higher for the new chain because it counts twice as many punches; found-minus-chance went from 9 to 42 points.
  - "Counts matching a label" is a lower bound on precision: unlabelled punches would count against it (user's standing caution).
- `hpc_rtm/lists/chain_clips.json`: 5,198 contact clips, one per counted punch (contact = frame of greatest extension of the punching arm, crop on that glove), for the landed model. HPC jobs submitted: extract + VideoMAE-L fine-tune per held-out gym that also judges the chain's clips and saves the model.

### Landed behind our own detector: it does NOT carry over yet, and why (`olympic/fullchain_landed.py`, 2026-11-14 ~23:10)
- HPC jobs 299268 (5,198 chain clips -> `clips/chain_20x224.npy`), 299269/299270: VideoMAE-L fine-tuned on one gym's referee-centred clips, then judging the chain's clips of the other gym; models saved under `/l/users/<user>/punch_rtm/models/`.
- 3,655 of the 5,198 counted punches match a referee label (62.8% landed, 12.2% blocked, 25.0% missed).
- **On those, the landed model is right 61.0% 3-way (always "landed" = 62.8%)**: missed caught 52%, blocked 28%, landed kept 71%. With the referee's contact frame and hand it was 74.2%.
- **Landed punches end to end (2,885 referee-labelled landed):**

| rule | called landed | landed found | landed real | F1 |
|---|---|---|---|---|
| every counted punch called landed | 5,198 | 79.5% | 44.1% | 0.568 |
| landed model, most likely class | 3,169 | 56.3% | 51.3% | 0.537 |
| landed model, P >= 0.7 | 2,633 | 48.7% | 53.4% | 0.509 |

  - So in the chain the landed step currently buys 7-9 points of precision for 23-31 points of recall: no net gain.
- **Diagnosis (matched punches):** our contact estimate (frame of greatest arm extension, 25 fps skeletons) vs the referee's contact frame: median +1 frame at 50 fps but 10th/90th percentile -10/+13; within +-2 frames only 31%, within +-5: 52%. The detector's hand agrees with the referee in 79.7%.

| subset | share | 3-way right | always "landed" |
|---|---|---|---|
| timing within +-2 frames and hand right | 28.2% | 72.4% | 55.4% |
| timing within +-5 and hand right | 46.9% | 71.0% | 60.1% |
| timing off by > 5 frames (0.1 s) | 47.7% | 50.7% | 64.9% |
| hand wrong (crop on the other glove) | 20.3% | 53.6% | 62.3% |

  - **The landed model is fine when it is shown the right moment and the right glove; the chain's contact timing and hand are what break it.** It was trained only on clips centred exactly on the referee's contact frame.
- Fix running (jobs 299317/299318, `--train-extra`, tag `chainx`): also train on the detector's OWN clips of the training gym, with the referee's outcome transferred from the matched label, so the model learns clips with our timing and crop errors. Other options not yet tried: a contact-moment estimator trained on the referee's contact frames, 50 fps skeletons, longer clips, averaging several time offsets at test.

### Landed in the chain, attempt 2: also training on the detector's own clips — NO GAIN (2026-11-14 23:40)
- Jobs 299317/299318 (`--train-extra`, tag `chainx`): VideoMAE-L trained on the referee-centred clips of one gym PLUS that gym's detector clips carrying a transferred referee label; tested on the other gym's detector clips. `runs/olympic_fullchain_landed_chainx.txt`.
- On counted punches that match a label: **3-way right 60.8%** (attempt 1: 61.0%; always "landed": 62.8%); missed caught 41%, blocked 30%, landed kept 75%.
- Landed end to end: most likely class: found 59.5%, real 49.7%, F1 0.542 (attempt 1: 56.3% / 51.3% / 0.537; every count called landed: 79.5% / 44.1% / 0.568).
- **Training on mistimed clips does not fix mistimed clips.** When the clip is centred a tenth of a second off, or on the other glove, the contact is simply not in view the way the model needs; the information has to be recovered upstream.
- **State of landed:** with the referee's contact frame and hand: 74.2% 3-way, AUC 0.83 (usable). Behind our detector: no gain over calling everything landed. The blocker is the contact-moment estimate (within +-2 frames in 31% of punches) and the hand (80%).
- Next (not started): a contact-moment estimator trained on the referee's 4,605 contact frames (pixels and/or 50 fps skeletons; RTMPose was only run on every 2nd frame), longer clips so timing matters less, several time offsets averaged at test, and scoring the landed model only where timing is trusted.

### Contact-moment model (`olympic/contact_tcn.py`, 2026-11-14 ~23:50)
- Chain streams cached: `olympic/chain_streams.py` -> `external/chain_streams/` (29 bouts; X incl. mirrored, M, K, frame).
- Per-frame "a punch of this boxer connects now, left / right hand" TCN (same network as the detector, 2 outputs), trained on the referee's contact moments (4,582 of 4,605 placed on a boxer stream: the stream whose labelled-hand glove is nearest the referee's box), sigma 1 frame at 25 fps, 2 seeds, train one gym / test the other. For each counted punch: frame + hand of the highest contact score inside the punch (grown by 4 frames), sub-frame peak.
- **On 3,631 counted punches matched to a referee label:**

| contact estimate | within +-2 frames (50 fps) | within +-5 | median error | hand right | +-2 AND hand right | +-5 AND hand right |
|---|---|---|---|---|---|---|
| greatest arm extension + detector's hand (before) | 26.7% | 49.0% | 5.5 frames | 80.1% | 24.5% | 43.8% |
| **contact model** | **48.2%** | **63.1%** | **2.5 frames** | 72.1% | **46.0%** | **59.0%** |

  - Timing nearly doubles at +-2 frames; the contact model's own hand is WORSE than the detector's (72% vs 80%). To try: take the peak in the detector's hand channel, or combine the two hand scores.
- `hpc_rtm/lists/chain_clips_v2.json` (same 5,198 punches, new contact frame, hand and glove crop). HPC: extract (`chain_v2_20x224.npy`) then `landed_predict.py` with the saved landed models (`ref` = trained on referee-centred clips, `chainx` = + detector clips), each clip judged by the model that never saw its gym -> `out/landed_v2_pred_{ref,chainx}.npz`. Score with `olympic/fullchain_landed.py "landed_v2_pred_ref.npz"` after pointing it at `chain_clips_v2.json`.
- User's idea logged: once contact timing is good, use it to cut BoxingWeb contact clips (6,500 punches with effective/target labels but no contact frame) and train the landed model on both datasets; the Olympic set itself cannot be expanded (already fully labelled).
- User asked whether to cut clips now or improve the contact model first. Expected gain from the v2 clips is small (~+2-3 points 3-way: "timing +-2 and hand right" went 24.5% -> 46%, landed model ~72% there, ~guessing elsewhere); the cut is kept as a cheap check of that arithmetic. **Overnight: RTMPose on EVERY frame (50 fps) of the 29 Olympic bouts** (jobs `rtm50-0/1`, `hpc_rtm/job50.sh`, `extract.py --step 1`, output `/l/users/<user>/punch_rtm/out50/`), so the contact model is not limited to 25 fps skeletons. Next for the contact model: 50 fps input, hand from the detector, pixels around the glove.

### Pixel contact + landed model: one network, 4 answers, scanned in time (2026-11-15 ~00:45)
- **Check of the skeleton contact timing (v2 clips, saved landed models):** on counted punches matching a label, 3-way right **65.2%** with the `ref` models (61.0% before; `chainx` models 63.3%; always "landed" 62.8%); landed end to end: found 62.3%, real 52.1%, F1 0.567 (every count called landed: 79.5% / 44.1% / 0.568). Matches the prediction (+2-4 points): timing is the lever.
- User: "shouldn't the contact model see pixels?" Yes. Skeleton-only contact = arm-motion guess (48% within +-2 frames). The 50 fps RTMPose run (jobs 299406/299407) was cancelled; one bout finished in `out50/`.
- Compute notes given to the user (accuracy first, optimise later; a lag of ~10 s is acceptable): the scan costs ~9-13 clips per punch; VideoMAE-B ~0.6 s of laptop-GPU work per second of fight at ~50 punches/min, VideoMAE-L ~1.7 s (needs a desktop/server GPU or a lighter scan); end-of-round processing always works. A 4 GB laptop GPU was discussed and parked.
- **New model (`hpc_rtm/contact_landed_finetune.py`):** VideoMAE-L, 4 classes per 16-frame clip: landed / blocked / missed (clip centred on the contact, +-1 frame) / off-centre (clip centred 4-14 frames early or late). Referee clips stored as 44 frames (contact at index 22), half of each batch shifted. After training: A. centred 3-way on the held-out gym; B. scan of referee clips (13 positions, -12..+12 frames, pick the lowest off-centre score) = pixel timing error; C. scan of the chain's own punches (`chain_clips_v3.json`: 44 frames around the skeleton model's rough contact, crop on the DETECTOR's hand glove), saved per position.
- Code path tested on the laptop with a toy model (`--tiny`) before submission. HPC: `cl-ext-ref`, `cl-ext-chain` (long clips, `extract_clips2.py`), then `cl-large-kam2/kam4` (`cl_job.sh`), outputs `out/cl_large_{gym}.npz`, models `models/cl_large_heldout_{gym}.pt`. ~1 h per fold expected.

### Night session (user asleep, laptop plugged in; decisions taken here are logged) — 2026-11-15 01:30
- User's instructions for the night: keep working and decide; after building the new (BoxingWeb) data, train with it and measure the improvement. Limits kept: no push to GitHub, nothing deleted.
- **Pixel contact + landed model, first fold finished (held-out kam4; kam2 finishing):**
  - A. centred 3-way on the held-out gym: 75.4% (kam2) / 73.3% (kam4), landed AUC 0.849 / 0.844: the 4th "off-centre" answer cost nothing (3-class model: 74.2%).
  - B. **pixel timing on referee clips: contact within +-2 frames in 84.6% / 81.1%** (+-4: 89.6% / 86.5%; skeleton contact model: 48%).
  - C. behind our detector (kam4 fold, 2,204 matched punches): timing within +-2 frames 50.0% (skeleton model 45.3%, arm extension 25.3%); **3-way right 66.2%** (65.2% with the skeleton model, 61.0% originally; always "landed" 62.5%).
  - Breakdown: detector's hand right AND rough moment within the scan's reach (+-12 frames): 61.8% of punches, verdict right **71.4%**; hand right but rough moment > 12 frames off: 16.7%, 51.8%; **hand wrong: 21.5%, 62.2%**. Rough-moment error percentiles 50/75/90: 3 / 12 / 18 frames. Part of the "far off" group is probably a neighbouring punch of a combination matched to the label.
- **Decisions:**
  1. Scan BOTH gloves for every counted punch and let the pixels choose the hand (`chain_clips_v3b.json`: same frames, crop on the other glove; jobs `ext-v3b` -> `scan-v3b`, `hpc_rtm/cl_scan.py`).
  2. BoxingWeb as extra training data (user's idea): `boxingweb/contact_clips.py` -> `hpc_rtm/lists/bw_clips.json`, 6,376 punches (172 skipped: no skeleton), 44 frames around the middle of each labelled punch, crop on the labelled hand's glove (median side 409 px: much closer footage than the Olympic set, 30 fps vs 50 fps). Jobs `ext-bw` -> `scan-bw-kam2/kam4`: the pixel model finds the contact inside each punch (11 positions, -10..+10); each Olympic fold uses the scan from the model that never saw its held-out gym.
  3. Joint training (`contact_landed_finetune.py --bw-json/--bw-npy/--bw-scan`): same network, second head "effective vs not" on BoxingWeb clips centred on their found contact (only where the scan's best centred score >= 0.5); half as many BoxingWeb clips as Olympic clips per step. Comparison = the run without BoxingWeb (above) vs with, same folds/epochs/seed: A, B, C, plus BoxingWeb test "effective" macro-F1 (BoxMind 0.621 / 0.645). Code path tested with `--tiny`.
- **Pixel contact scan behind our detector, BOTH folds (`runs/olympic_fullchain_scan.txt`, 3,631 matched punches):** timing within +-2 frames 51.7% (skeleton model 48.2%, arm extension 26.7%), +-5: 64.4%; **3-way right 66.5%** (65.2% skeleton timing, 61.0% original, 62.8% always "landed", 74.2% with the referee's moment and hand): missed caught 41%, blocked 33%, landed kept 83%. **Landed end to end: found 66.1%, real 52.1%, F1 0.583: the first time the landed step beats calling every count landed (F1 0.568).** Weighting the verdict over the scan instead of taking the peak: no difference (66.0%).
- **Both-gloves scan (`olympic/fullchain_scan_hands.py`, jobs `ext-v3b`/`scan-v3b`): NOT worth it.** Letting the higher "centred" score choose the hand switches 38% of punches and LOWERS hand accuracy (80.1% -> 63.5%; with a 0.1 margin: 75.8%): the two glove crops overlap and both show the contact area, so the score does not tell the gloves apart. 3-way right 66.5% -> 67.3-67.5%, landed end to end F1 0.583 -> 0.584. Decision: keep the detector's hand; do not double the scan cost.
- **BoxingWeb contact moments found by the pixel model (`scan_bw_kam2/kam4.npz`, 6,376 punches, 11 positions):** best centred score >= 0.8 in 93.5% (model heldout_kam2) / 87.7% (heldout_kam4); chosen contact relative to the middle of the labelled punch: 10/50/90 pct = -6/-2/+6 and -8/-2/+4 frames (30 fps); at the scan edge in ~10%. **The two independently trained models agree on the contact frame within +-2 frames for 79.4% of punches** (a sign the moments are real, not noise). Zero-shot, "effective" punches get a higher P(landed) than "ineffective" ones (0.83 vs 0.75; 0.78 vs 0.69) and are called missed less often (2.7% vs 9.9%).
- Joint training submitted (jobs 299758/299759, `cl_job.sh trainbw`, `--bw-min 0.8`): Olympic 4-class + BoxingWeb "effective" head, each fold using the BoxingWeb contact moments from the model that never saw its held-out gym. Outputs `out/cl_bw_large_{gym}.npz`.

### With vs without BoxingWeb as extra training data (user's request) — `olympic/compare_bw.py`, 2026-11-15 03:20
- Same folds, epochs, seed; "with" adds a second head trained on 3,993-4,288 BoxingWeb training punches centred on the contact moment the pixel model found (score >= 0.8), label effective / not.

| test (held-out gym) | WITHOUT BoxingWeb | WITH BoxingWeb | average of the two models |
|---|---|---|---|
| A. referee's moment + hand: 3-way right | 74.1% | 74.2% | **76.1%** |
| A. macro-F1 (landed / blocked / missed) | 0.629 (0.83 / 0.42 / 0.64) | **0.656 (0.83 / 0.47 / 0.67)** | |
| A. caught: landed / blocked / missed | 89% / 40% / 55% | 83% / **51%** / **64%** | |
| A. landed-vs-not AUC | 0.843 | 0.855 | 0.862 |
| A. share called landed (truth 62.1%) | 69.7% | 62.7% | |
| A. confident mode, P >= 0.7 | keeps 57%, 83.5% landed | keeps 47%, **88.2%** landed | |
| B. pixel timing on referee clips, +-2 frames | 82.8% | 83.9% | |
| C. behind our detector: 3-way right | 66.5% | 63.6% | 66.6% |
| C. landed-vs-not AUC | 0.710 | 0.712 | 0.720 |
| C. landed end to end (found / real / F1) | 66.1% / 52.1% / 0.583 | 59.4% / 53.6% / 0.564 | |

- **Reading:** BoxingWeb makes the landed model itself better balanced (blocked 40 -> 51%, missed 55 -> 64%, no longer over-calling landed, AUC +0.012, confident-mode precision +5 points) at the same overall accuracy; averaging the two models gives the best "A" so far (76.1%). **It does nothing for the chained result** (AUC 0.710 vs 0.712): behind our detector the limit is where we point the model (hand, rough moment), not how good it is. The lower chained accuracy "with" is the calibration effect: the baseline's habit of over-calling "landed" happens to pay when the clip is badly aimed.
- Side result: BoxingWeb test "effective" from PIXELS at the found contact: macro-F1 0.616 / 0.613, AUC 0.687 (our skeleton model: 0.559; BoxMind: 0.621 pose-only, 0.645 best).
- **Next decision (night):** remove the dependence on the hand. Centre the crop on the TARGET (the opponent's head/upper body) instead of the puncher's glove, so the same clip serves either hand and the crop does not jump with wrist jitter. Test on referee clips (A) and behind the detector (C) against the glove-centred baseline (74.1% / 66.5%).

### Target-centred clips vs glove-centred (night decision) — `olympic/compare_tc.py`, 2026-11-15 04:30
- Clips cropped on the opponent's shoulder midpoint, side 3x torso (`olympic/landed_clips_tc.py`; jobs 299810-299813, `cl_job.sh traintc`; outputs `external/landed_ft/cl_tc_large_kam*.npz`). Same folds, epochs, seed, same punches. 254 chain clips had no usable opponent skeleton and fell back to the glove crop.

| test (held-out gym) | glove-centred | target-centred | both averaged |
|---|---|---|---|
| A. referee's moment: 3-way right | 74.1% | 71.7% | **75.5%** |
| A. macro-F1 / landed-vs-not AUC | 0.629 / 0.843 | 0.614 / 0.817 | 0.648 / 0.847 |
| A. caught: landed / blocked / missed | 89 / 40 / 55% | 83 / 39 / 60% | 89 / 40 / 60% |
| B. pixel timing on referee clips, +-2 frames | 82.8% | 75.2% | |
| C. behind our detector: 3-way right (peak / weighted) | 66.5% / 66.0% | 64.2% / 65.5% | 67.9% / **68.3%** |
| C. landed end to end F1 (peak / weighted) | 0.583 / 0.584 | 0.563 / 0.573 | 0.586 / **0.591** |

- **Reading:** target-centring alone is WORSE everywhere (the wider crop makes the glove small; timing drops 82.8 -> 75.2%), so it is not a replacement and does not solve aiming. Averaged with the glove-centred model it gives the best chained numbers so far (68.3% 3-way, F1 0.591), a small gain (+1.8 points) for double the compute.
- **Where this leaves the landed step:** four different attempts at the chained result (skeleton timing, pixel scan, extra data, different crop) all land in 65-68%, against 74-76% with the referee's moment. The always-"landed" baseline is 62.8%. The remaining gap is the detector's rough moment and hand, not the landed model. Decision: stop tuning the landed model; next gains must come from the detector (hand 80%, moment >12 frames off for 17% of punches), which can be trained with Olympic labels (hand + contact frame) leaving one gym out.
- **Headroom check before spending more on aiming (from the breakdown logged earlier: hand right and moment within reach 62% of punches -> 71.4% right; moment >12 frames off 17% -> 52%; wrong hand 21.5% -> 62%):** a PERFECT hand would add at most ~2 points, a perfect rough moment at most ~3; both together cap the chained result near 71%. The skeleton contact model trained on Olympic hand labels already gave a worse hand (72%) than the BoxingWeb detector (80%). Decision: the landed step is declared at its usable level for this data (68.3% 3-way, landed F1 0.591 with the two-crop average; 66.5% / 0.583 with one model); no further aiming experiments tonight. The larger lever is the landed model itself (74-76% even with perfect aiming), which needs more labelled contact data or another signal (sound), a decision for the user.

### Landed step: user asks to improve it; options under test — 2026-11-15 09:30
- User ruled out: averaging more models (latency) and sound (variable, unreliable). Testing: bout totals, more-data check, same-gym check, wider time window; human ceiling prepared for the user.
- **Correction of my own statement:** the landed model does NOT see 25 fps. Clips are 44 consecutive 50 fps frames and the model takes 16 consecutive ones (0.32 s), at near-native resolution (crop median 294 px -> 224). So "50 fps / higher resolution" is not an untested option; the untested variant is the opposite, a WIDER window (every 2nd frame, 0.64 s: shows the reaction after the hit).
- **Bout totals (`olympic/landed_totals.py`, fully automatic chain, held-out gym for every bout, one scale factor fitted on the other gym):** 29 bouts, referee landed per bout median 100 (17-208).

| estimate of landed punches per bout | median bout error | within 15% | correlation with referee |
|---|---|---|---|
| every counted punch (no landed step) | 15.6% | 48% of bouts | 0.94 |
| counted punches called landed | 8.9% | 69% | 0.96 |
| sum of P(landed) | **7.6%** | 66% | 0.96 |

  Mean error 13.5-14% (a few bad bouts, 90th percentile 26-28%). Landed SHARE per bout (landed / thrown) follows the referee's with correlation 0.75. Two-crop average and the BoxingWeb-trained model: no better (7.9% / 12.2%). Uncalibrated, the soft total over-counts by 11.5%. **Reading: per punch the landed step is only ~5 points above "always landed", but for the number the counter reports (landed per bout) it halves the error.**
- HPC jobs 300151-300158 (`cl_job.sh trainx`, referee clips only): learning curve `--frac 0.25/0.5` per gym; same-gym split `--mix 0/1` (every 3rd bout of BOTH gyms held out; train 2,330 / 2,489 punches, like the cross-gym folds' 1,600 / 2,505); wider window `--stride 2` per gym. Outputs `external/landed_ft/x_*.npz`.
- **How others did it (read 2026-11-15):** BoxMind (arXiv 2601.11492v2, Table S3): "effect" is one of 4 outputs of a pose (TCN) + video (I3D) model with a pose-region mask and mixture-of-experts head; F1 0.620 video-only, 0.621 pose-only, 0.645 full; the lowest of their four attributes; not stated whether on true or detected punches. Stefanski/Kozak/Jach (Entropy 2024, the Olympic set's authors): frame-level punch detection only (F1 ~0.49 on punch frames), no landed verdict published. Jabbr DeepStrike (commercial): multi-camera, vendor-claimed 98-99% detection, unverified. No published single-camera per-punch landed/blocked/missed benchmark found.
- **Human ceiling test prepared (`olympic/human_test.py` -> `external/human_test/index.html`, not committed):** 90 referee punches, 30 per class, both gyms, 1.2 s clips (contact 0.7 s in), crop twice the model's so both boxers show; the page scores the viewer against the referee and against the model's answers for the same punches (model right on 68.9% of these 90, classes balanced). Waiting for the user to judge them.
- User feedback on the human test: some clips show one boxer blocking AND the other landing (exchanges), so it was unclear which punch to judge. Fixed: a yellow square (the referee's box) is drawn until 0.24 s before contact; answers restart (`ht2`). **Note for the model:** its clips have the same ambiguity in exchanges (the crop is centred on one glove, but nothing tells it which of two simultaneous punches is meant); worth measuring how many referee punches have another label within ~0.3 s and how the model does on them.
- **Planned solo route (user's idea, not started):** one boxer with a BOB dummy or punching bag -> detect the target, count punches and types, judge only touched / not touched (glove reaches the target's outline + the target's reaction). **User's rule: never switch routes just because only one boxer is found** (a boxer can be missed for a few seconds in sparring). The switch needs positive, sustained detection of the bag/BOB and no second boxer over a long window, decided per video or long segment. First test: is the target found reliably on B1 / C1 / C2.

### Whole chain on one video, rendered for the user to judge — `pipeline/count_video.py`, 2026-11-15 09:45
- One script: skeletons -> selector -> identity -> punch detector + type -> skeleton contact model (now trained on all 29 Olympic bouts, `runs/contact-tcn/model_all_seed*.pt`) -> 13-position pixel scan with `cl_large_heldout_kam2.pt` ON THE LAPTOP (MPS) -> verdict per punch -> annotated video with running counts per boxer (landed, thrown, landed by type). Clips are resampled to the model's 50 fps time base (nearest frame).
- Laptop model check: weights saved on the HPC use older parameter names (`q_bias`/`v_bias`), mapped to `query.bias`/`value.bias`. On 40 Olympic referee punches the laptop reproduces the HPC outputs: same verdict 40/40, mean probability difference 0.004 (max 0.045).
- **A2 sparring (5.3 min, 720p 30 fps, never trained on):** 137 punches counted (26/min). A: thrown 77, landed 63 (82%), blocked 13, missed 1. B: thrown 60, landed 37 (62%), blocked 20, missed 3. Types: straights 125 of 137. Scan: 280 ms per clip, 498 s for 137 punches x 13 positions; "centred" score median 0.89 (10th pct 0.48).
- **Caution:** only 4 of 137 called missed (Olympic: 26% of punches are misses) and A's landed share is 82%; either light sparring with few clean misses or the model over-calls landed on this footage (720p, 30 fps, other camera). No truth here: the user judges by eye.
- **Cosmetics in the rendered video (user's request):** health bar per boxer (100 points; a landed punch drains the boxer who was hit: L straight 3, R straight 5, L hook 6, R hook 7, L uppercut 6, R uppercut 8; blocked and missed deal none; FIXED, not scaled per video, "K.O." at zero: user's decision), impact flash + shock ring + sparks + screen shake for landed, hexagonal shield for blocked, streaks for missed, synthesised thump / slap / whoosh muxed as audio.
- **"Which punch" marker prepared (`olympic/landed_clips_arm.py`, `hpc_rtm/extract_clips3.py`, `cl_job.sh extract3 / trainarm`):** the puncher's upper arm (shoulder -> elbow) drawn as a thin magenta line in every clip frame; known in 94.7% of referee clip frames, 98.4% of chain clip frames. First version followed the wrist and sometimes jumped to the other boxer in exchanges (seen on a contact sheet); now follows the torso centre.
- **Learning curve (referee's moment, held-out gym, both folds pooled by test size):** 25% of the training punches (355 / 525): 3-way 62.3%, AUC 0.71; 50% (710 / 1,050): 70.8%, AUC 0.80; 100% (1,420 / 2,100): 74.1%, AUC 0.84. Still rising, flattening: +8.5 then +3.3 points per doubling. Reading: more labelled punches would still help, by roughly 1.5-2.5 points for the next doubling. Note the scored models train on ONE gym only; a deployment model trained on both gyms has twice the data (the same-gym split `mix_0/1`, running, estimates what that buys).

### Bag / BOB route (separate from the sparring chain) — `bagroute/`, 2026-11-15
- **Feasibility (side task, `bagroute/detect_target.py`, `analyse.py`, `runs/bagroute/summary.txt`):** OWLv2 (`google/owlv2-base-patch16-ensemble`, 0.4-0.5 s per frame on the laptop) finds the target on all three clips: BOB with "a training dummy torso" (100% of sampled frames), heavy bag and standing bag with "a punching bag" (100% / 99%). **"Bag detected" alone does not separate solo from sparring**: both sparring gyms have real bags in the background (A1 99%, A2 95% of frames, scores overlapping the true targets). What separates them is glove reach: gloves reach the true target in 52-78% of seconds, any sparring background bag in at most 19%. B1 has two BOBs and C1 two bags; the one with the highest reach is the one in use. This fits the user's rule (route chosen by positive, sustained evidence about the target, never by "one boxer found").
- **Skeletons for the bag clips on the laptop:** test segments cut to `external/raw/{B1_bob,C1_heavybag,C2_standingbag}_seg.mp4`; RTMPose via rtmlib in an isolated venv `.venv_rtm` (`bagroute/extract_local.py`, CPU, 13.5 fps).
- **Route v1 (`bagroute/count_bag.py`):** target = persistent detected place with the highest glove reach; boxer = largest skeleton not sitting on a detected place; punches + type = no-opponent punch TCN (`runs/punch-tcn-noopp`); contact = greatest arm extension; touched = glove point inside the target outline (+-2 frames), "clear" if the target's pixels move >= 1.35x more after than before.
  First run (target box from 2 s samples): B1 128 punches (43/min), glove inside the BOB for all but 2; C1 137 punches, reach 77%; C2 54 punches, reach 100%. Check sheets by eye: BOB contacts look right; on the swinging heavy bag the box lags (2 s sampling) and the reaction cue fails because the bag never stops moving (a clearly touching jab had ratio 0.9). Changes: detect the target every 0.5 s, "touched" = reach (reaction kept only as a confidence note), a "person" sitting on a detected place is never a boxer.
  Type caveat: on B1 the detector calls 54 of 128 punches "R uppercut" (they look like right hands to the body); type on bag work is not validated.
- **Bag route v2 results (target detected every 0.5 s; videos `runs/bagroute/{id}.mp4`, check sheets `{id}_check.jpg`, per-punch `{id}_punches.json`):**

| clip | punches counted | touched | no touch | notes |
|---|---|---|---|---|
| B1 BOB (3.0 min) | 128 (43/min) | 124 (97%) | 4 | the pose model draws BOTH BOB dummies as full people (17 keypoints, stand as legs) in every frame; ignored because they sit on a detected dummy; the idle BOB (reach 0%) is not chosen as target |
| C1 heavy bag (3.0 min) | 137 (46/min) | 120 (88%) | 17 | bag swings; reaction cue weak (ratio median 1.58); second bag at the edge not chosen |
| C2 standing bag (2.3 min) | 56 (24/min) | 56 (100%) | 0 | |

  No labels exist for these clips: the counts and touch calls are NOT validated beyond contact sheets judged by eye (24 punches per clip; on B1 all 24 look like real contacts, on C1 the remaining "no touch" calls include at least one real contact where the swinging bag's box lagged). On B1 the on-screen caption names the combination ("double jab-cross-R body hook") and the detector calls the right body hook "R uppercut": punch TYPE on bag work is unreliable.
  Important for routing: because a BOB is drawn as a person next to the boxer, the sparring chain would take it for a second boxer. So the route must be decided by the detected target (dummy/bag place + glove reach), exactly as the user asked, and never by counting people.
- **Same-gym split (`mix_0`, `mix_1`: every 3rd bout of BOTH gyms held out, train 2,330 / 2,489 punches):** 3-way 75.3% / 77.5% (mean 76.4%), landed AUC 0.855 / 0.867, pixel timing +-2 frames 84.7% / 85.3%. Against the cross-gym folds (74.1%, AUC 0.843, timing 82.8%): seeing the gym (and a little more data) is worth about +2 points. So the gym change is a small part of the gap; the amount of labelled data is the larger one (learning curve above). The deployment model trained on all 3,810 punches of both gyms (`cl_all_large`, queued) should sit at or above this level on these gyms.
- **Wider time window (`--stride 2`, 16 frames over 0.64 s): WORSE.** 3-way 71.2% (0.32 s window: 74.1%), landed AUC 0.788 (0.843), blocked caught 32% (40%), missed 48% (55%). The verdict lives in the few frames around the contact; spreading 16 frames over a longer time loses them. Decision: keep the 0.32 s window. (Its "timing +-2 frames 88-90%" is not comparable: the scan range was +-6 frames instead of +-12.)
- **Self-training on BoxingWeb considered and dropped before running:** the Olympic models' own confident verdicts on the 6,376 BoxingWeb punches are 92% "landed" (e.g. 4,786 landed / 117 blocked / 287 missed at confidence >= 0.7), so they would add almost nothing about the two classes the model is weak on and would push it further towards "landed". Not run.
- **Scan positions vs accuracy (existing chain scans, both folds):** 13 positions 66.5% / F1 0.583; **7 positions (step 4 frames) 66.3% / 0.583**; 5 positions 65.6% / 0.582; 3 positions 63.7-64.2%; 1 position 60.8% / 0.561. Decision: 7 positions in `pipeline/count_video.py` (about half the landed-step cost, no measurable loss).
- **"Which punch" arm marker: NO GAIN, dropped (`cl_arm_large_kam{2,4}.npz`, 2026-11-15 12:30).** Referee's moment, both folds: 3-way 73.9% (plain clips 74.1%), landed AUC 0.847 / 0.841. Per class it only moves the balance: landed kept 83% (89%), blocked caught 42% (40%), missed caught 66% (55%). On exchanges (another punch with a different outcome within 0.16 s, 564 punches) 59.2% (59.4%); within 0.30 s (1,229) 67.0% (65.3%). Behind our detector: 64.3% at the peak / 65.3% weighted (plain 66.5% / 66.0%), landed F1 0.568-0.578 (0.583). Pixel timing alone slightly better (+-2 frames 85.6% vs 82.5%). Reading: the model already knows which arm the clip is about; the errors in exchanges are not "wrong punch" errors. Decision: keep plain glove-centred clips; no arm drawing in `pipeline/count_video.py`.
- **Both-gyms landed model (`runs/landed_model/cl_all_large.pt`, HPC job `cl_all`):** trained on 3,810 referee punches from both gyms (2 bouts held out for stopping/checking: 295 punches). On those 2 bouts: 3-way 74.9%, AUC 0.847, pixel timing +-2 frames 86.4%. This is a small same-gym check (compare the same-gym split, 76.4%), not a new headline number; the model is the one the pipeline now uses because it has seen 1.8x the punches of either single-gym model.
- **A1 / A2 re-run with the both-gyms model and the 7-position scan (`runs/pipeline_demo/`, first version kept in `v1_kam2model/`), 2026-11-15 12:45.** Same punches (the detector is unchanged); only the landed verdict changes. Scan 251-267 ms per clip, 241 s (A2) and 475 s (A1), about half the first version.

| clip | boxer | thrown | landed v1 -> new | blocked v1 -> new | missed v1 -> new |
|---|---|---|---|---|---|
| A2 | A | 77 | 63 -> 55 | 13 -> 14 | 1 -> 8 |
| A2 | B | 60 | 37 -> 31 | 20 -> 21 | 3 -> 8 |
| A1 | A | 152 | 113 -> 76 | 32 -> 58 | 7 -> 18 |
| A1 | B | 102 | 82 -> 57 | 5 -> 19 | 15 -> 26 |

  The two models give the same verdict on 106 of 137 A2 punches (77%) and only 173 of 254 A1 punches (68%). **Caution: there is no truth on A1/A2, so this does not say which model is right.** The new model's landed share (A2 63%, A1 52%) is nearer the Olympic referee rate (62%) than the first version's (73%, 77%), and it no longer calls almost nothing "missed", but a one-in-three disagreement on A1 (dark gym, 720p) shows the verdict is unstable on footage unlike the training bouts. A few hundred hand-labelled sparring punches would be needed to measure it.
- **Bag route videos re-rendered in the sparring style (`bagroute/count_bag.py` render, 2026-11-15 15:00, user's request):** boxer skeleton, target box (red on a touch), flash + shock ring + sparks + screen shake at the glove for every touched punch, streaks for no touch, floating damage number over the target, synthesised sound, and a health bar for the TARGET: 100 points, the same fixed damage per punch type as the sparring route; at zero "K.O.!" is shown, a K.O. counter goes up and the bar refills (a bag takes ~120 touches in 3 min, so one bar would be empty after ~30 s). Videos: `runs/bagroute/{B1_bob,C1_heavybag,C2_standingbag}_seg.mp4`. Counts unchanged (B1 128 / 124 touched, C1 137 / 120, C2 56 / 56).

### The "which boxer threw it" rule was wrong for a third of the Olympic labels — found and fixed, 2026-11-15 15:30
- Started from the user's go-ahead to improve the chain without new labels. First try, **detector trained on BoxingWeb + the other gym's Olympic labels (`olympic/punch_tcn_oly.py`, old puncher rule): no gain** (found 73.6%, real 74.5%, F1 0.740 vs 0.746; hand 80.6% vs 79.7%; BoxingWeb test F1 0.786 -> 0.78).
- That run printed "right boxer 60%", which led to `olympic/boxer_match_check.py` / `credit_check.py`: around 48% of referee labels BOTH boxers have a counted punch, and each fighter's share of a bout's punches (referee vs counted) correlated only 0.19-0.23. Not caused by the opponent input (same without it).
- **Contact sheets (24 cases, looked at by the assistant): the referee's box sits on the TARGET (the head/body of the boxer being hit).** Every Olympic script so far took "the boxer whose labelled-hand glove is nearest the box" as the puncher; a defender's glove in guard is next to his own head, so the rule picked the DEFENDER often. On the sheet of "detector fired only on the other boxer", in 3 of 6 the detector was right and the rule wrong.
- **New rule (`punch_tcn_oly.puncher_stream`, `landed_clips_p.puncher`): target = the boxer whose head (torso for body shots) is nearest the box; puncher = the other.** The two rules agree on only 65% of labels. With the new rule: puncher has a counted punch around the label 84% (old rule 73%; blocked punches 85% vs 57%), "other boxer only" 8.7% (19.3%), and fighter A's share of thrown punches per bout correlates **0.89** with the counts (0.23).
- **What the wrong rule had contaminated:** (1) the landed model's referee training clips (`landed_clips.json`): crop centred on the defender's glove for **34.6%** of clips (head 34%, body 42%, block 48%, miss 28%); the contact was still in view (the defender's glove is by the target), which is why 74% held, but the chain's clips are always centred on the puncher's glove; (2) the skeleton contact model's training targets (wrong boxer stream for a third); (3) the arm marker (drew the defender's arm for a third) and target-centred crops (centred on the puncher for a third): **both "no gain" results are not valid tests of those ideas**; (4) the label-to-punch matching of the chain score.
- **Fairer chain score with the existing models (`olympic/fullchain_scan_rematch.py`): a label goes to the time-compatible counted punch whose glove is nearest the referee's box (one-to-one) instead of the first in time.** 3-way right **69.1%** (was 66.5%), landed end to end found 68.0% / real 53.6% / **F1 0.599** (0.581), pixel timing +-2 frames 59.2% (52.1%). Nothing changed in the system; part of the gap to 74.1% was the score crediting labels to the other boxer's punch.
- **Per-fighter totals (`olympic/landed_per_boxer.py`, new rule, one scale factor from the other gym):** thrown per fighter error median 10.9%, correlation 0.96; landed per fighter error median 17.5%, correlation 0.94. Each fighter's share of the bout: thrown 3.5 points off (correlation 0.89), landed 6.5 points off (0.70). The fighter who landed more by the referee is the one who landed more by us in 85% of the 20 bouts with a split of at least 55/45 (69% of all 29).
- **Running:** referee clips re-cut on the puncher's glove (`olympic/landed_clips_p.py` -> `landed_clips_p.json`, HPC `ext-p` 300505) and the landed model retrained on them, both folds with the chain scan (`p_kam2/p_kam4` 300506/7 -> `cl_p_large_kam*.npz`); contact model and joint detector retrained on the laptop with the new rule.
- **Retrained on the laptop with the corrected puncher rule (2026-11-15 15:50):**
  - Skeleton contact model (`olympic/contact_tcn.py`, `runs/olympic_contact_tcn_p.txt`): timing within +-2 frames **53.0%** (old rule 48.2%), +-5: 66.8% (63.1%), median error 2.0 frames (2.5); its own hand 74.8% (72.1%; the detector's hand, 80%, stays the one used). The pipeline's all-bouts version is retrained as `runs/contact-tcn/model_allp_seed*.pt` (the old `model_all_seed*.pt` kept). `chain_clips_v2.json` was rewritten by this run; the earlier file is `chain_clips_v2_oldrule.json`.
  - Detector + Olympic labels (`olympic/punch_tcn_oly.py --mode joint`, `runs/olympic_punch_tcn_oly_joint.txt`): found 74.6%, real 76.3%, **F1 0.754** (BoxingWeb-only 0.746), hand 80.8% (79.7%); BoxingWeb test F1 0.775-0.789 (0.786), type right 63.5% (65.9%). A small gain in precision for a small loss in type; NOT adopted (type matters to the user, and the gain is within what the threshold choice moves).
- **Queued on the HPC:** `p_all` (both-gyms landed model on the corrected clips -> `cl_p_all_large.pt`), and the arm marker redone with the puncher's arm (`landed_clips_arm.py --p` -> `landed_clips_armp.json`, jobs `ext-armp`, `armp_kam2/kam4` -> `cl_armp_large_kam*.npz`).
- **Landed model retrained on clips centred on the true puncher's glove (`cl_p_large_kam{2,4}`, 2026-11-15 16:50): WORSE, not adopted.** Referee's moment: 3-way 72.1% (old clips 74.1%), AUC 0.838 / 0.847, pixel timing +-2 frames 80.5% (82.8%). Behind our detector with the fair matching (`runs/olympic_fullchain_scan_rematch_cl_p.txt`): 3-way **65.2%** (old model 69.1%), landed F1 0.567 (0.599); it calls "missed" more (missed caught 58%, landed kept 73%).
  Why: the old (wrong-boxer) rule always took whichever same-hand glove was nearest the referee's box, so its crops were always tight on the contact (glove more than half a crop from the box in 1% of clips). The true puncher's glove from the skeleton is off the contact area in 7-10% of clips (pose errors at the moment of contact), so the corrected clips are noisier training data. The old rule was a wrong answer to "who punched" but a good answer to "where is the contact".
  Decision: keep `cl_all_large.pt` (old clips) as the pipeline's landed model; the corrected rule stays for everything that needs the boxer (per-fighter truth, contact model, label matching). `p_all` cancelled. Arm marker redo (`armp_*`) still running; it is compared against 72.1%.
- **State of the chain numbers after today (Olympic, held-out gym):** landed verdict behind our detector 69.1% 3-way, landed F1 0.599 (fair matching, same system as this morning); with the referee's moment 74.1%; bout totals 7.6% median error; per fighter 10.9% (thrown) / 17.5% (landed).

### User's go-ahead for the next improvements (bag route set aside) — 2026-11-15 17:30
Plan agreed: (1) reject counted punches that are not real punches, (2) punch type from pixels, (5) a faster landed model. (3) truth on sparring needs the user.
- **(1) Truth for "real punch" (`olympic/reject_false_counts.py`):** a counted punch is real if a referee label of the SAME boxer (corrected rule) is matched to it one-to-one. Real: 65.9% of the 5,198 counts. With this stricter, boxer-aware score the chain is: thrown found 74.4%, real 65.9%, F1 0.699; landed end to end found 62.5%, real 49.3%, F1 0.551; verdict 3-way right on real ones 68.4%.
- **Cheap rejector from signals the chain already has (gradient boosting on the pixel scan, contact score, detector confidence, reach, other boxer; trained on the other gym): AUC 0.776** (best single cues: scan "centred" 0.71, contact score 0.71, detector confidence 0.66). At the threshold that is best on the training gym it drops 7.3% of counts (74% of them not real, 2.9% of real punches lost): thrown F1 0.699 -> 0.706, landed F1 0.551 -> 0.554. **Not worth adopting.** Totals do not improve either (dropping counts: per-fighter thrown error 10.9% -> 13.9%): false counts are roughly proportional to real ones, so the one scale factor already absorbs them.
- **What the false counts are (contact sheet of 12 counts with no referee label within 0.6 s, judged by the assistant from stills):** about 5 clinch / in-fighting (skeletons tangled), 3 the other boxer's movement or a guard, 1 a referee break (arm stretched out), 2 look like real punches the referee did not label. No single cause.
- **Pixel version queued (`contact_landed_finetune.py --neg`, jobs `neg_kam2/kam4` -> `cl_neg_large_kam*.npz`):** an extra 2-way head "real punch / not" on the landed model, trained on the chain's own counted punches of the training bouts (`chain_clips_v3r.json`), no extra cost at run time. Decision rule: adopt only if clearly above AUC 0.776.
- **(2) Type baseline (`boxingweb/type_baseline.py`):** on the 1,201 labelled BoxingWeb test punches (true span, true boxer, true hand) the skeleton detector gets the family (straight / hook / uppercut) right in **72.9%** (straight 84%, hook 65%, uppercut 61%; macro 70.1%; always "straight" 43.8%); all 6 classes 65.7%. **Pixel type model queued** (`hpc_rtm/type_finetune.py`, job `type_l` -> `type_large.npz`, `models/type_large.pt`): VideoMAE-L on the BoxingWeb clips already on the HPC (16 frames at 30 fps around the middle of the punch), same train/val/test rounds.
- **(5) Smaller landed model queued (`base_kam2/kam4` -> `cl_base_kam*.npz`):** VideoMAE-Base, same 4-answer training and chain scan as the large one (Base was 71.3% vs 74.2% in the earlier 3-answer comparison and 2.7x faster on the laptop).
- **Arm marker redone with the true puncher's arm (`cl_armp_large_kam{2,4}`, 2026-11-15 17:30): still NOT adopted.** Referee's moment 3-way 72.8% (true-puncher clips without the arm 72.1%; plain old clips 74.1%); pixel timing +-2 frames 88.1% (80.5%; old 82.8%). Behind our detector, fair matching: 66.8% (65.2%; old 69.1%), landed F1 0.583 (0.567; 0.599). So the arm line does what it was meant to (it helps the model find the moment and gains +0.7 / +1.6 points over the same clips without it), but the true-puncher clips cost more than the arm gives back. The plain model stays.
- **HPC results for the three items (2026-11-15 19:15):**
  - **(2) Punch type from pixels (`type_large`): on its own no better, fused with the skeleton clearly better.** On the 1,201 BoxingWeb test punches: skeleton family right 72.3% (macro 69.1%; straight 83%, hook 65%, uppercut 59%), pixels 68.4% (macro 69.3%; straight 64%, hook 72%, uppercut 72%), **average of the two probabilities 77.0% (macro 76.3%; straight 81%, hook 73%, uppercut 74%)**, geometric mean 77.7%. The two agree on only 61% of punches and at least one is right on 87%: they make different mistakes (the skeleton over-calls straight, pixels see the bent arm). Caution: the equal weight is the natural default but the weights 0.3 / 0.7 (74.3% / 73.2%) were also looked at on the test set; and this is with the true span, boxer and hand. To adopt: one more VideoMAE-L clip per punch (the landed scan already costs 7).
  - **(1) Pixel "real punch or not" head (`cl_neg_large_kam{2,4}`): AUC 0.783 / 0.744, not above the cheap rejector (0.786 / 0.767 on the same punches).** Rank-averaging both: 0.817 / 0.782. Dropping the lowest 20% then keeps real share 75% (from 66%) but loses 8-10% of real punches: thrown F1 about 0.71 against 0.699. **Not adopted**; rejecting false counts is closed as "no cheap win" (the landed verdict of these models is unchanged: 75.1% / 74.0% at the referee's moment).
  - **(5) VideoMAE-Base landed model (`cl_base_kam{2,4}`): faster but worse, not adopted.** Referee's moment 3-way 74.2% / 73.5% (large 74.1% overall), but behind our detector with the fair matching 65.1% (large 69.1%), landed F1 0.574 (0.599): the smaller model is less tolerant of our rougher contact moment (median pixel timing error 2 frames against 0). Accuracy first, so the large model stays; Base remains the option if speed ever matters more (about 2.7x faster).
- **Type fusion behind our own detector (`boxingweb/type_chain.py`, `runs/type_chain.txt`, 2026-11-15 19:40):** on the 923 labelled BoxingWeb test punches that our detector counted (span and hand from the detector, hand right 86.6%): all 6 classes right 66.0% -> **67.6%**; family right when the hand is right 76.2% -> **78.1%**, macro 71.8% -> **74.8%** (straight 86 -> 83%, hook 68 -> 75%, uppercut 61 -> 66%). Pixels alone 71.1% (macro 71.1%). So the gain is real but smaller than with the labelled spans (72.3 -> 77.0%): about +2 points overall, and hooks / uppercuts are no longer swallowed by "straight". Cost: one more VideoMAE-L clip per punch (249 ms on the laptop; the landed scan is 7).
  **Adopted in `pipeline/count_video.py`**: a type clip per punch (16 frames at 30 fps around the middle of the detected span, crop on the detected hand's glove) is cut in the same pass as the landed clips; family = mean of skeleton and pixel probabilities, hand from the detector; `cls_det` keeps the detector's class, `_scan.npz` also stores `type`. Weights `runs/type_model/type_large.pt` (git-ignored, CC-BY-NC VideoMAE base, trained on BoxingWeb train rounds). Note: BoxingWeb training clips were native video frames (30 fps, a few rounds 24), the pipeline samples at 30 fps by time. Previous A1/A2 outputs moved to `runs/pipeline_demo/v2_before_type/`.
- **A1/A2 re-run with fused types and the retrained contact model (`model_allp`), 2026-11-15 19:40:** pipeline ran end to end (type model 286-331 ms per punch on the laptop). Same punches counted (137 / 254).
  - Types: A2 9 of 137 changed (8 hook -> straight, 1 uppercut -> straight; now 134 straight, 2 hook, 1 uppercut); A1 18 of 254 changed (10 hook -> straight, 4 straight -> uppercut, 4 straight -> hook; now 202 straight, 21 hook, 31 uppercut). On these clips the pixel model mostly REMOVES hooks, the opposite of BoxingWeb; no truth here to say which is right, and 98% straights on A2 is worth a look by eye.
  - Verdicts: A2 A landed 55 -> 52, blocked 14 -> 17, missed 8; B landed 31 -> 27, blocked 21 -> 27, missed 8 -> 6. A1 A landed 76 -> 86, blocked 58 -> 56, missed 18 -> 10; B landed 57 -> 61, blocked 19 -> 16, missed 26 -> 25. Only 99 of 137 (A2) and 195 of 254 (A1) punches kept their verdict although the landed model is the same: the only change is the rough contact moment from the retrained contact model. **The per-punch verdict on sparring is sensitive to a few frames of timing**; totals move by under 10%.

### Bag route: uncounted punches fixed, box follows the swinging bag, boxer bar added — 2026-11-15 20:00
User's feedback on the bag videos: "because the bag moves a lot many of the punches that land on it while moving don't count", and the BOB video should show a bar for the fighter AND the BOB "even if the fighter one wouldn't move".
- **Cause 1, the main one: the punch detector (trained on bouts) misses many punches on a bag, mostly jabs.** Check: glove comes fast into the target box while no counted punch covers that moment; contact sheets of these (looked at by the assistant): nearly all real punches on B1 and C2, 2 of 3 on C1.
  **Rescue rule (`bagroute/count_bag.py` analyse):** a glove that enters the target (was outside for 6 frames), faster than 2 target widths per second, moving towards it, outside every counted punch (+-6 frames), is counted as a punch; both gloves arriving within 4 frames = one punch; family from the detector's probabilities around that moment. Marked `rescued` in the json.
- **Cause 2: on C1 a second bag hangs behind the hit one; the box stayed on that place while the hit bag swung away** (the old box came only from detections on the usual place, 70% of the time). `follow()`: per detection time, the bag-sized detection near the place that is nearest the boxer (among those scoring >= 0.7 x the best). This alone changed the count little (120 -> 118 touched) but the box is now on the bag that is hit.
- Reach margin 0.10 -> 0.15 of the target width (three real touches at the bag's edge were "no touch" at 0.12-0.14).
- **Counts (before -> after):** C1 heavy bag 137 -> 144 counted, touched 120 -> 128 (7 rescued); B1 BOB 128 -> 163, touched 124 -> 162 (35 rescued); C2 standing bag 56 -> 84, touched 56 -> 84 (28 rescued). No truth; checked on sheets and one rendered frame only. Risk: the rescue rule can count a push or a glove resting on the bag as a punch.
- Render: BOXER bar (stays at 100) next to the BOB / BAG bar. Videos `runs/bagroute/{B1_bob,C1_heavybag,C2_standingbag}_seg.mp4`; previous versions in `runs/bagroute/v1_fixed_place/`.
- **User's decision (2026-11-15): no hand-labelling of sparring and no scoring on A1/A2**; they stay demo clips judged by eye, accuracy is quoted from the Olympic / BoxingWeb bouts.
- **BOB mode by command-line switch (user's choice instead of automatic routing, 2026-11-15 20:15):** `python pipeline/count_video.py ID --bob` (or `--bag`) runs the bag route (`count_bag.run`); without the switch the sparring chain runs. The bag route is no longer tied to the three test clips: any ID with `external/raw/ID.mp4` and `external/pose_rtm/ID.npz` works, and the bag / dummy detection (OWLv2 every 0.5 s) runs first if `runs/bagroute/ID_det.json` is missing (that branch is untested: no new bag clip yet). Tested: `B1_bob_seg --bob --no-render` gives the same 163 / 162.

### Arcade HUD and hit feedback (branch `arcade-hud`, cosmetics only) — 2026-11-15 20:45
User pasted a fighting-game HUD spec ("ignore the timer one and make this in a branch so if I don't like it we switch"). New module `pipeline/arcade_fx.py` (all numbers in its `P` table), used by the sparring renderer in `pipeline/count_video.py`; `main` keeps the old renderer.
- Built: mirrored slanted bars (12 degrees, gunmetal bezel, gold inner stroke, hatch background), ghost damage layer (holds 0.45 s after the last hit, drains 0.35 s), green / amber / red by health with a 0.2 s blend, gloss + shine sweep, bar shake / white flash / shards per hit, low-health pulse + glow + sparks + red name, K.O. crack and grey bar; landed tiers LIGHT (L straight) / HEAVY (R straight, hooks) / CRITICAL (uppercuts, and a counter = landing within 0.35 s of the opponent's punch that did not land): hit-stop, camera shake, punch-in zoom, screen flash, chromatic aberration, starburst, shock ring, speed lines, sweat, glove afterimages, combo counter ("N HITS", then hits + damage), callout ("CRUSHING!" / "COUNTER!"); blocked: blue shield arc, sideways sparks, contracting ring, 2 px jolt; tiered synthesised sounds; `--reduce-fx` (shake and flashes x 0.25).
- Left out because a video overlay cannot do it or it does not apply: round timer (user), round-win pips, squash-and-stretch / slide-back / glove bounce / dust (would need to deform the footage), slow motion (breaks sync with the sound), chip damage (a block costs nothing here), perfect block (no block input), crowd ducking (only a synthesised crowd swell). Hit-stop = the picture holds and then jumps on, so length and sound sync are kept. The struck "victim flash" is a soft disc at the contact point (no silhouette of the boxer is known). Kept from the old renderer: skeletons, the per-punch label, the count lines, grey streaks for a miss.
- Checked by eye on 6 frames of a 25 s test of A2 (bars, ghost, flash, burst, ring, combo, block ring all drawn). The bag route still uses the old renderer. Old-style A1/A2 videos kept in `runs/pipeline_demo/v3_old_hud/`.
- **Anime-size hit effects (user: "so exaggerated, like anime"; same branch, 2026-11-15 22:40):** in `pipeline/arcade_fx.py` the per-tier numbers are roughly doubled (hit-stop 0.07 / 0.15 / 0.28 s, shake 8 / 20 / 38 px, punch-in 5 / 13 / 26%, burst 1.2 / 2.1 / 3.3, sweat 20 / 50 / 100; the spec's values are kept in a comment), and added: an inverted high-contrast impact frame (HEAVY 0.07 s, CRITICAL 0.13 s), full-screen concentration lines pointing at the impact, a slash through the impact along the punch, long spikes behind the star, a second ring, radial zoom blur on CRITICAL, and word callouts on HEAVY too ("BAM!", "WHAM!", "POW!", "THUD!"). `--reduce-fx` switches the impact frame off and weakens the rest. Checked by eye on 9 frames of a 14 s test of A2. This goes past the spec's rule "effects never hide the fighters for more than 0.1 s" on CRITICAL hits (0.13 s impact frame).
- **Cosmetics batch 2 (user's list, same branch, 2026-11-15 23:00; all driven by real outputs):**
  - combo ranks next to "N HITS" (2 GOOD!, 4 GREAT!, 6 BRUTAL!, 8 UNSTOPPABLE!);
  - floating damage numbers ("-5") from each landed hit, sized and coloured by tier;
  - "GUARD!" (blue) on a blocked verdict and "WHIFF!" (grey) on a missed one, one word per burst;
  - glove trails while a counted punch is in flight, coloured by punch family (straight blue, hook orange, uppercut magenta);
  - pulsing red vignette on the side of a boxer below 25% health;
  - K.O. finish when a bar first reaches zero: grey-out, screen cracks from the impact, letterbox, giant "K.O." and "BOXER X IS DOWN", boom + triple bell (2.4 s, the footage keeps running so sound stays in sync);
  - picture-in-picture slow-motion replay (x1/4) of each critical hit, bottom right (at most two queued in a burst);
  - intro card (names slide in, VS, FIGHT! + bell; 2.6 s) and results card (6 s: winner and how, then landed / thrown / accuracy / blocked / missed / biggest combo / best combo damage / health left / landed by type, better value in gold). Winner rule: first K.O., else more health left, else more landed punches, else draw. `--no-cards` turns the cards off; the original audio and the effects track are delayed by the intro length.
  - **Dropped:** fighter silhouettes. They need a person-mask model on every frame: 1.1 s per frame on the Mac's CPU (MPS was slower), about 3 h per video; the user said to skip it if it takes too long.
  - Test renders checked by eye: A2 165–177 s (intro, FIGHT!, K.O., replay, results) and 199–206 s (WHIFF!, GUARD!, damage, trail, vignette, 4 HITS GREAT!). Fixes after the first look: two misses at once printed "WWHIFF!", and GUARD! / the combo rank were hidden behind hit words (hit words now start lower). Note: health is a cosmetic score from landed punches by type, so in A2 both bars reach zero (B at 169.9 s, A at 231.6 s); the results card names the first K.O.
