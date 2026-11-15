# 3D pose from the original videos: report

Branch `video3d`, 2026-11-12. All 10 dataset videos were recovered from YouTube, re-processed with
MediaPipe Pose Landmarker (heavy), aligned frame-exactly to the existing 2D skeletons, and cut into
a drop-in 3D dataset whose rows match `punch_data.load_dataset()` one to one. **No relabelling is
needed.**

## TL;DR

- **All 10 videos PASS alignment, every one at frame offset 0** (no time scaling, no drift).
  The dataset's frame numbers are 0-based indices of the decoded YouTube video frames.
- **`RGB_videos/Meta_data.ods` has two errors** (found and fixed here, see section 4):
  1. the links for **V2 and V3 are swapped**;
  2. the **V1 link (`3ov6zVAXHJI`) is the wrong video**. Same channel (NateBowerFitness), same gym,
     but not the source. The true source is **`kbgkeTTSau8`** ("Most EFFECTIVE 30 Minute Boxing Heavy
     Bag HIIT Workout"), found by searching the channel.
- 3D data: `Skeleton3D_data/` (56 MB, all files < 10 MB, committed). Load with
  `from punch_data3d import load_dataset_3d`.
- 99.9% of the 55,818 annotated real frames have a detection that matches the original 2D within
  half a torso length; median 2D disagreement is 10.4 px on a 1280x720 frame.
- **Caveats for the main workstream:** wrists and elbows are often low-visibility in side-on videos
  (V1: 48% of wrist values below 0.5 visibility). MediaPipe's 3D arm lengths vary about 8 to 16%
  within a clip, so depth on the arms is noisy (section 6). **V4 is a genuine southpaw**, not a
  mirrored video (section 5).

## 1. What was done

| Step | Script | Output |
|---|---|---|
| Download 10 videos (+ true V1) | `video3d/download.sh` (HPC login node, yt-dlp) | `video3d/raw/videos/*.mp4` (git-ignored; also on HPC `/l/users/<user>/punch_video3d/video3d/raw/`) |
| Manifest | `video3d/make_manifest.py` | `video3d/manifest.json`: URL, YouTube id, title, format, codec, resolution, fps, frame count, duration, bytes, sha256 per file; model URL and sha256; dataset-to-file mapping |
| MediaPipe on every frame | `video3d/extract_pose.py` (HPC, CPU job, 16 CPUs, 39 min for 312k frames + 6 min for V1_true) | `video3d/raw/pose/<file>.npz` (git-ignored, ~350 MB): up to 3 people per frame, 33 image landmarks (x, y, z, visibility, presence) + 33 world landmarks |
| Find the source of every dataset video | `video3d/crossmatch.py`, `video3d/find_source.py` | `video3d/crossmatch.{json,log}`, `runs/v3d-findsrc-V1/` |
| Frame alignment + pass/fail | `video3d/align.py` | `video3d/alignment.json`, `video3d/alignment.log` |
| Cut into clips | `video3d/build_dataset3d.py` | `Skeleton3D_data/`, `video3d/build_summary.json` |
| Loader | `punch_data3d.py` | `load_dataset_3d()` with row-match assertions |
| QC | `video3d/qc.py` | `video3d/qc/` (summary JSON, alignment-profile plot, PNG + GIF per video) |

HPC logs of every job: `runs/v3d-extract/`, `runs/v3d-extract-V1true/`, `runs/v3d-findsrc-V1/`.
`hpc.py` gained `--no-gpu` and `--env` options on this branch. `video3d/raw/` is excluded from its
`--delete` push. All HPC work used `HPC_REMOTE=/l/users/<user>/punch_video3d`. The QOS
caps a user at under 32 CPUs, so extraction used 16, leaving room for the other workstream's
8-CPU GPU jobs.

**Tools and versions:** yt-dlp 2026.10.26 (the Mac's 2026.06.09 got HTTP 403 from YouTube on
every video); MediaPipe 1.2.3 Tasks API, `pose_landmarker_heavy.task` (float16,
sha256 `64437af8...cbc7b`), VIDEO running mode, `num_poses=3`, default confidences 0.5; OpenCV
4.11.0 (sequential decode); numpy 1.26.4; pandas 2.2.2; Python 3.12. HPC env `punch3d` (new; the
`punch` env was not touched). Locally the formerly empty `/opt/anaconda3/envs/mediapipe` env now
has mediapipe, opencv-python-headless, pandas 2.2.2, matplotlib, pillow. Use pandas 2.2.x:
pandas 3 changes string dtypes, which breaks `punch_data._read_annotations`' `dtype == object` test.

**Why MediaPipe:** it was the default, it gives metric world 3D directly per frame, it runs in real
time on a phone/laptop (matches the app goal), and its 2D agrees with the existing skeletons to
about 1% of the frame width (below). No alternative was needed.

## 2. Downloads and coordinate convention

All videos were available. Each is offered by YouTube at a single native frame rate. Downloaded as
format 136 (1280x720 H.264, square pixels, constant frame rate). Aspect 16:9 for every video.

| Dataset | Video file | YouTube id | fps | Frames | Duration |
|---|---|---|---|---|---|
| V1 | V1_true.mp4 | kbgkeTTSau8 | 24 | 46,113 | 32:01 |
| V2 | V3.mp4 | Cx6JfrRO4eA | 23.976 | 33,927 | 23:35 |
| V3 | V2.mp4 | JGLLJUgm-0c | 23.976 | 32,611 | 22:40 |
| V4 | V4.mp4 | IVvOmoB88FM | 23.976 | 35,363 | 24:35 |
| V5 | V5.mp4 | V59fcX1YC7E | 29.97 | 15,519 | 8:38 |
| V6 | V6.mp4 | cQW-h8yGbRc | 23.976 | 46,705 | 32:28 |
| V7 | V7.mp4 | sqU3DmLppcQ | 29.97 | 28,062 | 15:36 |
| V8 | V8.mp4 | tSNybMUpMKk | 29.97 | 24,081 | 13:24 |
| V9 | V9.mp4 | WuBPbI0x50o | 23.976 | 22,541 | 15:40 |
| V10 | V10.mp4 | OfZFdwW322E | 25 | 26,917 | 17:57 |

(`V1.mp4` = `3ov6zVAXHJI`, the wrong link, is kept for the record; it matches nothing.)

**Coordinate convention, verified:** least-squares fit dataset = a * MediaPipe + b over all
aligned joints. x: a = 0.95-1.00 (V4 0.90), b = 0.00-0.04; y: a = 0.85-0.97, b = 0.01-0.06
(`alignment.json` -> `coord_fit_dataset_from_mediapipe`). Slopes slightly below 1 are the expected
regression dilution from per-frame noise (strongest on y and in V4, where the range is smallest),
not a scale change: the median absolute error is about 1% of the frame. **Conclusion: the dataset is
x/width and y/height of the full 16:9 frame, origin top-left, the same as MediaPipe's image
landmarks.** Values slightly above 1 in the dataset are extrapolated out-of-frame joints.

## 3. Alignment method and pass rule

For each dataset video, a map {original frame number -> dataset 2D} is built over all annotated
real frames. A consistency check first: in every video, overlapping annotations are bit-identical
on shared frame numbers, so all clips were cut from one continuous per-frame stream. Clip frames
are absolute frame numbers.

Cost at offset d = per frame, over the (up to 3) detected people, the minimum mean distance over
shoulders, elbows, wrists and hips (COCO 5-12) between dataset 2D at frame f and MediaPipe 2D at
video frame f + d; score = median over frames. Face points are defined differently across
estimators, and knees/ankles are often out of frame, so those are reported (`median_err_px_all17`)
but not used to align.

1. coarse search d in [-3000, 3000] on 1,500 random frames; fine search +-30 on all frames;
2. drift / fps check: best offset separately in 8 time segments, slope of offset vs frame;
3. per-clip local offset in [-3, +3];
4. affine coordinate fit (above); mirror test (flip x, swap left/right).

**Ground truth (V6):** V6 stores original frame numbers explicitly. Its annotated frames give
offset 0, and the full 46,497-frame stream gives offset 0, 10.1 px, all 8 segments at 0. The last
stream frame (46,704) is exactly the last decoded frame index of the 46,705-frame video. So the
method recovers the known answer, and the frame numbers are 0-based decoded-frame indices.

**Pass rule** (all must hold; `align.py: verdict`):

| Criterion | Threshold | Why |
|---|---|---|
| Sharp | error at the best +-1 neighbour >= 1.1 x the minimum | the offset is resolved to one frame; at 24-30 fps punches move ~5 px per frame |
| Unambiguous | best offset > 10 frames away >= 1.5 x the minimum | rules out repeated motions matching by chance |
| Stable | all 8 time segments give the same offset | no drift, no fps mismatch, no edits |
| Tight | median error <= 0.15 torso lengths | two different estimators on the same frame differ by ~0.05-0.10 torso (measured: V6 ground truth 0.07). A wrong frame or person is >= 0.3 |
| Consistent | >= 80% of annotated frames within 0.25 torso | the agreement holds for most frames, not only on average |

## 4. Per-video results

| Dataset | Source file | Available | Res. | fps | **Offset** | Median 2D err (norm / px / torso) | Mean / p90 px | Err at -1 / +1 px | Frames < 0.25 torso | Detect rate | Clips at local offset 0 (+-1) | **Result** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V1 | V1_true (kbgkeTTSau8) | yes | 1280x720 | 24 | **0** | 0.0137 / 12.2 / 0.08 | 14.2 / 23.5 | 17.8 / 17.2 | 98.3% | 99.99% | 93% (99%) | **PASS** |
| V2 | V3.mp4 (link swap) | yes | 1280x720 | 23.976 | **0** | 0.0096 / 8.6 / 0.07 | 10.9 / 19.3 | 11.8 / 11.5 | 96.4% | 100% | 88% (96%) | **PASS** |
| V3 | V2.mp4 (link swap) | yes | 1280x720 | 23.976 | **0** | 0.0093 / 8.4 / 0.07 | 10.6 / 19.1 | 13.0 / 12.5 | 98.9% | 100% | 96% (100%) | **PASS** |
| V4 | V4.mp4 | yes | 1280x720 | 23.976 | **0** | 0.0123 / 11.3 / 0.10 | 14.4 / 25.5 | 15.6 / 15.6 | 92.2% | 99.67% | 88% (98%) | **PASS** |
| V5 | V5.mp4 | yes | 1280x720 | 29.97 | **0** | 0.0080 / 7.2 / 0.05 | 10.9 / 23.0 | 15.4 / 16.0 | 96.8% | 100% | 98% (100%) | **PASS** |
| V6 | V6.mp4 | yes | 1280x720 | 23.976 | **0** | 0.0128 / 10.8 / 0.07 | 12.1 / 17.6 | 15.3 / 15.2 | 99.7% | 100% | 99% (100%) | **PASS** (ground truth) |
| V7 | V7.mp4 | yes | 1280x720 | 29.97 | **0** | 0.0115 / 10.0 / 0.06 | 12.5 / 20.7 | 17.3 / 17.0 | 98.4% | 100% | 100% (100%) | **PASS** |
| V8 | V8.mp4 | yes | 1280x720 | 29.97 | **0** | 0.0100 / 8.6 / 0.06 | 10.6 / 17.3 | 16.0 / 16.3 | 99.2% | 100% | 99% (100%) | **PASS** |
| V9 | V9.mp4 | yes | 1280x720 | 23.976 | **0** | 0.0087 / 7.7 / 0.05 | 7.9 / 10.3 | 12.4 / 12.8 | 100% | 100% | 100% (100%) | **PASS** |
| V10 | V10.mp4 | yes | 1280x720 | 25 | **0** | 0.0104 / 8.7 / 0.08 | 9.5 / 13.2 | 12.7 / 11.9 | 99.9% | 100% | 98% (100%) | **PASS** |

Norm = mean joint distance in normalised units (x/width, y/height); px = in 1280x720 pixels.
Drift slope = 0 for every video. No time scaling (each video's native fps is what the
annotations use). Profile plot: `qc/alignment_profiles.png`. Every curve has a sharp V at 0.
The per-clip "local offset +-1" misses are single short clips where +-1 frame is within noise.

**The metadata errors, and how they were found** (`crossmatch.py`, `find_source.py`). With the links
as given, V1, V2 and V3 failed every criterion (no offset within +-3000 frames; median error
0.4-0.9 torso; logged in the first block of `alignment.log`). A full 10 x 10 cross-match (40
clips per dataset video, each slid over the whole of every video, RANSAC line fit with slope
0.95-1.05) gives (`crossmatch.log`):

```
dataset V1: no video above 2/40 (chance)
dataset V2: video V3.mp4 20/40   (all others <= 3)
dataset V3: video V2.mp4 30/40   (all others <= 2)
dataset V4..V10: own video 21-40/40, all others <= 3
```

So V2/V3 are swapped. Dataset V1's skeleton plainly comes from the same gym as the V1 link, but
its hip track (parked at one bag for minutes) does not match that video at any offset, slope or
mirror. I scanned the NateBowerFitness channel: 152 heavy-bag videos >= 24 min; 70 could be
scored, 77 were members-only, 2 needed cookies. Each candidate was scored on 60 sampled
annotated frames at offset 0 (positive control V6 -> 10.3 px). **`kbgkeTTSau8` scored 13.7 px with
80% of frames < 25 px; the next best was 111 px, and the metadata link 145 px.** Full extraction
then gave the PASS above.

## 5. Other findings

- **V4 is a genuine southpaw, not a mirrored video.** It aligns unmirrored (11 px vs 246 px
  mirrored), and the QC renders show the jab thrown with the right hand (`qc/V4_clips.png`). This
  confirms the project's stance-canonicalisation decision rather than a data flip.
- **The original 2D was very likely made by a similar top-down estimator on the same frames:**
  median disagreement is about 1% of the frame width, comparable to the jitter of a single estimator.
- Multiple people: only 0-0.4% of annotated frames had more than one detected person (V1 0.43%, V4
  0.37%), and the dataset-matching person was a different candidate than MediaPipe's first in
  <= 0.3% of frames. The person choice is recorded per frame (`meta['person']`).

## 6. Quality checks (`qc/qc_summary.json`)

**Detection and match** (annotated real frames): no detection 0% everywhere except V4 (0.3%).
`frame_ok` (detected and within 0.5 torso of the original 2D) 99.0% (V4) to 100%. Clips with
every frame ok: 95% (V4), 98.5% (V6), >= 99.3% elsewhere.

**Low-visibility joints** (fraction of detected joint values with MediaPipe visibility < 0.5):

| Video | Wrists | Elbows | Knees | Ankles | Shoulders / hips / face |
|---|---|---|---|---|---|
| V1 | **48%** | 43% | 20% | 7% | 0% |
| V2 | 31% | 36% | 12% | 1% | 0% |
| V3 | 31% | 36% | 6% | 1% | 0% |
| V4 | 29% | 40% | 10% | 1% | 0% |
| V5 | 21% | 22% | 2% | 1% | 0% |
| V6 | 15% | 27% | 17% | 7% | 0% |
| V7 | 19% | 23% | 4% | 2% | 0% |
| V8 | 18% | 24% | 14% | 9% | 0% |
| V9 | **1%** | 5% | 3% | 1% | 0% |
| V10 | 10% | 16% | 15% | 8% | 0% |

As expected, the side-on videos (V1-V4) have far more occluded wrists and elbows than the frontal
ones (V9, V10). MediaPipe still outputs a position for these joints (guessed from context), so
low visibility means uncertain, not missing. Use `conf` as a weight or mask if needed.

**3D bone-length stability** (within-clip coefficient of variation of world-space bone length,
median over clips, ok frames only):

| Video | Upper arm | Forearm | Shoulder width | Hip width | Thigh | Shin |
|---|---|---|---|---|---|---|
| V1 | 11.0% | 15.2% | 5.5% | 3.8% | 3.8% | 2.9% |
| V2 | 8.3% | 10.8% | 5.1% | 1.9% | 2.0% | 1.7% |
| V3 | 7.6% | 8.6% | 4.9% | 2.1% | 2.1% | 1.8% |
| V4 | 9.3% | 11.6% | 2.9% | 4.4% | 3.0% | 2.2% |
| V5 | 7.5% | 10.1% | 4.1% | 3.3% | 3.1% | 2.3% |
| V6 | 7.6% | 10.3% | 3.9% | 4.3% | 3.7% | 2.5% |
| V7 | 10.9% | 12.1% | 5.0% | 4.1% | 3.4% | 2.4% |
| V8 | 10.0% | 11.4% | 4.9% | 4.8% | 3.7% | 2.7% |
| V9 | 7.6% | 9.8% | 3.4% | 3.2% | 2.4% | 2.3% |
| V10 | 11.6% | 16.2% | 5.5% | 4.9% | 3.6% | 2.8% |

Legs and torso are stable (2-5%); **arms vary 8-16% within a punch**, worst in V1 and V10. That is
MediaPipe's depth uncertainty on fast, extended or occluded arms. Median world lengths are plausible
and consistent across videos (upper arm 0.21-0.24 m, forearm 0.19-0.22 m, shoulder width
0.26-0.36 m). MediaPipe world coordinates are not true metric for a specific person: the model
predicts a body of roughly average size.

**Visual checks** (`video3d/qc/`): `V{v}_clips.png` (2 clips per video; V6 has only hooks/uppercuts,
so 1). Green is the dataset 2D, red MediaPipe 2D, on the actual frame, plus MediaPipe 3D below.
`V{v}_clip.gif` animates one clip. In all of them the two 2D skeletons sit on top of each other and
on the boxer. The 3D skeletons are plausible.

## 7. How to load and use

```python
from punch_data3d import load_dataset_3d
X3d, X2d_new, conf, y, video_id, lengths = load_dataset_3d()           # 5,428 rows
# or with diagnostics:
X3d, X2d_new, conf, y, video_id, lengths, meta = load_dataset_3d(return_meta=True)
```

- `X3d` (N, 25, 17, 3): MediaPipe **world** landmarks in metres, COCO-17 order (as `punch_data.KEYPOINTS`).
  Origin = hip midpoint, recomputed every frame (so global translation is removed). Axes follow the
  **camera**: x = image right, **y = down**, z = depth (negative = towards the camera). The skeleton
  is NOT rotated to face the camera; that's the main use: you can now rotate every skeleton to a
  canonical view (e.g. make the shoulder or hip line perpendicular to the viewing direction about
  the vertical axis) and remove the side-on/frontal difference the 2D models struggle with.
- `X2d_new` (N, 25, 17, 2): MediaPipe 2D, same convention as `Skeleton_data` (x/width, y/height).
- `conf` (N, 25, 17): per-joint visibility 0-1.
- `y`, `video_id`, `lengths`: **identical** to `punch_data.load_dataset()`. The loader asserts that
  labels, video ids, lengths and the original 2D all match, and that padding frames are zero.
- Real frames first, zero padding after, same start..end inclusive convention; V6 cut exactly as
  `punch_data._clips_from_stream`.
- `meta`: `X2d_orig`, `frame_ok`, `match_err_px`, `video_frame` (0-based frame in the source video,
  -1 = padding), `presence`, `person`, `aligned`. `drop_unaligned=True` exists for any future video
  that fails alignment; currently every video is aligned, so it changes nothing.
- Normalising: like the 2D pipeline, scale by the clip's median torso length if you want
  person-size invariance; for the 3D view-normalisation, the hip line (11 -> 12) in x-z is the
  most stable reference (hip width CV 2-5%).
- Raw files on disk: `Skeleton3D_data/V{v}.npy` (3D), `V{v}_2d.npy`, `V{v}_conf.npy`, `V{v}_meta.npz`.
  Largest file 9.5 MB.
- Works with the base Python (`/opt/anaconda3/bin/python`, numpy + pandas 2.2); no torch, no MPS.

## 8. What the main workstream must know

1. **Nothing to exclude**: all 10 videos pass with frame-exact alignment.
2. **Fix the metadata**: `RGB_videos/Meta_data.ods` links for V1 (-> `kbgkeTTSau8`) and V2/V3
   (swapped). `video3d/manifest.json` -> `dataset_to_video_file` has the correct mapping. I did not
   edit the `.ods` itself.
3. Arms in 3D are the noisiest part (depth on arms). Expect the 3D to help mostly through a view
   normalisation of the torso/shoulders, less through exact wrist trajectories. Weigh by `conf`.
4. V4 really is a southpaw (keep mirroring it to orthodox).
5. Regenerating: `bash video3d/download.sh` (HPC) -> `python hpc.py --no-gpu --env punch3d ...
   video3d/extract_pose.py` -> `video3d/align.py` -> `video3d/build_dataset3d.py` -> `video3d/qc.py`.
   Re-downloads are verifiable against the sha256 in `manifest.json`. YouTube may serve a
   re-encoded file in future; the alignment check would catch any frame changes.

## 9. Uncertainties / known limitations

- The V1 source identification rests on 2D matching (13.7 px vs next best 111 px, then a full PASS
  on 18,561 frames at offset 0, with all segments agreeing). That is strong evidence, but no
  metadata from the annotators confirms it.
- 79 channel videos could not be scored (members-only / cookies). This doesn't matter now that
  the source is found.
- MediaPipe's world z is a learned estimate from a single view; its absolute depth is not reliable,
  and arm length jitter (above) shows its limits. No ground-truth 3D exists to validate it.
- Visibility is MediaPipe's own estimate, not checked against human judgement.
- The pass thresholds were chosen by me before seeing the full results, guided only by the V6
  ground truth and the smoke test. All videos pass them with wide margins (worst: V4,
  0.10 torso, 92% within 0.25 torso).
