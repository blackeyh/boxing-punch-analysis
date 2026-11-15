# Shared window definition for punch detection (both detectors MUST use this)

- Unit: frames of the ORIGINAL downloaded videos, 0-based decoded frame index (the same indices
  the annotations use; verified frame-exact by the video3d alignment, offset 0 for all videos).
- Dataset video -> downloaded file (from video3d/common.py VIDEO_OF):
  V1 -> V1_true (true source kbgkeTTSau8), V2 -> V3 file, V3 -> V2 file, V4..V10 -> same name.
- Windows: length W = 16 frames, stride S = 4 frames, start frames 0, 4, 8, ... while start+16 <= n_frames.
  A window is identified by (video, start); it covers frames start .. start+15 inclusive.
- Labelled punch frames: every annotation row kept by punch_data.load_dataset() (valid label,
  0 < end-start < 25), frames start..end inclusive (video3d/common.annotations(v) gives the rows).
- Window label helpers (for evaluation only):
    overlap = fraction of the window's 16 frames inside any labelled punch.
    "labelled-punch window": contains >= 50% of some labelled punch's frames.
    "gap window": overlap == 0 and at least 8 frames away from any labelled punch.
- Output format for a detector: one file per video, `detect/scores/<detector>/V{v}.npz` with
  arrays start (int32), score (float32, P(punch) or a monotone punch score), plus optional extras.
