#!/bin/bash
# Download the 10 source videos (full length; the &t= offsets in Meta_data.ods are ignored)
# at 720p H.264 (YouTube format 136, video only). Each video is offered at one native fps
# only, which matches the annotation frame ranges (see REPORT.md).
# Output: video3d/raw/videos/V{i}.mp4 + V{i}.info.json   (git-ignored)
# Needs a recent yt-dlp (2026.06 got HTTP 403 from YouTube; 2026.10.26 works).
# Run on the HPC login node:  conda activate punch3d && bash video3d/download.sh
set -u
cd "$(dirname "$0")" && mkdir -p raw/videos && cd raw/videos
ids=(3ov6zVAXHJI JGLLJUgm-0c Cx6JfrRO4eA IVvOmoB88FM V59fcX1YC7E cQW-h8yGbRc sqU3DmLppcQ tSNybMUpMKk WuBPbI0x50o OfZFdwW322E)
yt-dlp --version
for i in "${!ids[@]}"; do
  v=$((i+1)); id=${ids[$i]}
  [ -s "V$v.mp4" ] && { echo "V$v exists"; continue; }
  yt-dlp --no-warnings --no-progress -f 136 --write-info-json -o "V$v.%(ext)s" \
    "https://www.youtube.com/watch?v=$id" || echo "V$v FAILED"
done
# The dataset's V1 skeleton does NOT come from the V1 link above (3ov6zVAXHJI). Its true source,
# found by video3d/find_source.py, is kbgkeTTSau8 (same channel/gym). Saved as V1_true.mp4.
[ -s V1_true.mp4 ] || yt-dlp --no-warnings --no-progress -f 136 --write-info-json -o "V1_true.%(ext)s" \
  "https://www.youtube.com/watch?v=kbgkeTTSau8" || echo "V1_true FAILED"
