"""Prepare EXTERNAL test footage (testing only, never training; see detect/TEST_FOOTAGE_SPEC.md).

    1. Fill external/clips.csv:  id,category,url,start,end,notes
         id        short name, e.g. A1_sparring_gym      category  A | B | C1 | C2
         url       a YouTube URL, OR an absolute path to a local video file
         start/end mm:ss (or hh:mm:ss) of the qualifying continuous segment
    2. /opt/anaconda3/bin/python external/ingest.py            # download + probe + manifest
    3. /opt/anaconda3/envs/mediapipe/bin/python external/extract.py   # skeletons, up to 3 people

Downloads the WHOLE video (yt-dlp, verified HTTPS; YouTube refuses section-only downloads with 403)
into external/raw/{id}.mp4 (git-ignored), records resolution, fps, frame count, sha256 and the
segment as original-video frame numbers in external/manifest.json (committed), and checks the
segment against the spec (>= 720p, >= 24 fps, >= 2 min). extract.py processes only the segment.
"""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
YTDLP = "/opt/anaconda3/bin/yt-dlp"


def probe(path):
    cap = cv2.VideoCapture(str(path))
    info = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS)), frames_header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    info["duration_s"] = round(info["frames_header"] / info["fps"], 1) if info["fps"] else None
    info["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return info


def secs(t):
    parts = [float(x) for x in t.strip().split(":")]
    return sum(x * 60 ** i for i, x in enumerate(reversed(parts)))


def check(info):
    issues = []
    if min(info["width"], info["height"]) < 720:
        issues.append(f"resolution {info['width']}x{info['height']} < 720p")
    if info["fps"] < 23.9:
        issues.append(f"fps {info['fps']:.2f} < 24")
    if info["duration_s"] is not None and info["duration_s"] < 120:
        issues.append(f"duration {info['duration_s']} s < 120 s")
    return issues


def main(csv_path=HERE / "clips.csv"):
    RAW.mkdir(exist_ok=True)
    man_path = HERE / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        sys.exit("external/clips.csv has no clips yet")
    for r in rows:
        cid = r["id"].strip(); out = RAW / f"{cid}.mp4"
        src = r["url"].strip()
        if not out.exists() and (src.startswith("/") or src.startswith("~")):
            # local file supplied by the user: copy it in (the original is left untouched)
            import shutil
            sp = Path(src).expanduser()
            if not sp.exists():
                print(f"[{cid}] FAILED: local file not found: {sp}"); manifest[cid] = dict(**r, status="missing_local_file"); continue
            print(f"[{cid}] copying local file {sp}", flush=True); shutil.copyfile(sp, out)
        if not out.exists():
            cmd = [YTDLP, "--no-playlist",
                   # VIDEO ONLY, single stream (no audio needed): 136 = 720p mp4, as used for our own videos.
                   # Merging separate audio/video streams goes through ffmpeg, which YouTube answers with 403.
                   "-f", "136/bv*[height<=1080][ext=mp4]", "--no-warnings", "--no-progress",
                   "-o", str(out), r["url"].strip()]
            print(f"[{cid}] downloading {r['url']} {r['start']}-{r['end']}", flush=True)
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0 or not out.exists():
                print(f"[{cid}] FAILED: {res.stderr.strip()[-300:]}"); manifest[cid] = dict(**r, status="download_failed"); continue
        info = probe(out)
        t0, t1 = secs(r["start"]), secs(r["end"])
        info["segment_s"] = [t0, t1]; info["segment_frames"] = [int(round(t0 * info["fps"])), int(round(t1 * info["fps"]))]
        info["duration_s"] = round(t1 - t0, 1)
        issues = check(info)
        manifest[cid] = dict(**{k: v.strip() for k, v in r.items()}, **info, spec_issues=issues, status="ok" if not issues else "check")
        print(f"[{cid}] {info['width']}x{info['height']} {info['fps']:.2f} fps {info['duration_s']} s  "
              f"{'OK' if not issues else 'ISSUES: ' + '; '.join(issues)}", flush=True)
    man_path.write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
