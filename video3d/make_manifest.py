"""Write video3d/manifest.json: everything needed to re-download and verify the source videos.

    python video3d/make_manifest.py     (needs ffprobe; reads video3d/raw/videos/)
"""
import hashlib
import json
import re
import subprocess
import zipfile

from common import HERE, RAW, ROOT, VIDEO_OF


def links():
    xml = zipfile.ZipFile(ROOT / "RGB_videos" / "Meta_data.ods").read("content.xml").decode()
    return {f"V{a}": b.replace("&amp;", "&") for a, b in re.findall(r"V(\d+)</text:p>.*?(https://[^<]+)", xml)}


def main():
    model = RAW / "models" / "pose_landmarker_heavy.task"
    man = {"download": {"tool": "yt-dlp 2026.10.26 (HPC, conda env punch3d)", "format": "136 (720p H.264 MP4, video only)",
                        "command": "video3d/download.sh", "note": "full videos; &t= offsets in Meta_data.ods ignored"},
           "pose_model": {"file": "pose_landmarker_heavy.task",
                          "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
                          "sha256": hashlib.sha256(model.read_bytes()).hexdigest()},
           "videos": {}}
    items = sorted(links().items(), key=lambda kv: int(kv[0][1:]))
    items.append(("V1_true", "https://www.youtube.com/watch?v=kbgkeTTSau8  (NOT in Meta_data.ods; found by find_source.py)"))
    man["dataset_to_video_file"] = {f"V{d}": f + ".mp4" for d, f in VIDEO_OF.items()}
    man["metadata_errors"] = ["Meta_data.ods links for V2 and V3 are swapped",
                              "Meta_data.ods link for V1 (3ov6zVAXHJI) is not the source of dataset V1; true source kbgkeTTSau8"]
    for v, url in items:
        p = RAW / "videos" / f"{v}.mp4"
        info = json.loads((RAW / "videos" / f"{v}.info.json").read_text())
        pr = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
             "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_read_packets,duration,sample_aspect_ratio",
             "-of", "json", str(p)], capture_output=True, text=True).stdout)["streams"][0]
        num, den = map(int, pr["avg_frame_rate"].split("/"))
        man["videos"][v] = dict(
            url_in_metadata=url, youtube_id=info["id"], title=info.get("title"), channel=info.get("channel"),
            upload_date=info.get("upload_date"), format_id=info.get("format_id"), codec=pr["codec_name"],
            width=pr["width"], height=pr["height"], aspect=f"{pr['width']}:{pr['height']}",
            sample_aspect_ratio=pr.get("sample_aspect_ratio"), fps=round(num / den, 6), fps_exact=pr["avg_frame_rate"],
            frame_count=int(pr["nb_read_packets"]), duration_s=float(pr["duration"]),
            bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        print(v, man["videos"][v]["fps_exact"], man["videos"][v]["frame_count"], flush=True)
    (HERE / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
