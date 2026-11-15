"""Register the Olympic bouts with the external-footage pipeline (external/manifest.json + symlinks in
external/raw/), so extract.py / track.py / run_test.py work on them unchanged. Ids: O_<camera>_<clip>.
    python -I olympic/register.py "<dataset root>"
"""
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__)); ext = os.path.join(os.path.dirname(here), "external")
root = sys.argv[1]
vids = json.load(open(os.path.join(here, "videos.json")))
man_path = os.path.join(ext, "manifest.json"); man = json.load(open(man_path))
for v, i in vids.items():
    cid = "O_" + v
    link = os.path.join(ext, "raw", cid + ".mp4")
    if not os.path.lexists(link):
        os.symlink(os.path.join(root, i["task"], "data", i["file"]), link)
    man[cid] = dict(id=cid, category="O", url="kaggle: Olympic Boxing Punch Classification Video Dataset (Stefanski, Kozak, Jach 2024)",
                    notes="real youth bout, corner camera, referee-labelled punches (olympic/events.json)",
                    width=i["width"], height=i["height"], fps=i["fps"], frames_header=i["frames"], duration_s=i["seconds"],
                    segment_frames=[0, i["frames"]], spec_issues=[], status="ok")
json.dump(man, open(man_path, "w"), indent=1)
print(len(vids), "bouts registered")
