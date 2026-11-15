"""Parse the Kaggle 'Olympic Boxing Punch Classification Video Dataset' (Stefanski, Kozak, Jach, Entropy 2024)
CVAT track annotations into one flat event list: olympic/events.json + olympic/videos.json.

Usage: python -I olympic/parse.py "<dataset root>"
Each event: video, start/end frame (inclusive, 50 fps), hand (left/right = the PUNCHER's hand as labelled),
outcome (head / body / block / miss), landed (head or body), box per keyframe (x1,y1,x2,y2 in 1920x1080).
"""
import json, os, subprocess, sys, collections

LABELS = {
    "Głowa lewą ręką": ("left", "head"), "Głowa prawą ręką": ("right", "head"),
    "Korpus lewą ręką": ("left", "body"), "Korpus prawą ręką": ("right", "body"),
    "Blok lewą ręką": ("left", "block"), "Blok prawą ręką": ("right", "block"),
    "Chybienie lewą ręką": ("left", "miss"), "Chybienie prawą ręką": ("right", "miss"),
}

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", path],
                         capture_output=True, text=True).stdout
    s = json.loads(out)["streams"][0]
    n, d = s["r_frame_rate"].split("/")
    return dict(width=s["width"], height=s["height"], fps=float(n) / float(d),
                frames=int(s["nb_frames"]), seconds=float(s["duration"]))

def main(root):
    here = os.path.dirname(os.path.abspath(__file__))
    events, videos = [], {}
    for task in sorted(os.listdir(root)):
        ann = os.path.join(root, task, "annotations.json")
        if not os.path.isfile(ann):
            continue
        vid = task.replace("task_", "")
        files = [f for f in os.listdir(os.path.join(root, task, "data")) if f.lower().endswith(".mp4")]
        assert len(files) == 1, (task, files)
        info = probe(os.path.join(root, task, "data", files[0]))
        info.update(task=task, file=files[0], camera=vid.split("_")[0])
        jobs = json.load(open(ann, encoding="utf-8"))
        n = 0
        for job in jobs:
            assert not job["shapes"] and not job["tags"], task
            for tr in job["tracks"]:
                hand, outcome = LABELS[tr["label"]]
                vis = [s for s in tr["shapes"] if not s["outside"]]
                out = [s["frame"] for s in tr["shapes"] if s["outside"]]
                if not vis:
                    continue
                start = min(s["frame"] for s in vis)
                # CVAT: the track stays visible until the first 'outside' keyframe after it
                later = [f for f in out if f > start]
                end = (min(later) - 1) if later else max(s["frame"] for s in vis)
                end = max(end, max(s["frame"] for s in vis if s["frame"] <= end))
                events.append(dict(video=vid, start=start, end=end, hand=hand, outcome=outcome,
                                   landed=outcome in ("head", "body"),
                                   closed=bool(later),
                                   boxes=[[s["frame"]] + [round(p, 1) for p in s["points"]] for s in vis]))
                n += 1
        info["events"] = n
        videos[vid] = info
    events.sort(key=lambda e: (e["video"], e["start"]))
    json.dump(events, open(os.path.join(here, "events.json"), "w"))
    json.dump(videos, open(os.path.join(here, "videos.json"), "w"), indent=1)
    return events, videos

if __name__ == "__main__":
    ev, vids = main(sys.argv[1])
    import statistics as st
    print(f"{len(vids)} videos, {sum(v['seconds'] for v in vids.values())/60:.1f} min, {len(ev)} events")
    print("outcome x hand:", dict(collections.Counter((e['outcome'], e['hand']) for e in ev)))
    L = [e['end'] - e['start'] + 1 for e in ev]
    print("event length frames: median", st.median(L), "p5/p95", sorted(L)[len(L)//20], sorted(L)[-len(L)//20], "max", max(L))
    print("unclosed tracks:", sum(not e['closed'] for e in ev))
    for v, i in vids.items():
        es = [e for e in ev if e['video'] == v]
        cov = sum(e['end'] - e['start'] + 1 for e in es) / i['frames']
        print(f"{v:18s} {i['width']}x{i['height']} {i['fps']:.0f}fps {i['seconds']:6.0f}s events {len(es):4d} "
              f"({60*len(es)/i['seconds']:5.1f}/min) labelled frames {100*cov:4.1f}% first {es[0]['start'] if es else '-'} last {es[-1]['end'] if es else '-'} / {i['frames']}")
