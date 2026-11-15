"""Tracklets + selector features for every RTMPose dump, in parallel (cached as external/pose_rtm/{id}.feat.npz).
    python olympic/precompute_feats.py [PREFIX]      (default O_)
"""
import glob, os, sys, time, warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_boxers as sb

def run(cid):
    t = time.time(); D = sb.load(cid)
    return f"{cid} frames {len(D['frame'])} tracklets {D['tid'].max() + 1} in {time.time() - t:.0f}s"

if __name__ == "__main__":
    pre = sys.argv[1] if len(sys.argv) > 1 else "O_"
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(sb.ROOT / "external" / "pose_rtm" / f"{pre}*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as p:
        for r in p.imap_unordered(run, ids):
            print(r, flush=True)
    print("ALL DONE")
