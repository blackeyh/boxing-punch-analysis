"""Cache the chain's per-boxer skeleton streams for the Olympic bouts (selector -> identity -> streams), so later
steps do not have to redo identity:  external/chain_streams/{id}.npz
  X (4, F, 157) features [boxer A, boxer B, A mirrored, B mirrored] (boxingweb/punch_data.feats), M (2, F) present,
  K (2, F, 17, 2) pixel keypoints (interpolated over short gaps), frame (F,) video frame numbers, fps.
    python olympic/chain_streams.py
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "boxingweb"))
import mem_guard  # noqa: F401
import numpy as np
import fullchain as fc, identity_reid2 as ir2, punch_data as pd, punch_fullchain as pf

def main():
    from multiprocessing import Pool
    out = os.path.join(ROOT, "external", "chain_streams"); os.makedirs(out, exist_ok=True)
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "O_*.npz")) if os.path.basename(p).count(".") == 1)
    ids = [c for c in ids if not os.path.exists(os.path.join(out, c + ".npz"))]
    if not ids:
        print("all cached"); return
    with Pool(6) as pool:
        prs = pool.map(fc.prepare, ids)
    emb = ir2.make_models()["OSNet-AIN (re-identification)"]
    for pr in prs:
        ident, R = pf.identities(pr, emb); kp, sc = R["kp"].astype(np.float64), R["score"]; W = float(R["width"])
        role = np.where(ident == 1, 1, np.where(ident == 0, 2, 0)).astype(np.int8); st = {r: pd.stream(kp, sc, role, r) for r in (1, 2)}; X = []
        for mir in (False, True):
            for r in (1, 2):
                K, C, M = st[r]; Ko, Co, Mo = st[3 - r]
                if mir:
                    K, C = pd.mirror(K, C, W); Ko, Co = pd.mirror(Ko, Co, W)
                X.append(pd.feats(K, C, M, Ko, Mo))
        np.savez_compressed(os.path.join(out, pr["id"] + ".npz"), X=np.stack(X), M=np.stack([st[1][2], st[2][2]]), K=np.stack([st[1][0], st[2][0]]).astype(np.float32),
                            frame=R["frame"], fps=float(R["fps"]))
        print(pr["id"], "streams saved", flush=True)

if __name__ == "__main__":
    main()
