"""Identity by recognition, round 2: models BUILT to tell people apart, and an A-vs-B classifier fitted in each video.

Models (all off the shelf, no training by us):
  ResNet-50 (ImageNet)        : generic object features (reference from identity_reid.py)
  OSNet x1.0 (MSMT17)         : person re-identification, trained on 4,101 people across 15 cameras
  OSNet-AIN x1.0 (MSMT17)     : same, with instance normalisation for unseen domains
  (weights: huggingface.co/kaiyangzhou/osnet, MIT; model code: third_party/osnet/ from KaiyangZhou/deep-person-reid)
Ways of using them, per round (the first K stretches are the "pictures we were shown", truth-labelled = a clean start):
  gallery     : mean embedding of each boxer from those K stretches; later stretches go to the nearer one
  classifier  : logistic regression A-vs-B on the crops of those K stretches (learns what differs between THESE two)
  classifier+ : the same, re-fitted as the round goes on with its own answers (self-training)
Score: share of apart frames after the first K stretches with the right boxer, never corrected, camera cuts included.
Each in COLOUR and GREYSCALE.   BoxingWeb data_test.
    python boxingweb/identity_reid2.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "third_party", "osnet"))
import mem_guard  # noqa: F401
import cv2, numpy as np, torch
import identity_episodes as ie
N_CROPS, K = 6, 5
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def crops_for_round(rd):
    """-> [(stretch, track, square 224 crop, tight 256x128 crop)] RGB."""
    R = np.load(os.path.join(ROOT, "external", "pose_rtm", rd["id"] + ".npz")); W, H = int(R["width"]), int(R["height"])
    want = {}
    for si, s in enumerate(rd["segs"]):
        for j in np.unique(np.linspace(0, len(s["frames"]) - 1, N_CROPS).astype(int)):
            want.setdefault(int(s["frames"][j]), []).extend((si, t, int(s["tr"][t][j])) for t in (0, 1))
    cap = cv2.VideoCapture(os.path.join(ROOT, "external", "raw", rd["id"] + ".mp4")); out = []; pos = -1
    for i in sorted(want):
        f = int(R["frame"][i])
        if f != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read(); pos = f
        if not ok:
            continue
        for si, t, p in want[i]:
            x1, y1, x2, y2 = R["box"][i, p]; w, h = x2 - x1, y2 - y1
            x1, y1, x2, y2 = int(max(x1 - .05 * w, 0)), int(max(y1 - .05 * h, 0)), int(min(x2 + .05 * w, W)), int(min(y2 + .05 * h, H))
            c = cv2.cvtColor(im[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            if c.size == 0:
                continue
            s = max(c.shape[:2]); sq = np.zeros((s, s, 3), np.uint8); oy, ox = (s - c.shape[0]) // 2, (s - c.shape[1]) // 2
            sq[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
            out.append((si, t, cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA), cv2.resize(c, (128, 256), interpolation=cv2.INTER_AREA)))
    return out

def make_models():
    import torchvision
    from huggingface_hub import hf_hub_download
    import osnet, osnet_ain
    mean, std = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    def load_os(ctor, fname):
        m = ctor(num_classes=1, pretrained=False, loss="softmax")
        sd = torch.load(hf_hub_download("kaiyangzhou/osnet", fname), map_location="cpu", weights_only=True); sd = sd.get("state_dict", sd)
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}; own = m.state_dict()
        sd = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
        missing = [k for k in own if k not in sd and not k.startswith("classifier")]
        assert not missing, missing[:5]
        own.update(sd); m.load_state_dict(own); return m.to(DEV).eval()
    rn = torchvision.models.resnet50(weights="IMAGENET1K_V1"); rn.fc = torch.nn.Identity(); rn = rn.to(DEV).eval()
    os1 = load_os(osnet.osnet_x1_0, "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
    os2 = load_os(osnet_ain.osnet_ain_x1_0, "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth")
    def run(net, which):
        @torch.no_grad()
        def emb(crops, grey):
            o = []
            for a in range(0, len(crops), 64):
                x = np.stack([c[which] for c in crops[a:a + 64]])
                if grey:
                    x = np.repeat((x @ np.array([0.299, 0.587, 0.114]))[..., None], 3, -1)
                x = torch.from_numpy(x.astype(np.float32)).to(DEV).permute(0, 3, 1, 2) / 255
                o.append(torch.nn.functional.normalize(net((x - mean) / std), dim=1).cpu().numpy())
            return np.concatenate(o)
        return emb
    return {"ResNet-50 (generic)": run(rn, 2), "OSNet (re-identification)": run(os1, 3), "OSNet-AIN (re-identification)": run(os2, 3)}

def score_round(rd, crops, E):
    from sklearn.linear_model import LogisticRegression
    segs = rd["segs"]; per = {}
    for (si, t, *_), e in zip(crops, E):
        per.setdefault((si, t), []).append(e)
    per = {k: np.array(v) for k, v in per.items()}
    if len(segs) <= K + 2 or any((s, t) not in per for s in range(len(segs)) for t in (0, 1)):
        return None
    lab = lambda si, t: int((segs[si]["r0"] == 1) == (t == 0))          # 1 = red boxer, by truth
    Xk = np.vstack([per[(s, t)] for s in range(K) for t in (0, 1)]); yk = np.concatenate([[lab(s, t)] * len(per[(s, t)]) for s in range(K) for t in (0, 1)])
    g = [Xk[yk == 1].mean(0), Xk[yk == 0].mean(0)]
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(Xk, yk)
    Xs, ys = [Xk], [yk]; clf2 = clf
    out = np.zeros(5)                                                   # frames, gallery, classifier, classifier+, separations right (gallery-free, adjacent)
    for si in range(K, len(segs)):
        n = len(segs[si]["frames"]); a, b = per[(si, 0)], per[(si, 1)]; truth = lab(si, 0)          # is track 0 the red boxer?
        gal = (a.mean(0) @ g[0] + b.mean(0) @ g[1]) > (a.mean(0) @ g[1] + b.mean(0) @ g[0])
        d = lambda c: c.decision_function(a).mean() - c.decision_function(b).mean()
        c1 = d(clf) > 0; c2 = d(clf2) > 0
        out += [n, n * (gal == truth), n * (c1 == truth), n * (c2 == truth), 0]
        Xs += [a, b]; ys += [np.full(len(a), int(c2)), np.full(len(b), int(not c2))]
        if (si - K) % 5 == 4:
            clf2 = LogisticRegression(C=1.0, max_iter=1000).fit(np.vstack(Xs), np.concatenate(ys))
    return out

def main():
    from multiprocessing import Pool
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_test*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as pool:
        rds = pool.map(ie.load_round, ids); crops = pool.map(crops_for_round, rds)
    print(f"{len(rds)} test rounds, {sum(len(c) for c in crops)} crops; shown: the first {K} stretches of each round ({K * 2 * N_CROPS} crops), then never corrected", flush=True)
    res = {}
    for name, emb in make_models().items():
        for grey in (False, True):
            t = np.zeros(5); used = 0
            for rd, c in zip(rds, crops):
                s = score_round(rd, c, emb(c, grey))
                if s is not None:
                    t += s; used += 1
            tag = f"{name}, {'greyscale' if grey else 'colour'}"
            print(f"  {tag:42s} right boxer in apart frames: gallery {t[1] / t[0]:.1%} | A-vs-B classifier {t[2] / t[0]:.1%} | classifier, self-updating {t[3] / t[0]:.1%}   ({used} rounds)", flush=True)
            res[tag] = t.tolist()
    json.dump(res, open(os.path.join(ROOT, "runs", "identity_reid2.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
