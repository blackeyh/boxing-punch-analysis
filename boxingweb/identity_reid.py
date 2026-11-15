"""User's idea (2026-11-14): don't track identity, RECOGNISE it. A ready-made vision model, no training:
show it a few clear pictures of each fighter, then ask "which of the two is this?" whenever it matters.

Test on BoxingWeb data_test (truth known). Pictures = crops of each boxer in the "apart" stretches
(identity_episodes.segments; N_CROPS frames per boxer per stretch -> mean embedding).
  A. per separation (no camera cut): same way round or swapped?  -> compare with motion continuity (92.0%)
  B. the same at camera cuts, where continuity has nothing to go on
  C. whole round, NEVER corrected, cuts included: gallery = the first stretch; every later stretch is labelled by
     similarity to the gallery ("fixed": first-stretch pictures only; "updated": gallery grows with each stretch)
Each in COLOUR and in GREYSCALE (no hue: closer to fighters in similar kit).
Models: DINOv2-small (self-supervised, general purpose) and ResNet-50 (ImageNet), both off the shelf.
    python boxingweb/identity_reid.py
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mem_guard  # noqa: F401
import cv2, numpy as np, torch
import identity_episodes as ie
N_CROPS = 6
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def crops_for_round(rd):
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
            c = im[y1:y2, x1:x2]
            if c.size == 0:
                continue
            s = max(c.shape[:2]); sq = np.zeros((s, s, 3), np.uint8); oy, ox = (s - c.shape[0]) // 2, (s - c.shape[1]) // 2
            sq[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
            out.append((si, t, cv2.cvtColor(cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)))
    return out

def make_models():
    import torchvision
    from transformers import AutoModel
    mean, std = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    rn = torchvision.models.resnet50(weights="IMAGENET1K_V1"); rn.fc = torch.nn.Identity(); rn = rn.to(DEV).eval()
    def run(fn):
        @torch.no_grad()
        def emb(imgs):
            o = []
            for a in range(0, len(imgs), 64):
                x = torch.from_numpy(np.stack(imgs[a:a + 64])).to(DEV).permute(0, 3, 1, 2).float() / 255
                e = fn((x - mean) / std); o.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
            return np.concatenate(o)
        return emb
    return {"DINOv2-small": run(lambda x: dino(pixel_values=x).pooler_output), "ResNet-50": run(lambda x: rn(x))}

def seg_embeddings(crops, emb, grey):
    imgs = [np.repeat(cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)[..., None], 3, 2) if grey else c for *_, c in crops]
    E = emb(imgs); out = {}
    for (si, t, _), e in zip(crops, E):
        out.setdefault((si, t), []).append(e)
    return {k: (lambda m: m / np.linalg.norm(m))(np.mean(v, 0)) for k, v in out.items()}

def evaluate(rds, E):
    a = dict(n=0, ok=0, nc=0, okc=0, fr=0, fix=0, upd=0, cont_fr=0)
    for rd, e in zip(rds, E):
        segs = rd["segs"]
        if not segs or (0, 0) not in e or (0, 1) not in e:
            continue
        gal_fix = [e[(0, 0)], e[(0, 1)]]; gal = [[e[(0, 0)]], [e[(0, 1)]]]
        truth0 = segs[0]["r0"]; n0 = len(segs[0]["frames"]); a["fr"] += n0; a["fix"] += n0; a["upd"] += n0
        for k, row in enumerate(rd["rows"]):
            A, B = (k, 0), (k, 1); C, Dd = (k + 1, 0), (k + 1, 1)
            if not all(q in e for q in (A, B, C, Dd)):
                continue
            keep = e[A] @ e[C] + e[B] @ e[Dd] > e[A] @ e[Dd] + e[B] @ e[C]
            if row["cut"]:
                a["nc"] += 1; a["okc"] += int(keep) == row["y"]
            else:
                a["n"] += 1; a["ok"] += int(keep) == row["y"]
            s = segs[k + 1]; n = len(s["frames"]); a["fr"] += n
            same_fix = gal_fix[0] @ e[C] + gal_fix[1] @ e[Dd] > gal_fix[0] @ e[Dd] + gal_fix[1] @ e[C]      # track0 of this stretch = gallery A?
            a["fix"] += n * (same_fix == (s["r0"] == truth0))
            g = [np.mean(gal[0], 0), np.mean(gal[1], 0)]
            same_upd = g[0] @ e[C] + g[1] @ e[Dd] > g[0] @ e[Dd] + g[1] @ e[C]
            a["upd"] += n * (same_upd == (s["r0"] == truth0))
            gal[0].append(e[C] if same_upd else e[Dd]); gal[1].append(e[Dd] if same_upd else e[C])
    return a

def main():
    from multiprocessing import Pool
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_test*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as pool:
        rds = pool.map(ie.load_round, ids); crops = pool.map(crops_for_round, rds)
    print(f"{len(rds)} test rounds, {sum(len(r['segs']) for r in rds)} apart stretches, {sum(len(c) for c in crops)} crops; "
          f"separations: {sum(not r['cut'] for rd in rds for r in rd['rows'])} without a cut, {sum(r['cut'] for rd in rds for r in rd['rows'])} at camera cuts", flush=True)
    res = {}
    for name, emb in make_models().items():
        for grey in (False, True):
            a = evaluate(rds, [seg_embeddings(c, emb, grey) for c in crops]); tag = f"{name}, {'greyscale' if grey else 'colour'}"
            print(f"  {tag:26s} A separations right {a['ok'] / a['n']:.1%} | B at camera cuts {a['okc'] / max(a['nc'], 1):.1%} | "
                  f"C whole round never corrected: fixed gallery {a['fix'] / a['fr']:.1%}, updated gallery {a['upd'] / a['fr']:.1%} of apart frames", flush=True)
            res[tag] = {k: float(v) for k, v in a.items()}
    json.dump(res, open(os.path.join(ROOT, "runs", "identity_reid.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
