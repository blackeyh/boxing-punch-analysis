"""Turn extracted window features into detector scores + validation (runs locally, CPU, minutes).

    python videodet/score.py            # reads videodet/cache/feats/V*.npz

Detectors written to detect/scores/<name>/V{v}.npz (keys start, score; probes also type, hand,
p_type):
  videomae_zs     zero-shot: P(punching bag) + P(punching person (boxing)) from VideoMAE-B K400
  xclip_zs        zero-shot: X-CLIP softmax mass on the punch prompts (vs the not-punch prompts)
  videomae_probe  LEAVE-ONE-VIDEO-OUT logistic probe on frozen features (the main detector)

Probe training (per held-out video k, trained on the other 9 videos ONLY):
  positives = labelled-punch windows (labelled with their punch type),
  negatives = gap windows ("none"), ambiguous windows (partial overlap / near a punch) unused.
  7-way multinomial logistic regression; score = 1 - P(none); type = argmax over the 6 punches.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import windows as Wd  # noqa: E402
from progress import progress  # noqa: E402

CACHE = HERE / "cache" / "feats"
SCORES = HERE.parent / "detect" / "scores"
CLASSES = ["none", "Jab", "Cross", "Lead Hook", "Rear Hook", "Lead Uppercut", "Rear Uppercut"]
VIDS = list(range(1, 11))
K400_PUNCH = [258, 259]          # "punching bag", "punching person (boxing)"


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def load():
    D = {}
    for v in VIDS:
        f = np.load(CACHE / f"V{v}.npz")
        lab = Wd.window_labels(v)
        assert np.array_equal(f["start"], lab["start"]), f"V{v}: window starts differ from WINDOWS.md"
        D[v] = dict(f=dict(f), **lab)
    return D


def feature(d, kind):
    f = d["f"]
    parts = {"base": [f["base_feat"]], "large": [f.get("large_feat")], "xclip": [f["xclip_emb"]],
             "base+xclip": [f["base_feat"], f["xclip_emb"]],
             "large+xclip": [f.get("large_feat"), f["xclip_emb"]],
             "base+large": [f["base_feat"], f.get("large_feat")]}[kind]
    if any(p is None for p in parts):
        return None
    return np.concatenate([p.astype(np.float32) for p in parts], 1)


def fit_softmax(X, y, C, n_cls=7, iters=300):
    """Multinomial logistic regression, same objective as sklearn LogisticRegression(C=C,
    class_weight='balanced'): sum_i w_i CE_i + ||W||^2 / (2C) (bias unpenalised), minimised with
    full-batch L-BFGS in torch (CPU, float64). sklearn's lbfgs needed ~1 min/fold here."""
    import torch
    Xt = torch.from_numpy(X.astype(np.float64))
    yt = torch.from_numpy(y.astype(np.int64))
    cnt = np.bincount(y, minlength=n_cls).astype(np.float64)
    cw = torch.from_numpy(np.where(cnt > 0, len(y) / (n_cls * np.maximum(cnt, 1)), 0.0))
    sw = cw[yt]
    W = torch.zeros(X.shape[1], n_cls, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_cls, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, tolerance_grad=1e-6, tolerance_change=1e-10,
                            history_size=20, line_search_fn="strong_wolfe")
    N = sw.sum()

    def closure():
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(Xt @ W + b, yt, reduction="none")
        loss = (sw * ce).sum() / N + (W ** 2).sum() / (2 * C * N)
        loss.backward()
        return loss
    opt.step(closure)
    return W.detach().numpy(), b.detach().numpy()


def predict_softmax(X, W, b):
    return softmax(X @ W + b)


def lovo_probe(D, kind, C=0.1, pervideo=False, head="7way"):
    """LOVO probe. pervideo: standardise each video's features with that video's OWN mean/std
    (unsupervised, needs no labels, so it is also allowed for the held-out video); removes
    per-video appearance offsets (scene, clothing, camera). head: 7way (none + 6 types) or binary."""
    X = {v: feature(D[v], kind) for v in VIDS}
    if X[1] is None:
        return None
    if pervideo:
        X = {v: (x - x.mean(0)) / (x.std(0) + 1e-6) for v, x in X.items()}
    y = {}
    for v in VIDS:
        t = np.full(len(X[v]), -1)
        t[D[v]["is_gap"]] = 0
        for c in range(1, 7):
            t[D[v]["is_punch"] & (D[v]["label"] == CLASSES[c])] = c if head == "7way" else 1
        y[v] = t
    out = {}
    for k in VIDS:
        tr = [v for v in VIDS if v != k]
        Xtr = np.concatenate([X[v][y[v] >= 0] for v in tr])
        ytr = np.concatenate([y[v][y[v] >= 0] for v in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        W, b = fit_softmax((Xtr - mu) / sd, ytr, C, n_cls=7 if head == "7way" else 2)
        out[k] = predict_softmax((X[k] - mu) / sd, W, b)
        progress.tick(1, note=f"probe {kind} held-out V{k}")
    return out


def punch_recall(D, v, s, t):
    """Fraction of labelled PUNCHES (not windows) with at least one window that contains >= 50%
    of the punch and scores >= t."""
    a = Wd.punches(v)
    st = D[v]["start"]
    hit = []
    for ps, pe in zip(a["start"], a["end"]):
        L = pe - ps + 1
        cov = (np.minimum(pe, st + Wd.W - 1) - np.maximum(ps, st) + 1).clip(0) / L
        m = cov >= 0.5
        hit.append(bool(m.any() and s[m].max() >= t))
    return float(np.mean(hit))


def validate(D, scores):
    """Per video: AUC punch-vs-gap (all gaps, and only gaps inside the annotated span, i.e. between
    the first and last labelled punch), window recall at fixed thresholds, gap windows above the
    per-video 95%-recall threshold, and with ONE global threshold giving 95% pooled window recall
    (plus punch-level recall at that global threshold)."""
    pos_all = np.concatenate([scores[v][D[v]["is_punch"]] for v in VIDS])
    t_glob = float(np.quantile(pos_all, 0.05))
    rows = []
    auc = lambda a, b: float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))
    for v in VIDS:
        s, p, g = scores[v], D[v]["is_punch"], D[v]["is_gap"]
        a = Wd.punches(v)
        gin = g & (D[v]["start"] >= a["start"].min()) & (D[v]["start"] + Wd.W - 1 <= a["end"].max())
        t95 = np.quantile(s[p], 0.05)
        rows.append(dict(video=v, n_punch=int(p.sum()), n_gap=int(g.sum()), n_gap_in=int(gin.sum()),
                         auc=auc(s[p], s[g]), auc_in=auc(s[p], s[gin]),
                         rec_03=float((s[p] >= 0.3).mean()), rec_05=float((s[p] >= 0.5).mean()), rec_07=float((s[p] >= 0.7).mean()),
                         gap_over_t95=int((s[g] >= t95).sum()), gap_frac_t95=float((s[g] >= t95).mean()),
                         gap_over_tglob=int((s[g] >= t_glob).sum()), gap_frac_tglob=float((s[g] >= t_glob).mean()),
                         rec_tglob=float((s[p] >= t_glob).mean()), punch_rec_tglob=punch_recall(D, v, s, t_glob)))
    pooled = dict(t_glob=t_glob, mean_auc=float(np.mean([r["auc"] for r in rows])),
                  mean_auc_in=float(np.mean([r["auc_in"] for r in rows])),
                  pooled_auc=auc(pos_all, np.concatenate([scores[v][D[v]["is_gap"]] for v in VIDS])))
    return rows, pooled


def type_accuracy(D, P):
    """Balanced accuracy of the 6-way type guess on labelled-punch windows (LOVO)."""
    yt, yp = [], []
    for v in VIDS:
        m = D[v]["is_punch"]
        yt += list(D[v]["label"][m]); yp += [CLASSES[1 + i] for i in P[v][m, 1:].argmax(1)]
    yt, yp = np.array(yt), np.array(yp)
    return float(np.mean([(yp[yt == c] == c).mean() for c in CLASSES[1:]]))


def save(name, D, scores, P=None, extra=None):
    """P: optional (n, 7) LOVO class probabilities (col 0 = none) used ONLY for the type/hand extras."""
    d = SCORES / name
    d.mkdir(parents=True, exist_ok=True)
    for v in VIDS:
        arr = dict(start=D[v]["start"].astype(np.int32), score=scores[v].astype(np.float32))
        if P is not None:
            pt = P[v][:, 1:] / np.clip(P[v][:, 1:].sum(1, keepdims=True), 1e-9, None)
            arr["p_type"] = pt.astype(np.float32)                 # columns = CLASSES[1:]
            arr["type"] = np.array(CLASSES[1:])[pt.argmax(1)]
            arr["hand"] = np.array(["lead" if t in ("Jab", "Lead Hook", "Lead Uppercut") else "rear" for t in arr["type"]])
            arr["type_classes"] = np.array(CLASSES[1:])
        np.savez_compressed(d / f"V{v}.npz", **arr)
    if extra:
        (d / "meta.json").write_text(json.dumps(extra, indent=1))


def fmt(rows, pooled, title):
    s = [f"### {title}", f"mean per-video AUC {pooled['mean_auc']:.3f} (in-span gaps only {pooled['mean_auc_in']:.3f}), "
         f"pooled AUC {pooled['pooled_auc']:.3f}, global 95%-recall threshold {pooled['t_glob']:.3f}", "",
         "| video | punch win | gap win (in span) | AUC | AUC in-span | rec@0.3 | rec@0.5 | rec@0.7 | gap > t95(video) | gap > t95(global) | win rec@t95(global) | punch rec@t95(global) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s.append(f"| V{r['video']} | {r['n_punch']} | {r['n_gap']} ({r['n_gap_in']}) | {r['auc']:.3f} | {r['auc_in']:.3f} | {r['rec_03']:.2f} | "
                 f"{r['rec_05']:.2f} | {r['rec_07']:.2f} | {r['gap_over_t95']} ({r['gap_frac_t95']:.0%}) | "
                 f"{r['gap_over_tglob']} ({r['gap_frac_tglob']:.0%}) | {r['rec_tglob']:.2f} | {r['punch_rec_tglob']:.2f} |")
    return "\n".join(s) + "\n"


def main():
    D = load()
    report, summary = [], {}
    # zero-shot
    zs = {"videomae_zs": {v: softmax(D[v]["f"]["base_logits"])[:, K400_PUNCH].sum(1) for v in VIDS}}
    npos = int(D[1]["f"]["n_pos_prompts"])
    zs["xclip_zs"] = {v: softmax(D[v]["f"]["xclip_logits"])[:, :npos].sum(1) for v in VIDS}
    for name, sc in zs.items():
        rows, pooled = validate(D, sc)
        save(name, D, sc, extra=dict(kind="zero-shot", **pooled))
        report.append(fmt(rows, pooled, name)); summary[name] = dict(rows=rows, **pooled)
    # LOVO probes: the reference first probe (before the grid) and the chosen configuration
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="large"); ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--pervideo", action="store_true"); ap.add_argument("--head", default="7way")
    args = ap.parse_args()
    configs = {"reference": ("large", 0.1, False, "7way"), "chosen": (args.kind, args.C, args.pervideo, args.head)}
    progress.start("videodet LOVO probes", total=10 * len(configs) + 10)
    for role, (kind, C, pv, head) in configs.items():
        P = lovo_probe(D, kind, C, pv, head)
        sc = {v: 1 - P[v][:, 0] for v in VIDS}
        rows, pooled = validate(D, sc)
        cfg = dict(features=kind, C=C, pervideo=pv, head=head)
        title = f"probe ({role}): {cfg}"
        if head == "7way":
            pooled["type_bal_acc"] = type_accuracy(D, P)
            title += f"; 6-way type balanced acc {pooled['type_bal_acc']:.3f}"
        report.append(fmt(rows, pooled, title)); summary[f"probe_{role}"] = dict(rows=rows, **cfg, **pooled)
        print(report[-1], flush=True)
        if role == "chosen":
            Pt, type_cfg = P, cfg
            if head != "7way":       # type/hand extras from a separate LOVO 7-way probe (best 7-way grid setting)
                type_cfg = dict(features=kind, C=1e-3, pervideo=pv, head="7way")
                Pt = lovo_probe(D, kind, 1e-3, pv, "7way")
                pooled["type_bal_acc"] = type_accuracy(D, Pt)
                summary[f"probe_{role}"]["type_bal_acc"] = pooled["type_bal_acc"]
                report[-1] += f"\n6-way type guess (extra keys, from {type_cfg}): balanced accuracy on labelled-punch windows {pooled['type_bal_acc']:.3f} (chance 0.167)\n"
            save("videomae_probe", D, sc, Pt,
                 extra=dict(kind="LOVO logistic probe on frozen VideoMAE features", **cfg, type_extras_from=type_cfg, **pooled))
    progress.finish()
    (HERE / "validation.md").write_text("\n".join(report))
    (HERE / "validation.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(report[:2]))


if __name__ == "__main__":
    main()
