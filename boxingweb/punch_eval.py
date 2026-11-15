"""Re-score saved punch TCNs on BoxingWeb data_test under two matching rules, and against the paper's numbers.

BoxMind (arXiv 2601.11492, B.2): precision 0.806, recall 0.763, F1 0.783 "on the BoxingWeb test set", with a temporal
IoU threshold of 0.5 mentioned (the exact matching rule is not stated). Their detector: 2D+3D poses of both boxers
from 4D-Humans, trained on 40 BoxingWeb rounds + 30 private BoxingStudio rounds. Attributes (Table S3, F1): technique
(straight/hook/uppercut) 0.692 pose-only TCN, 0.725 best (video+pose).
Ours: RTMPose 2D only, 35 BoxingWeb rounds for training (5 for validation).
  loose : any overlap in time (label grown by 3 frames)          strict: temporal IoU >= 0.5
    python boxingweb/punch_eval.py NAME [NAME ...]
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mem_guard  # noqa: F401
import numpy as np, torch
import punch_tcn as pt

def score(Ps, data, thr, strict):
    s = dict(lab=0, cnt=0, hit=0); tt, tp = [], []; hand = 0
    for P, d in zip(Ps, data):
        for b in (0, 1):
            ev = pt.events(P[b], d["M"][b], thr); lab = pt.labels_of(d["Y"][b]); used = set()
            for a, e, c in lab:
                best, bo = None, 0
                for k, (pa, pe, pc) in enumerate(ev):
                    if k in used:
                        continue
                    if strict:
                        inter = min(e, pe) - max(a, pa) + 1; o = inter / ((e - a + 1) + (pe - pa + 1) - inter) if inter > 0 else 0; ok = o >= 0.5
                    else:
                        o = min(e + pt.TOL, pe) - max(a - pt.TOL, pa) + 1; ok = o > 0
                    if ok and o > bo:
                        best, bo = k, o
                if best is not None:
                    used.add(best); s["hit"] += 1; pc = ev[best][2]; tt.append((c - 1) // 2); tp.append((pc - 1) // 2); hand += (pc % 2) == (c % 2)
            s["lab"] += len(lab); s["cnt"] += len(ev)
    r, p = s["hit"] / s["lab"], s["hit"] / max(s["cnt"], 1); tt, tp = np.array(tt), np.array(tp)
    f1s = []
    for k in range(3):
        tpk = ((tt == k) & (tp == k)).sum(); pr = tpk / max((tp == k).sum(), 1); rc = tpk / max((tt == k).sum(), 1); f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
    return dict(p=p, r=r, f1=2 * p * r / max(p + r, 1e-9), tech_acc=float((tt == tp).mean()), tech_f1=float(np.mean(f1s)), hand=hand / max(s["hit"], 1))

def main():
    ids = lambda pre: sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "punch_ds", pre + "*.npz")))
    te = pt.load(ids("W_test")); va = pt.load(ids("W_train")[-pt.VAL_N:]); out = {}
    for name in sys.argv[1:]:
        noopp = "noopp" in name
        def X_of(d):
            X = d["X"][:2].copy()
            if noopp:
                X[..., pt.OPP] = 0; X[..., -1] = 0
            return X
        rows = {"loose": [], "strict": []}; Pte_all, Pva_all = [], []
        for f in sorted(glob.glob(os.path.join(ROOT, "runs", name, "model_seed*.pt"))):
            m = pt.TCN(te[0]["X"].shape[-1]).to(pt.DEV); m.load_state_dict(torch.load(f, map_location=pt.DEV, weights_only=True))
            Pte = [pt.predict(m, X_of(d)) for d in te]; Pva = [pt.predict(m, X_of(d)) for d in va]; Pte_all.append(Pte); Pva_all.append(Pva)
            for mode in rows:
                thr = max(np.arange(0.2, 0.81, 0.05), key=lambda t: score(Pva, va, t, mode == "strict")["f1"]); rows[mode].append(score(Pte, te, thr, mode == "strict"))
        ens_te = [np.mean([P[i] for P in Pte_all], 0) for i in range(len(te))]; ens_va = [np.mean([P[i] for P in Pva_all], 0) for i in range(len(va))]
        print(f"\n{name} ({len(Pte_all)} seeds)")
        for mode in rows:
            g = lambda k: (np.mean([r[k] for r in rows[mode]]), np.std([r[k] for r in rows[mode]]))
            thr = max(np.arange(0.2, 0.81, 0.05), key=lambda t: score(ens_va, va, t, mode == "strict")["f1"]); e = score(ens_te, te, thr, mode == "strict")
            print(f"  {mode:6s} matching: precision {g('p')[0]:.3f}, recall {g('r')[0]:.3f}, F1 {g('f1')[0]:.3f} +- {g('f1')[1]:.3f} | technique (3 classes) accuracy {g('tech_acc')[0]:.3f}, macro-F1 {g('tech_f1')[0]:.3f} | hand {g('hand')[0]:.3f}"
                  f"   || 3-seed ensemble: P {e['p']:.3f} R {e['r']:.3f} F1 {e['f1']:.3f}, technique F1 {e['tech_f1']:.3f}")
            out[f"{name}/{mode}"] = dict(mean={k: float(g(k)[0]) for k in rows[mode][0]}, f1_std=float(g("f1")[1]), ensemble=e)
    json.dump(out, open(os.path.join(ROOT, "runs", "punch_eval.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
