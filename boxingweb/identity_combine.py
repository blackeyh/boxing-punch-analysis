"""Do recognition (ready-made ResNet-50 features) and motion continuity make DIFFERENT mistakes at a separation?
If so: trust them when they agree, and flag (or ask) only when they disagree.   BoxingWeb data_test.
    python boxingweb/identity_combine.py
"""
import glob, os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import mem_guard  # noqa: F401
import numpy as np
import identity_episodes as ie, identity_reid as ir

def main():
    from multiprocessing import Pool
    ids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "external", "pose_rtm", "W_test*.npz")) if os.path.basename(p).count(".") == 1)
    with Pool(8) as pool:
        rds = pool.map(ie.load_round, ids); crops = pool.map(ir.crops_for_round, rds)
    emb = ir.make_models()["ResNet-50"]; mins = sum(rd["sec"] for rd in rds) / 60
    for grey in (False, True):
        rows = []
        for rd, c in zip(rds, crops):
            e = ir.seg_embeddings(c, emb, grey)
            for k, row in enumerate(rd["rows"]):
                q = [(k, 0), (k, 1), (k + 1, 0), (k + 1, 1)]
                if row["cut"] or not all(x in e for x in q):
                    continue
                m = (e[q[0]] @ e[q[2]] + e[q[1]] @ e[q[3]]) - (e[q[0]] @ e[q[3]] + e[q[1]] @ e[q[2]])
                rows.append((int(m > 0), int(row["x"][0] >= 0), row["y"], abs(m), row["gap"] / 30))
        r = np.array(rows); reid, cont, y, marg, gap = r.T; agree = reid == cont
        print(f"\n{'GREYSCALE' if grey else 'COLOUR'}: {len(y)} separations | recognition right {np.mean(reid == y):.1%} | continuity right {np.mean(cont == y):.1%}")
        print(f"  they agree in {agree.mean():.1%} of separations; when they agree both are right {np.mean(reid[agree] == y[agree]):.1%} ({np.sum(reid[agree] != y[agree]) / mins:.2f} silent errors/min)")
        d = ~agree; print(f"  they disagree in {d.mean():.1%} ({d.sum() / mins:.2f}/min): recognition is the right one in {np.mean(reid[d] == y[d]):.0%}")
        short = gap < 1.0; rule = np.where(agree, cont, np.where(short, cont, reid))
        print(f"  rule 'agree -> take it; disagree -> continuity if the overlap was < 1 s, else recognition': right {np.mean(rule == y):.1%} ({np.sum(rule != y) / mins:.2f} errors/min)")

if __name__ == "__main__":
    main()
