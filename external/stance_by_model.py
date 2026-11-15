"""Stance by asking the punch model: run it on the skeleton as-is AND mirrored; the reading in which it
finds more (and clearer) punches is the boxer's real stance. Rationale: the model only knows orthodox
punches, so a southpaw read as-is (or an orthodox read mirrored) looks like odd, unclear movement.

Check on our own videos, leave-one-video-out (the model judging video v never saw v):
    /opt/anaconda3/bin/python external/stance_by_model.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mem_guard  # noqa: F401,E402
import numpy as np  # noqa: E402

from baselines import MIRROR_IDX  # noqa: E402
from deep import TorchSeqClassifier  # noqa: E402
from detect.train7 import features  # noqa: E402
from progress import progress  # noqa: E402


def mirror(fr):
    xy, world, vis = fr["xy"][:, MIRROR_IDX].copy(), fr["world"][:, MIRROR_IDX].copy(), fr["vis"][:, MIRROR_IDX]
    xy[..., 0] = np.where(fr["present"][:, None], 1 - xy[..., 0], 0); world[..., 0] = -world[..., 0]
    return dict(xy=xy, world=world, vis=vis, present=fr["present"])


def evidence(model, fr, stream):
    """Mean over decisions of P(punch) and number of counted punches."""
    t, probs, sm, ev = stream(model, fr)
    return dict(p_punch=float(np.mean(1 - probs[:, 6])), n_events=len(ev),
                type_conf=float(np.mean([e["type_share"] for e in ev])) if ev else 0.0)


def decide(as_is, mirrored):
    """+1 = keep as-is (orthodox), -1 = mirror (southpaw); margin = relative difference in counted punches."""
    a, b = as_is["n_events"], mirrored["n_events"]
    return (1 if a >= b else -1), (a - b) / max(a + b, 1)


if __name__ == "__main__":
    from external.run_test import stream

    class Wrap:                                     # TorchSeqClassifier -> CLASSES + [No Punch] order
        def __init__(self, m):
            from punch_data import CLASSES
            self.m, self.order = m, [list(m.classes_).index(c) for c in CLASSES + ["No Punch"]]

        def predict_proba(self, X):
            return self.m.predict_proba(X)[:, self.order]

    d = np.load(ROOT / "detect" / "dataset7" / "clips.npz"); F = features(d)
    progress.start("stance-by-model LOVO", total=10)
    rows = []
    for v in range(1, 11):
        tr = d["video"] != v
        m = Wrap(TorchSeqClassifier("cnn", epochs=60, seed=0).fit(F[tr], d["label"][tr]))
        fr = {k: np.asarray(x, np.float32) if k != "present" else x for k, x in np.load(ROOT / "detect" / "frames" / f"V{v}.npz").items()}
        fr = dict(xy=fr["xy"], world=fr["world"], vis=fr["vis"], present=fr["present"])
        n = len(fr["present"]); res = []
        for a in range(0, n - 300, int(60 * 25)):                       # 60-s stretches
            seg = {k: x[a:a + 1500] for k, x in fr.items()}
            if seg["present"].mean() < 0.5:
                continue
            e1, e2 = evidence(m, seg, stream), evidence(m, mirror(seg), stream)
            res.append((decide(e1, e2), e1, e2))
        # frames are canonical orthodox (V4 already mirrored), so the truth is always "as-is"
        right = np.mean([r[0][0] == 1 for r in res])
        tot1 = sum(r[1]["n_events"] for r in res); tot2 = sum(r[2]["n_events"] for r in res)
        rows.append(dict(video=v, stretches=len(res), right=float(right), punches_as_is=tot1, punches_mirrored=tot2))
        print(f"V{v:<2} 60-s stretches judged right {right:.0%} of {len(res)} | punches counted: correct reading {tot1}, mirrored {tot2}", flush=True)
        progress.tick(1)
    progress.finish()
    (ROOT / "runs" / "stance_by_model.json").write_text(json.dumps(rows, indent=1))
