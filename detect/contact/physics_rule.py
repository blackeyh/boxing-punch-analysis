"""Training-free contact rule from the PUNCHER's arm only (works for bag, BOB or opponent).

Physics: a target stops the fist early and abruptly; an air punch extends fully and the fist slows
down gradually on its own. Per punch (MediaPipe 3D world coords, conf-filled, from detect/frames):
  * punching arm: from the punch type (lead = fighter's left after orthodox canonicalisation), or the
    faster wrist when the type is unknown;
  * wrist speed relative to the same-side shoulder, in m/s (fps-aware), lightly smoothed;
  * ext(t) = |wrist - shoulder| / (upper arm + forearm length, clip median);
  * peak = frame of max wrist speed inside the punch (+3 frames of context each side);
  * stop_ms = time for the speed to fall from 80% to 30% of the peak after the peak (short = abrupt);
  * ext_stop = extension at the moment the speed has fallen to 30% (where the fist stopped);
  * ext_max = max extension during the punch.
Rule (set from physics, NOT fitted to the bag/shadow split):
  contact    if stop_ms <= STOP_MS_CONTACT and (straight punch -> ext_stop <= EXT_SHORT; bent punch -> any)
  no_contact if stop_ms >= STOP_MS_AIR      or (straight punch and ext_max >= EXT_FULL)
  else unclear
    /opt/anaconda3/bin/python detect/contact/physics_rule.py            # measurements + rule on V1-V10
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from punch_data import CLASSES  # noqa: E402

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
LEAD = {"Jab", "Lead Hook", "Lead Uppercut"}
STRAIGHT = {"Jab", "Cross"}
STOP_MS_CONTACT, STOP_MS_AIR = 60.0, 130.0
EXT_SHORT, EXT_FULL = 0.85, 0.93
CTX = 3


def measure(world, fps, s, e, ptype=None):
    """world: (F,17,3) for the whole video; s..e inclusive punch frames. Returns a dict of measurements."""
    a, b = max(0, s - CTX), min(len(world) - 1, e + CTX + 4)
    W = world[a:b + 1].astype(np.float32)
    def side(left):
        sh, el, wr = (L_SH, L_EL, L_WR) if left else (R_SH, R_EL, R_WR)
        rel = W[:, wr] - W[:, sh]
        arm = np.median(np.linalg.norm(W[:, el] - W[:, sh], axis=1) + np.linalg.norm(W[:, wr] - W[:, el], axis=1))
        sp = np.r_[0, np.linalg.norm(np.diff(rel, axis=0), axis=1)] * fps
        sp = np.convolve(sp, np.ones(2) / 2, "same")
        return sp, np.linalg.norm(rel, axis=1) / max(arm, 1e-3)
    if ptype is not None:
        left = ptype in LEAD
    else:
        left = side(True)[0].max() >= side(False)[0].max()
    sp, ext = side(left)
    lo, hi = s - a, e - a                                    # punch range inside the window
    pk = lo + int(np.argmax(sp[lo:hi + 1])) if hi >= lo else int(np.argmax(sp))
    vpk = sp[pk]
    after = sp[pk:]
    t80 = next((i for i, x in enumerate(after) if x <= 0.8 * vpk), None)
    t30 = next((i for i, x in enumerate(after) if x <= 0.3 * vpk), None)
    stop_ms = (t30 - t80) * 1000 / fps if t80 is not None and t30 is not None else np.nan
    ext_stop = float(ext[pk + t30]) if t30 is not None else np.nan
    return dict(peak_speed=round(float(vpk), 3), stop_ms=round(float(stop_ms), 1), ext_stop=round(ext_stop, 3),
                ext_max=round(float(ext[lo:hi + 1].max()), 3), arm="lead" if left else "rear")


def rule(m, ptype):
    straight = ptype in STRAIGHT if ptype else False
    if np.isnan(m["stop_ms"]):
        return "unclear"
    if m["stop_ms"] >= STOP_MS_AIR or (straight and m["ext_max"] >= EXT_FULL):
        return "no_contact"
    if m["stop_ms"] <= STOP_MS_CONTACT and (not straight or m["ext_stop"] <= EXT_SHORT):
        return "contact"
    return "unclear"


def segments(label):
    out, f = [], 0
    while f < len(label):
        if label[f] >= 0:
            g = f
            while g + 1 < len(label) and label[g + 1] == label[f]:
                g += 1
            out.append((f, g, CLASSES[label[f]])); f = g + 1
        else:
            f += 1
    return out


def main():
    rows = []
    for v in range(1, 11):
        fr = np.load(ROOT / "detect" / "frames" / f"V{v}.npz")
        world, fps = fr["world"], float(fr["fps"])
        for s, e, t in segments(fr["label"]):
            m = measure(world, fps, s, e, t)
            rows.append(dict(video=v, start=s, end=e, type=t, setting="bag" if v <= 4 else "air", **m, rule=rule(m, t)))
    out = ROOT / "detect" / "contact" / "physics_rule.json"
    out.write_text(json.dumps(dict(params=dict(STOP_MS_CONTACT=STOP_MS_CONTACT, STOP_MS_AIR=STOP_MS_AIR,
                                               EXT_SHORT=EXT_SHORT, EXT_FULL=EXT_FULL), punches=rows)))
    import pandas as pd
    df = pd.DataFrame(rows)
    print("=== raw measurements: median by setting and punch type (bag = V1-V4, air = V5-V10) ===")
    print(df.groupby(["type", "setting"])[["stop_ms", "ext_stop", "ext_max", "peak_speed"]].median().round(2).to_string())
    print("\n=== rule outcome by setting ===")
    print(pd.crosstab([df.setting], df.rule, normalize="index").round(2).to_string())
    print("\n=== rule outcome by video ===")
    print(pd.crosstab(df.video, df.rule, normalize="index").round(2).to_string())


if __name__ == "__main__":
    main()
