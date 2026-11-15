"""What the counter reports is a TOTAL of landed punches per bout, not a verdict per punch. How close are the totals?
Uses the existing fully automatic outputs (detector -> contact scan -> landed model, held-out gym for every bout).
Per bout:  truth = referee punches to head/body;  estimates:
  counted     every counted punch (no landed step)
  hard        counted punches the model calls landed
  soft        sum of P(landed) over counted punches
Each estimate is multiplied by ONE factor fitted on the OTHER gym's bouts (so nothing is tuned on the bout scored).
Reports the typical error of the bout total and how well bouts are ranked (correlation).
    python olympic/landed_totals.py [cl_large]
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)

def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "cl_large"
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); N = len(clips); ev = json.load(open(os.path.join(HERE, "events.json")))
    S = np.full((N, 13, 4), np.nan)
    for p in pre.split("+"):
        Sp = np.full((N, 13, 4), np.nan)
        for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", p + "_kam*.npz"))):
            z = np.load(f); Sp[z["chain_idx"]] = z["chain_scan"]
        S = Sp if np.isnan(S).all() else (S + Sp) / 2
    k = np.nan_to_num(1 - S[:, :, 3], nan=-1).argmax(1); p3 = S[np.arange(N), k, :3]; p3 = p3 / p3.sum(1, keepdims=True); ok = np.isfinite(p3[:, 0])
    bouts = sorted({c["id"] for c in clips}); cid = np.array([c["id"] for c in clips]); gym = np.array(["kam2" if "kam2" in b else "kam4" for b in bouts])
    truth = np.array([sum(("O_" + e["video"] == b) and e["outcome"] in ("head", "body") for e in ev) for b in bouts], float)
    thrown = np.array([sum("O_" + e["video"] == b for e in ev) for b in bouts], float)
    est = dict(counted=np.array([(cid == b).sum() for b in bouts], float),
               hard=np.array([((cid == b) & ok & (p3.argmax(1) == 0)).sum() for b in bouts], float),
               soft=np.array([np.nansum(p3[(cid == b), 0]) for b in bouts], float))
    print(f"{len(bouts)} bouts; referee: {thrown.sum():.0f} punches, {truth.sum():.0f} landed (per bout: median {np.median(truth):.0f}, range {truth.min():.0f}-{truth.max():.0f}); landed share per bout {np.percentile(truth / thrown, [10, 50, 90]).round(2).tolist()}")
    out = {}
    for name, e in est.items():
        cal = np.empty(len(bouts))
        for g in ("kam2", "kam4"):
            f = truth[gym != g].sum() / e[gym != g].sum(); cal[gym == g] = e[gym == g] * f
        err = np.abs(cal - truth) / truth; r = np.corrcoef(cal, truth)[0, 1]
        print(f"  {name:8s} raw total {e.sum():6.0f} | after one factor from the other gym: total {cal.sum():6.0f} ({cal.sum() / truth.sum() - 1:+.1%}), bout error median {np.median(err):.1%}, mean {err.mean():.1%}, 90th pct {np.percentile(err, 90):.1%}, within 15%: {np.mean(err <= 0.15):.0%} of bouts | correlation with truth {r:.2f}")
        out[name] = dict(median_err=float(np.median(err)), mean_err=float(err.mean()), corr=float(r))
    # landed SHARE per bout (accuracy of the puncher): does the model's share follow the referee's?
    sh_t = truth / thrown; sh_s = est["soft"] / est["counted"]; sh_h = est["hard"] / est["counted"]
    print(f"  landed share per bout (landed / thrown): correlation with the referee's share: soft {np.corrcoef(sh_s, sh_t)[0, 1]:.2f}, hard {np.corrcoef(sh_h, sh_t)[0, 1]:.2f}; referee share spread (sd) {sh_t.std():.3f}, soft {sh_s.std():.3f}")
    json.dump(out, open(os.path.join(ROOT, "runs", f"olympic_landed_totals_{pre}.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
