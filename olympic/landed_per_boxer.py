"""Totals PER FIGHTER (what the counter shows), fully automatic chain vs the referee, Olympic bouts, held-out gym.
The referee names no boxer, but the referee's box sits on the target: the puncher is the boxer whose head (torso for
body shots) is farther from the box (punch_tcn_oly.puncher_stream). Estimates per fighter: counted punches, and sum of P(landed); each scaled by one factor from the
other gym. Also: each fighter's SHARE of the bout's punches / landed punches, and whether the fighter who landed more
according to the referee is the one who landed more according to us.
    python olympic/landed_per_boxer.py [cl_large]
"""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "boxingweb"))
import punch_tcn_oly as po

def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "cl_large"
    clips = json.load(open(os.path.join(ROOT, "hpc_rtm", "lists", "chain_clips_v3.json"))); N = len(clips); ev = json.load(open(os.path.join(HERE, "events.json")))
    S = np.full((N, 13, 4), np.nan)
    for f in sorted(glob.glob(os.path.join(ROOT, "external", "landed_ft", pre + "_kam*.npz"))):
        z = np.load(f); S[z["chain_idx"]] = z["chain_scan"]
    k = np.nan_to_num(1 - S[:, :, 3], nan=-1).argmax(1); p3 = S[np.arange(N), k, :3]; p3 = p3 / p3.sum(1, keepdims=True)
    bouts = sorted({c["id"] for c in clips}); T = np.zeros((len(bouts), 2, 2)); E = np.zeros((len(bouts), 2, 2))        # [bout, boxer, (thrown, landed)]
    for bi, b in enumerate(bouts):
        z = np.load(os.path.join(ROOT, "external", "chain_streams", b + ".npz")); K = z["K"].astype(np.float64); fr = z["frame"]; F = len(fr); step = int(fr[1] - fr[0])
        for e in [e for e in ev if "O_" + e["video"] == b and e["end"] <= fr[-1]]:
            i = int(np.clip(round((e["start"] + e["end"]) / 2 / step), 0, F - 1)); w = 9 if e["hand"] == "left" else 10; bx = np.array(e["boxes"]); c = np.array([(bx[:, 1].min() + bx[:, 3].max()) / 2, (bx[:, 2].min() + bx[:, 4].max()) / 2])
            s = po.puncher_stream(K, i, c, e["outcome"])
            if s >= 0:
                T[bi, s, 0] += 1; T[bi, s, 1] += e["outcome"] in ("head", "body")
        for n, c in enumerate(clips):
            if c["id"] == b:
                s = "AB".index(c["boxer"]); E[bi, s, 0] += 1; E[bi, s, 1] += np.nan_to_num(p3[n, 0])
    gym = np.array(["kam2" if "kam2" in b else "kam4" for b in bouts]); cal = np.empty_like(E)
    for g in ("kam2", "kam4"):
        for q in (0, 1):
            cal[gym == g, :, q] = E[gym == g, :, q] * T[gym != g, :, q].sum() / E[gym != g, :, q].sum()
    print(f"{len(bouts)} bouts, {2 * len(bouts)} fighters; referee labels placed on a boxer: {int(T[:, :, 0].sum())}"); out = {}
    for q, name in ((0, "thrown"), (1, "landed")):
        t, e = T[:, :, q].ravel(), cal[:, :, q].ravel(); err = np.abs(e - t) / np.maximum(t, 1); sh_t = T[:, 0, q] / T[:, :, q].sum(1); sh_e = cal[:, 0, q] / cal[:, :, q].sum(1)
        big = np.abs(sh_t - 0.5) >= 0.05; agree = np.mean((sh_t[big] > 0.5) == (sh_e[big] > 0.5)); allb = np.mean((sh_t > 0.5) == (sh_e > 0.5))
        print(f"  {name}: per fighter (median {np.median(t):.0f} per bout) error median {np.median(err):.1%}, mean {err.mean():.1%}, within 15%: {np.mean(err <= 0.15):.0%} of fighters; correlation {np.corrcoef(t, e)[0, 1]:.2f}")
        print(f"     fighter A's share of the bout: mean error {np.mean(np.abs(sh_t - sh_e)) * 100:.1f} points (the referee's share spreads {sh_t.std() * 100:.1f} points), correlation {np.corrcoef(sh_t, sh_e)[0, 1]:.2f}; "
              f"same fighter ahead: {allb:.0%} of all bouts, {agree:.0%} of the {big.sum()} bouts where the referee's split is at least 55/45")
        out[name] = dict(median_err=float(np.median(err)), share_err=float(np.mean(np.abs(sh_t - sh_e))), share_corr=float(np.corrcoef(sh_t, sh_e)[0, 1]), same_leader=float(allb), same_leader_clear=float(agree), n_clear=int(big.sum()))
    json.dump(out, open(os.path.join(ROOT, "runs", f"olympic_landed_per_boxer_{pre}.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
