"""Generic planner comparison for an experiment: endpoint table, paired differences vs a reference
planner at several distances with Holm correction, and bands/FCR diagnostics."""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load, RESULTS
from analysis import ci95

def holm(pvals):
    p = np.asarray(pvals); m = len(p); o = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj

def paired_t(x):
    from scipy.stats import ttest_1samp
    x = np.asarray(x); x = x[~np.isnan(x)]
    if len(x) < 3 or np.allclose(x, 0): return 1.0
    return float(ttest_1samp(x, 0.0).pvalue)

def main(exp, group, ref="certified", dists=(50, 100, 150), metric="c_whole", extra_cols=()):
    df, runs = load(exp)
    if group == "cond":
        df["cond"] = "sr=" + df.cfg_sigma_r.astype(str) + ",wt=" + df.cfg_wall_thick.astype(str)
    if group not in df.columns:
        df[group] = "all"
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    agg = {c: (c, "mean") for c in ["c_whole", "cov_explored", "acc_committed", "empty", "ambiguous", "nodata", "path_len", "c_fcr", "acc_fcr", "fdr_marginal", "fdr_fcr",
                                    "band_lo", "band_mid", "undec_share_nocal", "team_sep", "team_overlap", "greedy_over_joint"] + list(extra_cols) if c in last.columns}
    lr = last.groupby("rid").agg(planner=("cfg_planner", "first"), seed=("cfg_seed", "first"), grp=(group, "first"), **agg).reset_index()
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    cols = [c for c in ["c_whole", "cov_explored", "acc_committed", "empty", "ambiguous", "nodata", "c_fcr", "fdr_marginal", "fdr_fcr", "band_lo", "band_mid", "undec_share_nocal", "team_sep", "team_overlap", "greedy_over_joint"] if c in lr.columns]
    rows = []
    for (g, p), d in lr.groupby(["grp", "planner"]):
        r = dict(grp=g, planner=p, n=len(d))
        for c in cols:
            m, h = ci95(d[c].values); r[c] = round(m, 3); r[c + "±"] = round(h, 3)
        rows.append(r)
    t = pd.DataFrame(rows); print("=== endpoint ==="); print(t.to_string(index=False))
    out = os.path.join(RESULTS, exp + "_analysis"); os.makedirs(out, exist_ok=True); t.to_csv(os.path.join(out, "endpoint.csv"), index=False)
    # paired differences vs ref at distances (interpolated on agent-averaged curves), Holm across all tests in the table
    print(f"\n=== paired {metric}: {ref} - other at distances {dists} (mean ± 95% CI; Holm-adjusted p) ===")
    per = {}
    for rid, dr in df.groupby("rid"):
        g = dr.groupby("block").agg(x=("path_len", "mean"), y=(metric, "mean")).sort_values("x")
        per[(dr[group].iloc[0], dr.cfg_seed.iloc[0], dr.cfg_planner.iloc[0])] = np.interp(dists, g.x.values, g.y.values, right=g.y.values[-1])
    rows, pv = [], []
    for grp in sorted({k[0] for k in per}, key=str):
        for other in sorted({k[2] for k in per if k[0] == grp}):
            if other == ref: continue
            diffs = np.array([per[(grp, s, ref)] - per[(grp, s, other)] for s in sorted({k[1] for k in per if k[0] == grp}) if (grp, s, ref) in per and (grp, s, other) in per])
            if len(diffs) == 0: continue
            for i, dd in enumerate(dists):
                m, h = ci95(diffs[:, i]); pv.append(paired_t(diffs[:, i]))
                rows.append(dict(grp=grp, other=other, dist=dd, n=len(diffs), diff=m, ci=h))
    pt = pd.DataFrame(rows)
    if len(pt):
        pt["p_holm"] = holm(pv); pt["sig"] = pt.p_holm < 0.05
        print(pt.round(3).to_string(index=False)); pt.to_csv(os.path.join(out, "paired.csv"), index=False)
    return t, pt

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("exp"); ap.add_argument("--group", default="cfg_p_flip"); ap.add_argument("--ref", default="certified")
    ap.add_argument("--dists", default="50,100,150"); ap.add_argument("--metric", default="c_whole")
    a = ap.parse_args(); main(a.exp, a.group, a.ref, tuple(float(x) for x in a.dists.split(",")), a.metric)
