"""One-shot report: every quantity cited in the paper, from cached results."""
import sys, os, itertools, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load, RESULTS
from analysis import ci95, visitation_hist, jsd
from compare import main as compare, holm, paired_t
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 50)
PY = sys.executable

def hdr(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)

def endpoint(exp, keys, cols=("c_whole", "cov_explored", "acc_committed", "empty", "ambiguous", "nodata", "c_fcr", "fdr_marginal", "band_lo", "band_mid", "undec_share_nocal", "path_len")):
    df, runs = load(exp)
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    extra = [k for k in ["cfg_seed", "cfg_planner"] if k not in keys]
    g = last.groupby(["rid"] + list(keys) + extra)[[c for c in cols if c in last.columns]].mean().reset_index()
    rows = []
    for k, d in g.groupby(list(keys)):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,))); r["n"] = len(d)
        for c in cols:
            if c in d.columns:
                m, h = ci95(d[c].values); r[c] = round(m, 3); r[c + "±"] = round(h, 3)
        rows.append(r)
    return pd.DataFrame(rows), df, runs, g

def paired(g, keys_cond, ref, others, col="c_whole"):
    rows = []
    for k, d in g.groupby(keys_cond) if keys_cond else [("all", g)]:
        a = d[d.cfg_planner == ref].set_index("cfg_seed")[col]
        for o in others:
            b = d[d.cfg_planner == o].set_index("cfg_seed")[col]; j = a.index.intersection(b.index)
            if len(j): m, h = ci95((a[j] - b[j]).values); rows.append(dict(cond=k, other=o, n=len(j), diff=round(m, 4), ci=round(h, 4), p=paired_t((a[j] - b[j]).values)))
    t = pd.DataFrame(rows)
    if len(t): t["p_holm"] = holm(t.p.values).round(3)
    return t

if __name__ == "__main__":
    what = sys.argv[1:] or ["e1", "e2", "e3", "e5", "e6", "e7", "e8", "e9", "e10", "e11", "e4"]
    if "e1" in what:
        hdr("E1 endpoint / paired (Holm over the table) / distances"); compare("e1", "cond", "certified", (25, 50, 75, 100, 150))
        df, runs = load("e1"); df["cond"] = list(zip(df.cfg_sigma_r, df.cfg_wall_thick))
        hdr("E1 partition paired diffs at budget (pooled over conditions)")
        last = df.sort_values("block").groupby("rid").tail(1)
        for o in ["entropy", "lse"]:
            for col in ["c_whole", "ambiguous", "nodata", "empty"]:
                v = []
                for cond, d in last.groupby("cond"):
                    a = d[d.cfg_planner == "certified"].set_index("cfg_seed")[col]; b = d[d.cfg_planner == o].set_index("cfg_seed")[col]; j = a.index.intersection(b.index); v.append((a[j] - b[j]).values)
                allv = np.concatenate(v); m, h = ci95(allv); print(f"  cert - {o:8s} {col:10s}: {m:+.4f} ± {h:.4f} (n={len(allv)})")
        hdr("E1 distance to c>=0.7 (mean over runs that reached; fraction reached)")
        from analysis import distance_to
        dt = distance_to(df, "c_whole", 0.7, keys=("cond", "cfg_planner")); print(dt.groupby(["cond", "cfg_planner"]).agg(dist=("dist", "mean"), reached=("reached", "mean")).unstack("cfg_planner").round(2).to_string())
        if os.path.isdir(os.path.join(RESULTS, "e1b")):
            hdr("E1b JSD reference"); subprocess.run([PY, os.path.join(os.path.dirname(__file__), "e1b_analysis.py")])
    if "e2" in what:
        hdr("E2 validity"); subprocess.run([PY, os.path.join(os.path.dirname(__file__), "e2_analysis.py")])
        t, df, runs, g = endpoint("e2", ("cfg_weighted", "cfg_alpha", "cfg_planner"), cols=("c_whole", "c_fcr", "thr_fcr", "fdr_marginal", "fdr_fcr", "acc_fcr"))
        hdr("E2 FCR (e-BH) vs marginal"); print(t.to_string(index=False))
    if "e3" in what:
        hdr("E3 teams"); subprocess.run([PY, os.path.join(os.path.dirname(__file__), "e3_analysis.py")])
        t, df, runs, g = endpoint("e3", ("cfg_N", "cfg_topology", "cfg_planner"), cols=("c_whole", "nodata", "ambiguous", "team_sep", "team_overlap", "q_i"))
        print(t.to_string(index=False))
        g["team"] = g.cfg_N.astype(str) + g.cfg_topology
        print(paired(g, ["team"], "certified", ["entropy", "entropy_u", "entropy_g"]).to_string(index=False))
    if "e5" in what:
        hdr("E5 margin score"); compare("e5", "cond", "certified", (50, 100, 150))
    if "e6" in what:
        hdr("E6 label flips"); t, df, runs, g = endpoint("e6", ("cfg_score", "cfg_p_flip", "cfg_planner"), cols=("c_whole", "cov_explored", "acc_committed", "empty", "ambiguous", "band_lo", "band_mid", "undec_share_nocal", "undec_share_mid", "q_i"))
        print(t.to_string(index=False)); t.to_csv(os.path.join(RESULTS, "e6_endpoint.csv"), index=False)
        g["sc"] = g.cfg_score + "@" + g.cfg_p_flip.astype(str); print(paired(g, ["sc"], "certified", ["entropy", "lse"]).to_string(index=False))
    if "e7" in what:
        hdr("E7 ablations"); subprocess.run([PY, os.path.join(os.path.dirname(__file__), "e7_analysis.py")])
        df, _ = load("e7"); ref = df[(df.cfg_planner.isin(["certified", "entropy", "frontier"])) & (df.cfg_alpha == 0.05) & (df.cfg_gate_r == 0.5) & (df.cfg_safety == 0.7) & (df.cfg_noise_var == 0.3) & (df.cfg_seg_len == 5.0) & (df.cfg_kde_bw == 0.5) & (df.cfg_eps_ucb == 0.0) & (df.cfg_ell == 0.4) & (df.cfg_front_dmin < 0) & (df.cfg_trav_thr == 0.3) & (df.cfg_trav_inflate == 0)]
        last = ref.sort_values("block").groupby("rid").tail(1)
        hdr("Mechanism bands at budget (E7 reference runs, all planners)")
        print(last.groupby("cfg_planner")[["band_lo", "band_mid", "band_hi", "committed_in_lo", "committed_in_mid", "undec_share_nocal", "undec_share_mid", "nocal_frac", "q_i"]].mean().round(3).to_string())
    if "e8" in what:
        hdr("E8 greedy vs joint"); df, _ = load("e8"); d = df[df.block >= 1]
        print(d.groupby("cfg_alloc").greedy_over_joint.describe().round(3).to_string())
        last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(alloc=("cfg_alloc", "first"), seed=("cfg_seed", "first"), c=("c_whole", "mean"))
        a = last[last.alloc == "greedy"].set_index("seed").c; b = last[last.alloc == "joint"].set_index("seed").c; j = a.index.intersection(b.index); m, h = ci95((b[j] - a[j]).values)
        print(f"c(joint) - c(greedy) = {m:+.4f} ± {h:.4f}; greedy c={a.mean():.3f}, joint c={b.mean():.3f}")
    if "e9" in what:
        hdr("E9 team size at fixed per-robot budget"); t, df, runs, g = endpoint("e9", ("cfg_N", "cfg_planner"), cols=("c_whole", "cov_explored", "nodata", "ambiguous", "team_sep", "team_overlap", "q_i", "wall"))
        print(t.to_string(index=False)); print(paired(g, ["cfg_N"], "certified", ["entropy", "entropy_u", "entropy_g"]).to_string(index=False))
    if "e10" in what:
        hdr("E10 heteroscedastic"); compare("e10", "cfg_p_flip_range", "certified", (50, 100, 150))
    if "e11" in what:
        hdr("E11 floor plans"); compare("e11", "all", "certified", (50, 100, 150))
        subprocess.run([PY, os.path.join(os.path.dirname(__file__), "e11_reach.py")])
    if "e4" in what:
        hdr("E4 stopping"); subprocess.run([PY, os.path.join(os.path.dirname(__file__), "e4_stopping.py")])
