"""E1 analysis: c(distance) curves, endpoint table, distance-to-level, trajectory divergence."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from runner import load, RESULTS
from analysis import curves, final_table, distance_to, visitation_hist, jsd, ci95, GRID

exp = sys.argv[1] if len(sys.argv) > 1 else "e1"
df, runs = load(exp)
df["cond"] = "sr=" + df.cfg_sigma_r.astype(str) + ",wt=" + df.cfg_wall_thick.astype(str)
out_dir = os.path.join(RESULTS, exp + "_analysis"); os.makedirs(out_dir, exist_ok=True)
PL = ["certified", "entropy", "lse", "frontier", "lawnmower", "random"]

# ---- endpoint table per condition ------------------------------------------------
cols = ["c_whole", "c_explored", "cov_explored", "acc_committed", "empty", "ambiguous", "nodata", "explored", "path_len"]
tab = final_table(df, cols, keys=("cond", "cfg_planner"))
pd.set_option("display.width", 250)
print("=== endpoint (at budget) ===")
print(tab.set_index(["cond", "cfg_planner"])[[c for c in tab.columns if c in cols or c.endswith("_ci")]].round(3).to_string())
tab.to_csv(os.path.join(out_dir, "endpoint.csv"), index=False)

# ---- curves ------------------------------------------------------------------------
conds = sorted(df.cond.unique())
fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.4), sharey=True)
for ax, cond in zip(np.atleast_1d(axes), conds):
    cv = curves(df[df.cond == cond], "c_whole", keys=("cfg_planner",))
    for p in PL:
        if p in cv:
            m, h, n = cv[p]
            ax.plot(GRID, m, label=p); ax.fill_between(GRID, m - h, m + h, alpha=0.15)
    ax.set_title(cond, fontsize=9); ax.set_xlabel("path length (m)"); ax.grid(alpha=0.3)
np.atleast_1d(axes)[0].set_ylabel("certified fraction $c$"); np.atleast_1d(axes)[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "c_vs_distance.png"), dpi=130)

# ---- c at fixed distances, paired differences vs entropy ----------------------------
print("\n=== paired difference c(certified) - c(other) at distances (mean [95% CI]) ===")
rows = []
for cond in conds:
    d = df[df.cond == cond]
    per = {}
    for (rid,), dr in d.groupby(["rid"]):
        per[(dr.cfg_seed.iloc[0], dr.cfg_planner.iloc[0])] = np.interp([25, 50, 75, 100], dr.groupby("block").path_len.mean().values, dr.groupby("block").c_whole.mean().values)
    for other in PL:
        if other == "certified": continue
        diffs = np.array([per[(s, "certified")] - per[(s, other)] for s in sorted({k[0] for k in per}) if (s, other) in per and (s, "certified") in per])
        m, h = ci95(diffs, 0)
        rows.append(dict(cond=cond, other=other, n=len(diffs), **{f"d{x}": f"{m[i]:+.3f}±{h[i]:.3f}" for i, x in enumerate([25, 50, 75, 100])}))
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "paired_diff.csv"), index=False)

# ---- paired endpoint differences on the partition ------------------------------------
print("\n=== paired endpoint differences certified - entropy / lse (mean ± 95% CI) ===")
last = df.sort_values("block").groupby("rid").tail(1)
for other in ["entropy", "lse"]:
    for col in ["c_whole", "ambiguous", "nodata", "empty", "explored"]:
        v = []
        for cond in conds:
            d = last[last.cond == cond]
            a = d[d.cfg_planner == "certified"].set_index("cfg_seed")[col]; b = d[d.cfg_planner == other].set_index("cfg_seed")[col]
            j = a.index.intersection(b.index); v.append((a[j] - b[j]).values)
        allv = np.concatenate(v); m, h = ci95(allv)
        print(f"  vs {other:8s} {col:10s}: {m:+.4f} ± {h:.4f} (n={len(allv)})  per-cond: " + "  ".join(f"{ci95(x)[0]:+.3f}±{ci95(x)[1]:.3f}" for x in v))

# ---- distance to reach c* ------------------------------------------------------------
print("\n=== distance to reach c >= level (mean over runs that reached it; fraction reached) ===")
for lvl in [0.5, 0.6, 0.7]:
    dt = distance_to(df, "c_whole", lvl, keys=("cond", "cfg_planner"))
    g = dt.groupby(["cond", "cfg_planner"]).agg(dist=("dist", "mean"), reached=("reached", "mean")).unstack("cfg_planner")
    print(f"-- level {lvl}"); print(g.round(2).to_string())

# ---- trajectory divergence -------------------------------------------------------
print("\n=== visitation-histogram JSD between planners (same world), mean ± 95% CI ===")
rows = []
for cond in conds:
    d = df[df.cond == cond]
    byseed = {}
    for (rid,), dr in d.groupby(["rid"]):
        byseed[(dr.cfg_seed.iloc[0], dr.cfg_planner.iloc[0])] = visitation_hist(runs[rid]["traj"][0])
    for a, b in itertools.combinations(PL, 2):
        v = [jsd(byseed[(s, a)], byseed[(s, b)]) for s in sorted({k[0] for k in byseed}) if (s, a) in byseed and (s, b) in byseed]
        m, h = ci95(v)
        rows.append(dict(cond=cond, pair=f"{a}-{b}", jsd=m, ci=h, n=len(v)))
jt = pd.DataFrame(rows)
print(jt[jt.pair.str.startswith("certified")].round(3).to_string(index=False))
jt.to_csv(os.path.join(out_dir, "jsd.csv"), index=False)
print("saved to", out_dir)
