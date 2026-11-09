"""E3: team size / topology at fixed total budget."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from runner import load, RESULTS
from analysis import ci95, curves, GRID

df, runs = load("e3")
out_dir = os.path.join(RESULTS, "e3_analysis"); os.makedirs(out_dir, exist_ok=True)
df["team"] = df.cfg_N.astype(str) + "-" + df.cfg_topology
df["team_path"] = df.path_len * df.cfg_N            # total team distance
last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(
    team=("team", "first"), planner=("cfg_planner", "first"), N=("cfg_N", "first"), seed=("cfg_seed", "first"),
    c_whole=("c_whole", "mean"), c_explored=("c_explored", "mean"), cov=("cov_explored", "mean"), cov_min=("cov_explored", "min"),
    nodata=("nodata", "mean"), ambiguous=("ambiguous", "mean"), empty=("empty", "mean"), acc=("acc_committed", "mean"),
    eps=("eps_i", "mean"), q=("q_i", "mean"), path=("path_len", "mean"), wall=("wall", "max"))
rows = []
for (team, p), d in last.groupby(["team", "planner"]):
    r = dict(team=team, planner=p, n=len(d))
    for c in ["c_whole", "c_explored", "cov", "cov_min", "nodata", "ambiguous", "empty", "acc", "q", "wall"]:
        m, h = ci95(d[c].values); r[c] = m; r[c + "_ci"] = h
    rows.append(r)
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t[["team", "planner", "n", "c_whole", "c_whole_ci", "c_explored", "cov", "cov_min", "nodata", "ambiguous", "empty", "acc", "q", "wall"]].round(3).to_string(index=False))
t.to_csv(os.path.join(out_dir, "team.csv"), index=False)
# paired certified - entropy by team
for other in ["entropy", "entropy_u"]:
    print(f"\npaired c_whole(certified) - c_whole({other}) by team:")
    for team, d in last.groupby("team"):
        a = d[d.planner == "certified"].set_index("seed").c_whole; b = d[d.planner == other].set_index("seed").c_whole
        j = a.index.intersection(b.index)
        if len(j): m, h = ci95((a[j] - b[j]).values); print(f"  {team}: {m:+.3f} ± {h:.3f} (n={len(j)})")
# curves vs team distance
fig, ax = plt.subplots(figsize=(5, 3.6))
grid = np.arange(0, 100.1, 2.5)
for team in ["1-mesh", "2-mesh", "3-mesh", "5-mesh", "5-ring"]:
    d = df[(df.team == team) & (df.cfg_planner == "certified")]
    arr = []
    for rid, dr in d.groupby("rid"):
        g = dr.groupby("block").agg(x=("team_path", "mean"), y=("c_whole", "mean")).sort_values("x")
        arr.append(np.interp(grid, g.x.values, g.y.values))
    m, h = ci95(np.array(arr), 0)
    ax.plot(grid, m, label=team); ax.fill_between(grid, m - h, m + h, alpha=0.15)
ax.set_xlabel("total team path length (m)"); ax.set_ylabel("certified fraction $c$"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "team_curves.png"), dpi=130)
print("saved", out_dir)
