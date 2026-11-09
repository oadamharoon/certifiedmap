"""E18 analysis: team gains under resampled start placements (5 worlds x 8 placements)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from scipy import stats
from runner import load, RESULTS
from analysis import ci95
df, _ = load("e18")
last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(
    w=("cfg_seed", "first"), st=("cfg_start_seed", "first"), p=("cfg_planner", "first"),
    het=("cfg_p_flip_agents", lambda v: len(v.iloc[0]) > 0), c=("c_whole", "mean"), covmin=("cov_explored", "min"), sep=("team_sep", "mean"))
rows = []
for het in [False, True]:
    d = last[last.het == het].pivot_table(index=["w", "st"], columns="p", values="c")
    for other in ["entropy", "entropy_g"]:
        if other not in d: continue
        diff = (d.certified - d[other]).dropna()
        m, h = ci95(diff.values)
        # variance decomposition: world vs placement
        g = diff.groupby(level="w")
        between = g.mean().var(ddof=1); within = g.var(ddof=1).mean()
        # per-world sign consistency
        per_world = g.mean()
        rows.append(dict(het=het, other=other, n=len(diff), diff=m, ci=h, p=stats.ttest_1samp(diff, 0).pvalue,
                         frac_pos=float((diff > 0).mean()), worlds_pos=int((per_world > 0).sum()), sd_between_worlds=np.sqrt(between), sd_within_world=np.sqrt(within),
                         c_cert=d.certified.mean(), c_other=d[other].mean(), sd_c_cert_within=np.sqrt(d.certified.groupby(level="w").var(ddof=1).mean())))
out = pd.DataFrame(rows); pd.set_option("display.width", 250); print(out.round(3).to_string())
print("coverage min over agents, all runs >= 0.95:", bool((last.covmin >= 0.95).all()), "min", round(last.covmin.min(), 3))
os.makedirs(os.path.join(RESULTS, "e18_analysis"), exist_ok=True); out.to_csv(os.path.join(RESULTS, "e18_analysis", "summary.csv"), index=False)
