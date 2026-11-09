"""E2: empirical coverage on the uniform explored law vs nominal, by planner x weighting x alpha."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from runner import load, RESULTS
from analysis import ci95

df, runs = load("e2")
out_dir = os.path.join(RESULTS, "e2_analysis"); os.makedirs(out_dir, exist_ok=True)
last = df.sort_values("block").groupby("rid").tail(1)
# per-block coverage (the certificate should hold at every block): average over blocks with explored > 5%
blk = df[df.explored > 0.05].groupby("rid").agg(cov_block_mean=("cov_explored", "mean"), cov_block_min=("cov_explored", "min"))
last = last.merge(blk, on="rid")
rows = []
for (p, w, a), d in last.groupby(["cfg_planner", "cfg_weighted", "cfg_alpha"]):
    m, h = ci95(d.cov_explored.values); mb, hb = ci95(d.cov_block_mean.values); mn, _ = ci95(d.cov_block_min.values)
    c, hc = ci95(d.c_explored.values); e, he = ci95(d["empty"].values / d.explored.values); am, ham = ci95(d.ambiguous.values / d.explored.values)
    acc, hacc = ci95(d.acc_committed.values)
    rows.append(dict(planner=p, weighted=w, alpha=a, nominal=1 - a, cov_final=m, cov_final_ci=h, cov_block=mb, cov_block_min=mn,
                     frac_runs_below=float((d.cov_explored < 1 - a).mean()), c_expl=c, c_ci=hc, empty=e, ambiguous=am, acc=acc, n=len(d)))
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
t.to_csv(os.path.join(out_dir, "validity.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, w in zip(axes, [True, False]):
    for p in ["lawnmower", "certified", "entropy"]:
        d = t[(t.weighted == w) & (t.planner == p)].sort_values("alpha")
        ax.errorbar(d.nominal, d.cov_final, yerr=d.cov_final_ci, marker="o", label=p, capsize=2)
    ax.plot([0.78, 1], [0.78, 1], "k--", lw=0.8, label="nominal")
    ax.set_title("weighted (ours)" if w else "unweighted"); ax.set_xlabel(r"nominal coverage $1-\alpha$"); ax.grid(alpha=0.3)
axes[0].set_ylabel("empirical coverage (uniform explored law)"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "validity.png"), dpi=130)
print("saved", out_dir)
