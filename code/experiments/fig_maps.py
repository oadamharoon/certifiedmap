"""Figure: fused partition maps at equal budget.
Row 1: one robot under three objectives (seed 3, 75 m).
Row 2: team of three with one label-unreliable teammate under three coordination states (seed 14, 100 m total);
seed 14 is the world whose certified-minus-mask and certified-minus-plans gaps are closest to the 20-world means."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from certmap.sim import Simulation, Config
from certmap.viz import plot_run, CMAP, NAMES
from runner import RESULTS

seed1 = int(sys.argv[1]) if len(sys.argv) > 1 else 3
seed2 = int(sys.argv[2]) if len(sys.argv) > 2 else 14
team_row = "--team" in sys.argv                  # optional second row (heterogeneous team); the paper uses one row
plt.rcParams.update({"font.size": 8})
fig, axes = plt.subplots(2 if team_row else 1, 3, figsize=(7.0, 3.9 if team_row else 1.95), squeeze=False)
stats = []
for ax, pl in zip(axes[0], ["certified", "entropy", "frontier"]):
    out = Simulation(Config(seed=seed1, N=1, planner=pl, budget=75.0, save_maps=True)).run()
    r = pd.DataFrame(out["log"]).iloc[-1]
    plot_run(out, ax, "")
    ax.set_title(pl, fontsize=8, pad=2)
    stats.append(f"row1 {pl}: c={r.c_whole:.3f} cov={r.cov_explored:.3f} undecided={r.ambiguous+r.nodata:.3f} empty={r['empty']:.3f}")
for ax, pl, title in zip(axes[1] if team_row else [], ["certified", "entropy_g", "entropy"], ["fused undecided set", "coverage mask", "exchanged plans"]):
    out = Simulation(Config(seed=seed2, N=3, topology="mesh", planner=pl, budget=100.0 / 3, sigma_r=0.05, p_flip_agents=[0.0, 0.0, 0.10], save_maps=True)).run()
    plot_run(out, ax, "")
    ax.set_title(title, fontsize=8, pad=2)
    r = pd.DataFrame(out["log"]); r = r[r.block == r.block.max()]
    stats.append(f"row2 {pl}: c={r.c_whole.mean():.3f} undecided-with-data={r.ambiguous.mean():.3f} nodata={r.nodata.mean():.3f} overlap={r.team_overlap.mean() if 'team_overlap' in r else float('nan'):.3f}")
fig.legend(handles=[Patch(color=CMAP(i), label=n) for i, n in enumerate(NAMES)] + [plt.Line2D([], [], color="C0", lw=1, label="trajectory")],
           loc="lower center", ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(pad=0.2, rect=(0, 0.05 if team_row else 0.1, 1, 0.98 if team_row else 0.96))
print("\n".join(stats))
os.makedirs(os.path.join(RESULTS, "figs"), exist_ok=True)
plt.savefig(os.path.join(RESULTS, "figs", "maps.pdf")); plt.savefig(os.path.join(RESULTS, "figs", "maps.png"), dpi=130)
print("saved")
