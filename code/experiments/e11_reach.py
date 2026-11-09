"""E11: reachable observable fraction per floor plan (from the start pose), and c relative to it."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import ndimage
from runner import load
from analysis import ci95
from certmap.world import World, sample_starts
df, runs = load("e11")
last = df.sort_values("block").groupby("rid").tail(1)
reach = {}
for k in range(20):
    occ = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "houseexpo", f"he{k:02d}.npy")); w = World(occ, 0.25)
    st = sample_starts(w, 1, np.random.default_rng(k))[0]
    lab, n = ndimage.label(~occ); comp = lab[tuple(w.cell_of(st[None, :2])[::-1])][0]
    free_reach = (lab == comp)
    surf = occ & ndimage.binary_dilation(free_reach, structure=np.ones((3, 3)))
    reach[k] = float((free_reach | surf).sum() / w.obs.sum())
last["reach"] = last.cfg_seed.map(reach); last["c_rel"] = last.c_whole / last.reach
print("reachable observable fraction: mean %.3f min %.3f max %.3f" % (np.mean(list(reach.values())), min(reach.values()), max(reach.values())))
for p, d in last.groupby("cfg_planner"):
    m, h = ci95(d.c_rel.values); m2, h2 = ci95(d.c_whole.values); print(f"{p:10s} c={m2:.3f}±{h2:.3f}  c/reachable={m:.3f}±{h:.3f}  cov={d.cov_explored.mean():.3f}")
a = last[last.cfg_planner == "certified"].set_index("cfg_seed").c_rel
for o in ["entropy", "lse", "frontier", "lawnmower", "random"]:
    b = last[last.cfg_planner == o].set_index("cfg_seed").c_rel; j = a.index.intersection(b.index); m, h = ci95((a[j] - b[j]).values)
    print(f"  paired c/reach certified - {o}: {m:+.3f} ± {h:.3f}")
