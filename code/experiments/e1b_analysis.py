"""E1b: trajectory divergence between planners vs the same planner under different randomness."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load, RESULTS
from analysis import visitation_hist, jsd, ci95

df1, runs1 = load("e1"); dfb, runsb = load("e1b")
for d in (df1, dfb):
    d["cond"] = "sr=" + d.cfg_sigma_r.astype(str) + ",wt=" + d.cfg_wall_thick.astype(str)
def hists(df, runs):
    h = {}
    for rid, dr in df.groupby("rid"):
        h[(dr.cond.iloc[0], dr.cfg_seed.iloc[0], dr.cfg_planner.iloc[0])] = visitation_hist(runs[rid]["traj"][0])
    return h
H1, Hb = hists(df1, runs1), hists(dfb, runsb)
rows = []
for p in ["certified", "entropy", "lse"]:
    v = [jsd(H1[k], Hb[k]) for k in H1 if k[2] == p and k in Hb]
    m, h = ci95(v); rows.append(dict(pair=f"{p} vs {p} (noise)", jsd=m, ci=h, n=len(v)))
for a, b in itertools.combinations(["certified", "entropy", "lse"], 2):
    v = [jsd(H1[k], H1[(k[0], k[1], b)]) for k in H1 if k[2] == a and (k[0], k[1], b) in H1]
    m, h = ci95(v); rows.append(dict(pair=f"{a} vs {b}", jsd=m, ci=h, n=len(v)))
    v = [jsd(H1[k], Hb[(k[0], k[1], b)]) for k in H1 if k[2] == a and (k[0], k[1], b) in Hb]
    m, h = ci95(v); rows.append(dict(pair=f"{a} vs {b} (noise)", jsd=m, ci=h, n=len(v)))
for b in ["frontier", "lawnmower", "random"]:
    v = [jsd(H1[k], H1[(k[0], k[1], b)]) for k in H1 if k[2] == "certified" and (k[0], k[1], b) in H1]
    m, h = ci95(v); rows.append(dict(pair=f"certified vs {b}", jsd=m, ci=h, n=len(v)))
t = pd.DataFrame(rows); print(t.round(3).to_string(index=False))
os.makedirs(os.path.join(RESULTS, "e1_analysis"), exist_ok=True); t.to_csv(os.path.join(RESULTS, "e1_analysis", "jsd_reference.csv"), index=False)
# endpoint c under different noise: same-planner repeatability of c
last1 = df1.sort_values("block").groupby("rid").tail(1); lastb = dfb.sort_values("block").groupby("rid").tail(1)
for p in ["certified", "entropy", "lse"]:
    a = last1[last1.cfg_planner == p].set_index(["cond", "cfg_seed"]).c_whole; b = lastb[lastb.cfg_planner == p].set_index(["cond", "cfg_seed"]).c_whole
    j = a.index.intersection(b.index); m, h = ci95(np.abs(a[j] - b[j]).values)
    print(f"{p}: mean |c - c'| across noise seeds = {m:.4f} ± {h:.4f}")
