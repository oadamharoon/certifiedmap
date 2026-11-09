"""E16 analysis: exact block-combined certificate (Corollary 1) vs pooled certificate."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load, RESULTS
from analysis import ci95
df, _ = load("e16")
last = df[df.block == df.groupby("rid").block.transform("max")]
rows = []
for (N, a), g in last.groupby(["cfg_N", "cfg_alpha"]):
    g = g.groupby("cfg_seed").mean(numeric_only=True)
    m = lambda k: ci95(g[k].values)
    rows.append(dict(N=N, alpha=a, n=len(g), cov_pooled=m("cov_explored"), cov_comb=m("cov_comb"), c_pooled=m("c_whole"), c_comb=m("c_comb"),
                     dc=ci95((g.c_whole - g.c_comb).values), acc_pooled=m("acc_committed"), acc_comb=m("acc_comb"), empty_pooled=m("empty"), empty_comb=m("empty_comb"),
                     frac_below_pooled=float((g.cov_explored < 1 - a).mean()), frac_below_comb=float((g.cov_comb < 1 - a).mean())))
out = pd.DataFrame(rows)
pd.set_option("display.width", 250); print(out.to_string())
os.makedirs(os.path.join(RESULTS, "e16_analysis"), exist_ok=True); out.to_csv(os.path.join(RESULTS, "e16_analysis", "summary.csv"), index=False)
