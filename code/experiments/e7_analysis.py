"""E7: one-factor ablations; endpoint c/cov per level and paired certified-entropy."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load, RESULTS
from analysis import ci95
from e7_ablation import LEVELS
df, _ = load("e7")
last = df.sort_values("block").groupby("rid").tail(1)
ref = {"alpha": 0.05, "gate_r": 0.5, "safety": 0.7, "noise_var": 0.3, "seg_len": 5.0, "kde_bw": 0.5, "eps_ucb": 0.0, "ell": 0.4, "front_dmin": -1.0, "trav_thr": 0.3, "trav_inflate": 0}
def is_ref(r): return all(abs(r[f"cfg_{k}"] - v) < 1e-9 for k, v in ref.items())
last["is_ref"] = last.apply(is_ref, axis=1)
rows = []
def row(key, val, d):
    out = dict(factor=key, level=val)
    for p in ["certified", "entropy", "frontier"]:
        dp = d[d.cfg_planner == p]
        if len(dp) == 0: continue
        for c in ["c_whole", "cov_explored", "empty", "ambiguous", "band_lo", "undec_share_nocal"]:
            m, h = ci95(dp[c].values); out[f"{p[:4]}_{c}"] = round(m, 3)
        out[f"{p[:4]}_n"] = len(dp)
    a = d[d.cfg_planner == "certified"].set_index("cfg_seed").c_whole; b = d[d.cfg_planner == "entropy"].set_index("cfg_seed").c_whole
    j = a.index.intersection(b.index)
    if len(j): m, h = ci95((a[j] - b[j]).values); out["cert-entr"] = f"{m:+.3f}±{h:.3f}"
    return out
rows.append(row("reference", "-", last[last.is_ref]))
for key, vals in LEVELS.items():
    for v in vals:
        d = last[(np.abs(last[f"cfg_{key}"] - v) < 1e-9) & ~last.is_ref]
        if len(d): rows.append(row(key, v, d))
for dm in [1.0, 5.0]:
    d = last[(np.abs(last.cfg_front_dmin - dm) < 1e-9)]
    if len(d): rows.append(row("front_dmin", dm, d))
t = pd.DataFrame(rows); pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
cols = ["factor", "level", "cert_c_whole", "entr_c_whole", "cert-entr", "cert_cov_explored", "entr_cov_explored", "cert_empty", "cert_ambiguous", "cert_band_lo", "cert_undec_share_nocal", "fron_c_whole", "fron_cov_explored"]
print(t[[c for c in cols if c in t.columns]].to_string(index=False))
os.makedirs(os.path.join(RESULTS, "e7_analysis"), exist_ok=True); t.to_csv(os.path.join(RESULTS, "e7_analysis", "ablation.csv"), index=False)
