"""E33 analysis: is the heterogeneity result robust to a drifting pose estimate, to team size, and to where the
robots start? Paired differences in certified fraction against the coverage-mask planner, Holm-corrected within
condition, exactly as in e32_analysis."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load
from analysis import ci95
from compare import holm, paired_t

COLS = ["c_whole", "cov_explored", "acc_committed", "undecided_whole"]
REF = "entropy_g"


def label(r):
    dr = tuple(r.cfg_drift_agents) if isinstance(r.cfg_drift_agents, (list, tuple)) else ()
    pf = tuple(r.cfg_p_flip_agents) if isinstance(r.cfg_p_flip_agents, (list, tuple)) else ()
    se = tuple(r.cfg_sigma_e_agents) if isinstance(r.cfg_sigma_e_agents, (list, tuple)) else ()
    N = int(r.cfg_N)
    if r.cfg_start_seed >= 0:                       # placement-resampling replicate of an E32 condition
        return ("placement: flip" if pf else "placement: registration") + f" (start {int(r.cfg_start_seed)})"
    if N == 5:
        if dr: return "N=5 drift"
        if sum(1 for x in pf if x > 0) == 2: return "N=5 two unreliable"
        if pf: return "N=5 one unreliable"
        return "N=5 clean"
    if dr and pf: return "drift + flip"
    if dr: return "drift_hi" if max(dr) > 0.03 else "drift_lo"
    return "other"


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df, _ = load("e33")
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    run = last.groupby(["rid"] + [c for c in df.columns if c.startswith("cfg_")], dropna=False)[COLS].mean().reset_index()
    run["cond"] = run.apply(label, axis=1)
    order = ["drift_lo", "drift_hi", "drift + flip", "N=5 clean", "N=5 one unreliable", "N=5 two unreliable",
             "N=5 drift"] + sorted(c for c in run.cond.unique() if c.startswith("placement"))

    print("=== endpoint by condition and planner (20 worlds) ===")
    t = run.groupby(["cond", "cfg_planner"])[COLS].mean().round(3)
    print(t.loc[[c for c in order if c in t.index.get_level_values(0)]].to_string())

    print(f"\n=== paired difference in certified fraction vs {REF} (Holm within condition) ===")
    for cond in order:
        d = run[run.cond == cond]
        if d.empty or REF not in set(d.cfg_planner):
            continue
        piv = d.pivot_table(index="cfg_seed", columns="cfg_planner", values="c_whole")
        others = [p for p in piv.columns if p != REF]
        diffs = [(piv[p] - piv[REF]).dropna().values for p in others]
        ps = holm([paired_t(x) for x in diffs])
        print(f"  {cond}:")
        for p, x, pv in zip(others, diffs, ps):
            m, h = ci95(x)
            print(f"    {p:12s} - {REF}: {m:+.3f} +- {h:.3f}   holm p = {pv:.3f}{'  *' if pv < 0.05 else ''}")
