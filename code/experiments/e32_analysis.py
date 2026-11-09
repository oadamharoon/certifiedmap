"""E32 analysis: does the certified state beat a coverage mask when a teammate is untrustworthy for reasons
other than flipped labels? Endpoint metrics per condition, paired differences against the coverage-mask planner
(entropy_g) with Holm correction, and the drop-the-untrusted-robot baseline (N=2)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load
from analysis import ci95
from compare import holm, paired_t

COLS = ["c_whole", "cov_explored", "acc_committed", "undecided_whole", "explored", "nodata"]
REF = "entropy_g"          # coverage-mask planner: the baseline the certified state must beat


def label(r):
    if r.cfg_N == 2:
        # two clean robots instead of three: either keeping the team's total distance (150 m each, the dropped
        # robot's budget transfers) or keeping each robot's own budget (100 m each, a third of capacity lost)
        return "drop: 2x150 m" if r.cfg_budget > 120 else "drop: 2x100 m"
    pf = tuple(r.cfg_p_flip_agents) if isinstance(r.cfg_p_flip_agents, (list, tuple)) else ()
    se = tuple(r.cfg_sigma_e_agents) if isinstance(r.cfg_sigma_e_agents, (list, tuple)) else ()
    if not pf and not se:
        return "clean"
    if pf and not se:
        return "flip"
    if se and not pf:
        return "pose_hi" if max(se) > 0.4 else "pose"
    return "mixed"


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df, _ = load("e32")
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    run = last.groupby(["rid"] + [c for c in df.columns if c.startswith("cfg_")], dropna=False)[COLS].mean().reset_index()
    run["cond"] = run.apply(label, axis=1)
    order = ["clean", "flip", "pose", "pose_hi", "mixed", "drop: 2x150 m", "drop: 2x100 m"]

    print("=== endpoint by condition and planner (mean over 20 worlds) ===")
    t = run.groupby(["cond", "cfg_planner"])[COLS].agg(["mean", "count"]).round(3)
    t.columns = [a if b == "mean" else "n" for a, b in t.columns]
    print(t.loc[[c for c in order if c in t.index.get_level_values(0)]].to_string())

    print(f"\n=== paired differences in certified fraction vs {REF} (Holm within condition) ===")
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

    print("\n=== certified fraction: heterogeneous conditions against the drop-untrusted baseline ===")
    for drop in ["drop: 2x150 m", "drop: 2x100 m"]:
        base = run[run.cond == drop].pivot_table(index="cfg_seed", columns="cfg_planner", values="c_whole")
        if base.empty:
            continue
        print(f"  against {drop}:")
        for cond in ["flip", "pose", "pose_hi", "mixed"]:
            d = run[run.cond == cond]
            if d.empty:
                continue
            piv = d.pivot_table(index="cfg_seed", columns="cfg_planner", values="c_whole")
            for p in ["certified", "entropy_g"]:
                if p in piv and p in base:
                    m, h = ci95((piv[p] - base[p]).dropna().values)
                    print(f"    {cond:8s} N=3 {p:10s} - {drop}: {m:+.3f} +- {h:.3f}")
