"""E36: does a longer planning horizon change the single-robot conclusion?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load
from analysis import ci95
from compare import holm, paired_t

if __name__ == "__main__":
    df, _ = load("e36")
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    run = last.groupby(["rid", "cfg_planner", "cfg_lookahead", "cfg_seed"])[["c_whole", "explored", "path_len"]].mean().reset_index()
    print("=== endpoint by acquisition and horizon (20 worlds) ===")
    print(run.groupby(["cfg_lookahead", "cfg_planner"])[["c_whole", "explored"]].mean().round(3).to_string())
    print("\n=== certified - entropy at each horizon (paired by world) ===")
    for la, d in run.groupby("cfg_lookahead"):
        piv = d.pivot_table(index="cfg_seed", columns="cfg_planner", values="c_whole")
        x = (piv["certified"] - piv["entropy"]).dropna().values
        m, h = ci95(x)
        print(f"  lookahead {la}: {m:+.3f} +- {h:.3f}  (p = {paired_t(x):.3f})")
    print("\n=== the horizon's own effect, per acquisition (paired by world) ===")
    for pl, d in run.groupby("cfg_planner"):
        piv = d.pivot_table(index="cfg_seed", columns="cfg_lookahead", values="c_whole")
        diffs, labels = [], []
        for la in [2, 3]:
            if la in piv and 1 in piv:
                diffs.append((piv[la] - piv[1]).dropna().values); labels.append(la)
        ps = holm([paired_t(x) for x in diffs])
        for la, x, pv in zip(labels, diffs, ps):
            m, h = ci95(x)
            print(f"  {pl:10s} lookahead {la} - 1: {m:+.3f} +- {h:.3f}  holm p = {pv:.3f}{'  *' if pv < 0.05 else ''}")
