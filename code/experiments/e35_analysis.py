"""E35: does anything depend on our candidate generator? Compares the bearing-sector generator with an
RRT-exploration-style one (tree detects frontiers, grid planner navigates) under every acquisition, and re-tests
the heterogeneity result under both."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import load
from analysis import ci95
from compare import holm, paired_t

if __name__ == "__main__":
    pd.set_option("display.width", 200)
    df, _ = load("e35")
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    run = last.groupby(["rid", "cfg_planner", "cfg_generator", "cfg_N", "cfg_seed"]).agg(
        hetero=("cfg_p_flip_agents", lambda s: str(s.iloc[0]) != "()"),
        c=("c_whole", "mean"), cov=("cov_explored", "mean"), acc=("acc_committed", "mean")).reset_index()
    run["setting"] = np.where(run.cfg_N == 1, "single robot",
                              np.where(run.hetero, "team, one unreliable", "team, clean"))

    print("=== certified fraction by setting, acquisition and generator (20 worlds) ===")
    print(run.pivot_table(index=["setting", "cfg_planner"], columns="cfg_generator", values="c").round(3).to_string())

    print("\n=== rrt - sectors, paired by world (Holm within setting) ===")
    for setting, d in run.groupby("setting"):
        piv = d.pivot_table(index=["cfg_seed", "cfg_planner"], columns="cfg_generator", values="c").reset_index()
        rows, labels = [], []
        for pl, g in piv.groupby("cfg_planner"):
            x = (g["rrt"] - g["sectors"]).dropna().values
            rows.append(x); labels.append(pl)
        ps = holm([paired_t(x) for x in rows])
        print(f"  {setting}:")
        for pl, x, pv in zip(labels, rows, ps):
            m, h = ci95(x)
            print(f"    {pl:12s}: {m:+.3f} +- {h:.3f}  holm p = {pv:.3f}{'  *' if pv < 0.05 else ''}")

    print("\n=== the heterogeneity result under each generator (certified - coverage mask) ===")
    for gen, d in run[run.setting == "team, one unreliable"].groupby("cfg_generator"):
        piv = d.pivot_table(index="cfg_seed", columns="cfg_planner", values="c")
        if "certified" in piv and "entropy_g" in piv:
            m, h = ci95((piv["certified"] - piv["entropy_g"]).dropna().values)
            print(f"  {gen:8s}: {m:+.3f} +- {h:.3f}")
