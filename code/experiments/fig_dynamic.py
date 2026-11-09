"""Changing-map figure: what each traversal rule costs and delivers.

Left: realised miss rate per entered cell against the level alpha, by guard and hazard density/speed -- the
validity check. Right: the safety-progress plane, miss rate against time-averaged certified area, so the cost of
each guarantee is visible at a glance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from runner import load
from analysis import ci95
from e26_analysis import per_run

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps", "figs"); os.makedirs(FIGS, exist_ok=True)
STYLE = {"none": ("no guard", "#999999", "o"), "inflate": ("obstacle inflation", "#cc6677", "s"),
         "cp_track": ("conformal hazard tracking", "#ddcc77", "^"), "prob": ("decaying log-odds grid", "#88ccee", "v"),
         "fresh": ("seen-last-scan heuristic", "#44aa99", "D"), "certified": ("persistence certificate", "#4477aa", "*"),
         "oracle": ("oracle map", "#000000", "x")}
SETTINGS = [(5, 1.0, "5 hazards\n1 m/block"), (10, 1.0, "10 hazards\n1 m/block"),
            (10, 3.0, "10 hazards\n3 m/block"), (20, 1.0, "20 hazards\n1 m/block")]

if __name__ == "__main__":
    df, _ = load("e26")
    df["cfg_hazard_motion"] = df.get("cfg_hazard_motion", "walk")
    last = per_run(df)
    std = (last.cfg_guard_alpha < 0) & (last.cfg_prob_decay == 0.9) & (last.cfg_alpha == 0.05) & (last.cfg_gate_window == 0)
    d = last[(last.cfg_planner == "entropy") & std]

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    w = 0.11
    for j, (g, (lab, col, mk)) in enumerate(STYLE.items()):
        xs, ys, es = [], [], []
        for i, (nh, st, _) in enumerate(SETTINGS):
            s = d[(d.cfg_guard == g) & (d.cfg_n_hazards == nh) & (d.cfg_hazard_step == st)]
            if s.empty: continue
            m, h = ci95(s.miss_cell.values); xs.append(i + (j - 3) * w); ys.append(m); es.append(h)
        ax[0].errorbar(xs, ys, yerr=es, fmt=mk, ms=4, color=col, capsize=1.5, lw=1, label=lab)
    ax[0].axhline(0.05, color="k", ls="--", lw=0.8)
    ax[0].text(3.3, 0.052, r"level $\alpha$", fontsize=7, ha="right")
    ax[0].set_xticks(range(len(SETTINGS))); ax[0].set_xticklabels([s[2] for s in SETTINGS], fontsize=7)
    ax[0].set_ylabel("misses per cell entered", fontsize=8); ax[0].tick_params(labelsize=7); ax[0].set_yscale("log")

    for g, (lab, col, mk) in STYLE.items():
        s = d[(d.cfg_guard == g) & (d.cfg_n_hazards == 10)]
        if s.empty: continue
        ax[1].scatter(s.c_time.mean(), s.miss_cell.mean(), s=38, color=col, marker=mk, label=lab, zorder=3)
        ax[1].annotate(lab.split("\n")[0], (s.c_time.mean(), s.miss_cell.mean()), fontsize=6.5,
                       xytext=(4, 3), textcoords="offset points")
    ax[1].axhline(0.05, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("certified area over time (progress)", fontsize=8)
    ax[1].set_ylabel("misses per cell entered", fontsize=8); ax[1].tick_params(labelsize=7)
    ax[1].set_title("safer and faster is down and right", fontsize=7.5)
    ax[0].legend(fontsize=6.2, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    plt.tight_layout(pad=0.4, rect=(0, 0, 1, 0.88))
    plt.savefig(os.path.join(FIGS, "dynamic.pdf")); print("dynamic.pdf")
