"""Heterogeneity figure: what the certified state is worth when observers are unequal.

Left: the share of the map left explored-but-uncertified by the coverage-mask planner and by the certified
state, per condition (what the team must come back for). Right: the paired difference in certified fraction
with 95 % CIs, and the same difference under the RRT-exploration candidate generator where available.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from runner import load
from analysis import ci95

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps", "figs"); os.makedirs(FIGS, exist_ok=True)
ORDER = [("clean", "all robots\nreliable"), ("flip", "one robot flips\n10% of labels"), ("registration", "one robot misregisters\n(0.6 m)"),
         ("drift", "one robot drifts\n(0.05 m/block)"), ("mixed", "two robots, two\ndifferent faults"),
         ("n5_one", "five robots,\none unreliable"), ("n5_two", "five robots,\ntwo unreliable")]


def cond_of(c):
    pf = tuple(c.get("p_flip_agents") or ()); se = tuple(c.get("sigma_e_agents") or ()); dr = tuple(c.get("drift_agents") or ())
    if c.get("start_seed", -1) >= 0 or c["N"] == 2: return None
    if c["N"] == 5:
        if dr: return None
        k = sum(1 for x in pf if x > 0)
        return {1: "n5_one", 2: "n5_two"}.get(k)
    if dr and pf: return None          # drift+flip is a separate condition, reported in the text
    if dr: return "drift" if max(dr) == 0.05 else None    # the rate the table reports; 0.02/0.1/0.2 are the sweep
    if pf and se: return "mixed"
    if pf: return "flip"
    if se: return "registration" if max(se) > 0.4 else None
    if c["N"] == 3: return "clean"
    return None


def collect():
    out = {}
    for exp in ["e32", "e33"]:
        df, runs = load(exp)
        last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid")[["c_whole", "undecided_whole"]].mean()
        for rid, r in last.iterrows():
            c = runs[rid]["config"]; k = cond_of(c)
            if k and c["planner"] in ("certified", "entropy_g"):
                out.setdefault((k, c["planner"]), []).append((c["seed"], r.c_whole, r.undecided_whole))
    return out


if __name__ == "__main__":
    D = collect()
    fig, ax = plt.subplots(1, 1, figsize=(3.34, 2.5)); ax = [ax, None]
    labels = [l for k, l in ORDER if (k, "certified") in D]
    keys = [k for k, l in ORDER if (k, "certified") in D]
    y = np.arange(len(keys))

    unc_m = [100 * (1 - np.mean([v[1] for v in D[(k, "entropy_g")]])) for k in keys]
    unc_c = [100 * (1 - np.mean([v[1] for v in D[(k, "certified")]])) for k in keys]
    ax[0].barh(y + 0.2, unc_m, 0.38, color="#bbbbbb", edgecolor="k", lw=0.4)
    ax[0].barh(y - 0.2, unc_c, 0.38, color="#4477aa", edgecolor="k", lw=0.4)
    for i, (a, b) in enumerate(zip(unc_m, unc_c)):
        ax[0].text(max(a, b) + 0.4, i, f"-{100*(1-b/a):.0f}%", va="center", fontsize=7)
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=7); ax[0].invert_yaxis()
    ax[0].set_xlabel("map left uncertified (%)", fontsize=8)
    ax[0].tick_params(labelsize=7)
    fig.legend(handles=[Patch(facecolor="#bbbbbb", edgecolor="k", lw=0.4, label="coverage map"),
                        Patch(facecolor="#4477aa", edgecolor="k", lw=0.4, label="certified state")],
               loc="lower center", ncol=2, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    for sp in ("top", "right"):
        ax[0].spines[sp].set_visible(False)
    plt.tight_layout(pad=0.4, rect=(0, 0.1, 1, 1))
    plt.savefig(os.path.join(FIGS, "hetero.pdf")); print("hetero.pdf", list(zip(keys, [round(u, 1) for u in unc_c])))
