"""Figure: does the margin survive a sparser or irregular communication graph, and where does it come from?

Top: the three graphs at N=5, with the unreliable robot filled. The complete graph and the ring are both
vertex-transitive, so one placement covers each; only the star distinguishes a leaf from the hub. E40 also runs
the complete graph with the fault at index 0 as a control that moving the fault is not a start-pose effect; it
duplicates the complete-graph bar, so it is reported in the text rather than plotted.
Bottom: certified state minus coverage-map planner, mean over 20 worlds with 95% CI.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from runner import load
from analysis import ci95

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps", "figs")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8})
COL = 3.34

df, runs = load("e40")
last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid")[["c_whole"]].mean()
D = {}
for rid, r in last.iterrows():
    c = runs[rid]["config"]
    place = "leaf" if c["p_flip_agents"][-1] > 0 else "hub"
    D.setdefault((place, c["topology"], c["planner"]), {})[c["seed"]] = r.c_whole

ORDER = [("leaf", "mesh", "complete"), ("leaf", "ring", "ring"), ("leaf", "star", "star, leaf"),
         ("hub", "star", "star, hub")]
vals, errs, labs = [], [], []
for place, topo, lab in ORDER:
    a, b = D[(place, topo, "certified")], D[(place, topo, "entropy_g")]
    s = sorted(set(a) & set(b)); d = np.array([a[k] - b[k] for k in s])
    m, h = ci95(d); vals.append(m); errs.append(h); labs.append(lab)

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(COL, 1.68), gridspec_kw={"height_ratios": [1, 2.2]})

# --- graph schematics, one above each bar -----------------------------------
def nodes(n=5, r=1.0):
    t = np.pi / 2 + np.arange(n) * 2 * np.pi / n
    return np.c_[r * np.cos(t), r * np.sin(t)]

def draw(ax, kind, bad, x0):
    P = nodes() * 0.40 + np.array([x0, 0.0])
    if kind == "complete":
        E = [(i, j) for i in range(5) for j in range(i + 1, 5)]
    elif kind == "ring":
        E = [(i, (i + 1) % 5) for i in range(5)]
    else:
        E = [(0, j) for j in range(1, 5)]
    for i, j in E:
        ax.plot(*zip(P[i], P[j]), "-", lw=0.5, color="0.65", zorder=1)
    ok = [i for i in range(5) if i != bad]
    ax.scatter(P[ok, 0], P[ok, 1], s=14, c="white", edgecolors="0.25", lw=0.7, zorder=2)
    ax.scatter(P[bad, 0], P[bad, 1], s=20, c="#e53935", edgecolors="k", lw=0.5, zorder=3)

for k, (kind, bad) in enumerate([("complete", 4), ("ring", 4), ("star", 4), ("star", 0)]):
    draw(ax0, kind, bad, k)
ax0.set_xlim(-0.5, 3.5); ax0.set_ylim(-0.55, 0.55); ax0.axis("off")

# --- margins, one bar under each graph --------------------------------------
x = np.arange(len(vals))
ax1.bar(x, vals, 0.55, yerr=errs, color="#4477aa", edgecolor="k", lw=0.4,
        error_kw=dict(ecolor="0.2", lw=0.8, capsize=2.5))
ax1.set_xticks(x); ax1.set_xticklabels(labs, fontsize=7.5)
ax1.set_xlim(-0.5, 3.5)
ax1.set_ylabel("margin")
ax1.grid(axis="y", alpha=0.3); ax1.set_axisbelow(True)
ax1.set_ylim(0, max(np.array(vals) + np.array(errs)) * 1.12)
plt.tight_layout(pad=0.3, h_pad=0.1, rect=(0.015, 0, 1, 1))
plt.savefig(os.path.join(FIGS, "topology.pdf")); print("topology.pdf", [f"{v:+.3f}" for v in vals])
