"""Changing-map figure in the style of the paper's map figures: one world, three traversal rules.

Hazards move over the static map; the robot's path is drawn over the certified map it ends with, and every cell
it entered while occupied is marked. Panels: no guard, obstacle inflation, the persistence certificate. The
numbers under the panels are the realised misses per entered cell and the certified area over time.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap
from certmap.sim import Simulation, Config

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps", "figs"); os.makedirs(FIGS, exist_ok=True)
CM = ListedColormap(["#e6e6e6", "#ffffff", "#111111", "#f6a800", "#8e24aa", "#e53935"])
NAMES = ["no data", "certified free", "certified occupied", "undecided", "empty", "wrong commitment"]
plt.rcParams.update({"font.size": 8})
GUARDS = [("none", "no guard"), ("inflate", "obstacle inflation"), ("certified", "traversal certificate")]


def image(sim, a):
    w = sim.world; F = a.fused
    cat = np.zeros(w.M, int)
    cat[F.committed & (F.label < 0)] = 1; cat[F.committed & (F.label > 0)] = 2
    cat[F.undecided & ~F.nodata] = 3; cat[F.empty] = 4
    cat[F.committed & (F.label != w.labels)] = 5
    cat[~w.obs] = 2 * (w.labels[~w.obs] > 0)
    cat[~w.obs & ~sim.static_occ.ravel()] = 0
    return cat.reshape(w.ny, w.nx)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    base = dict(seed=seed, N=1, budget=300.0, sigma_r=0.05, n_hazards=10, hazard_size=3.0,
                hazards_move_after_plan=True, K=120, planner="entropy", hazard_step=1.0, save_maps=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
    for ax, (g, title) in zip(axes, GUARDS):
        sim = Simulation(Config(**base, guard=g)); sim.run(); a = sim.agents[0]; w = sim.world
        ax.imshow(image(sim, a), cmap=CM, vmin=0, vmax=5, origin="lower", extent=[0, w.W, 0, w.H], interpolation="nearest")
        for c in sim.hazards:                                  # where the hazards ended up
            ax.add_patch(Rectangle((c[0] - 1.5, c[1] - 1.5), 3, 3, fill=False, edgecolor="#1976d2", lw=0.8, ls="--"))
        tr = np.array(a.traj); ax.plot(tr[:, 0], tr[:, 1], "-", lw=0.9, color="C0")
        ax.plot(tr[0, 0], tr[0, 1], "o", color="C0", ms=3.5, mec="k", mew=0.4)
        mx = np.array(getattr(a, "miss_xy", [])).reshape(-1, 2)
        if len(mx):
            ax.plot(mx[:, 0], mx[:, 1], "x", color="#e53935", ms=3.5, mew=0.9)
        r = sim.log[-1]
        tc = sum(r[f"tc{b}"] for b in range(6)); to = sum(r[f"to{b}"] for b in range(6))
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel(f"{to/max(tc,1):.3f} misses per cell entered", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.6)
    handles = [Patch(facecolor=c, edgecolor="k", lw=0.3, label=l) for c, l in zip(CM.colors, NAMES)]
    handles += [plt.Line2D([], [], color="#e53935", marker="x", ls="", label="entered while occupied"),
                plt.Line2D([], [], color="#1976d2", ls=":", label="hazard, final position")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(pad=0.3, rect=(0, 0.19, 1, 1))
    plt.savefig(os.path.join(FIGS, "guard_maps.pdf")); print("guard_maps.pdf")
