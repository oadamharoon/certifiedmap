"""Mechanism figure for the ICCPS draft: what a coverage map and a certified map disagree about when one robot
in the team cannot be trusted.

One world, three robots, the third flipping 10 % of its labels. Panels:
  1 ground truth with the three trajectories;
  2 what a coverage map says: every cell any robot observed, i.e. the set a mask planner treats as finished;
  3 what the certificate says: committed free / committed occupied / observed-but-undecided -- the undecided
    band is exactly the unreliable robot's territory, and it is what the mask cannot see;
  4-5 the final maps of a mask planner and of a planner reading the fused undecided set, at equal budget.
The claim of the paper is the difference between panels 2 and 3, and its consequence is panels 4 and 5.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from certmap.sim import Simulation, Config

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps", "figs"); os.makedirs(FIGS, exist_ok=True)
# the palette, category order and names of the paper's map figures (experiments/fig_maps2.py)
CM = ListedColormap(["#e6e6e6", "#ffffff", "#111111", "#f6a800", "#8e24aa", "#e53935", "#9ecae1"])
NAMES = ["no data", "certified free", "certified occupied", "undecided", "empty", "wrong commitment", "observed"]
plt.rcParams.update({"font.size": 8})


def cat_map(sim, a, coverage_only=False):
    w = sim.world; F = a.fused
    cat = np.zeros(w.M, int)
    if coverage_only:
        seen = np.any(np.stack([b.cert.gate for b in sim.agents]), 0)
        cat[seen] = 6                      # a coverage map records observation, not a label
    else:
        cat[F.committed & (F.label < 0)] = 1
        cat[F.committed & (F.label > 0)] = 2
        cat[F.undecided & ~F.nodata] = 3
        cat[F.empty] = 4
        cat[F.committed & (F.label != w.labels)] = 5
    cat[~w.obs] = 2 * (w.labels[~w.obs] > 0)
    cat[~w.obs & ~w.occ.ravel()] = 0
    return cat.reshape(w.ny, w.nx)


def draw(ax, sim, img, title, trajs=False):
    w = sim.world
    ax.imshow(img, cmap=CM, vmin=0, vmax=6, origin="lower", extent=[0, w.W, 0, w.H], interpolation="nearest")
    if trajs:
        for k, a in enumerate(sim.agents):
            tr = np.array(a.traj); unreliable = bool(sim.cfg.p_flip_agents) and sim.cfg.p_flip_agents[k] > 0
            ax.plot(tr[:, 0], tr[:, 1], "--" if unreliable else "-", lw=0.9, color=f"C{k}")
            ax.plot(tr[0, 0], tr[0, 1], "o", color=f"C{k}", ms=3.5, mec="k", mew=0.4)
    ax.set_title(title, fontsize=8, pad=2); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    cfg = dict(seed=seed, N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120, save_maps=True,
               p_flip_agents=(0.0, 0.0, 0.10))
    sims = {}
    for pl in ["entropy_g", "certified"]:
        s = Simulation(Config(**cfg, planner=pl)); s.run(); sims[pl] = s
    gt = sims["certified"]; w = gt.world
    truth = np.where(w.occ.ravel(), 2, 1).reshape(w.ny, w.nx)

    mask = sims["entropy_g"]
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.60))
    draw(ax[0], gt, truth, "ground truth, three robots", trajs=True)
    draw(ax[1], mask, cat_map(mask, mask.agents[0], coverage_only=True), "what the coverage map shows")
    draw(ax[2], mask, cat_map(mask, mask.agents[0]), "what the certificate shows")
    draw(ax[3], gt, cat_map(gt, gt.agents[0]), "planning on the certificate")
    for a, s_, lab in [(ax[1], mask, "all observed"), (ax[2], mask, None), (ax[3], gt, None)]:
        u = float(s_.agents[0].fused.undecided[s_.world.obs].mean())
        a.text(0.015, 0.02, lab if lab else f"{100*u:.0f}% undecided", transform=a.transAxes,
               fontsize=6.5, ha="left", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0))
    fig.legend(handles=[Patch(facecolor=c, edgecolor="k", lw=0.3, label=l) for c, l in zip(CM.colors, NAMES)],
               loc="lower center", ncol=7, fontsize=6.5, frameon=False, columnspacing=1.0,
               handletextpad=0.5, bbox_to_anchor=(0.5, 0.006))
    plt.tight_layout(pad=0.3, rect=(0, 0.105, 1, 1))
    plt.savefig(os.path.join(FIGS, "unequal.pdf"))
    print("unequal.pdf", {k: round(float(s.agents[0].fused.committed[s.world.obs].mean()), 3) for k, s in sims.items()})
