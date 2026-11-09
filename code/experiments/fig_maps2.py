"""Maps figure (occupancy-grid visual language). Row 1: single robot on a world with an unreliable region
(theta_R = 0.4): ground truth | marginal certificate (adopted) | reliability-conditional certificate (ours), both under
the entropy planner | reliability-conditional certificate under the certified planner (ours). Row 2: three robots, third
unreliable (flip 0.10): ground truth | certified on fused U (ours) | entropy on coverage mask | entropy on exchanged plans."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap
from certmap.sim import Simulation, Config
FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figs")
# categories: 0 no data, 1 committed free, 2 committed occupied, 3 undecided (with data), 4 empty, 5 wrong commitment
CM = ListedColormap(["#e6e6e6", "#ffffff", "#111111", "#f6a800", "#8e24aa", "#e53935"])
NAMES = ["no data", "certified free", "certified occupied", "undecided", "empty", "wrong commitment"]
plt.rcParams.update({"font.size": 8})

def image(sim, a):
    w = sim.world; F = a.fused; obs = w.obs
    cat = np.zeros(w.M, int)
    cat[F.committed & (F.label < 0)] = 1; cat[F.committed & (F.label > 0)] = 2
    cat[F.undecided & ~F.nodata] = 3; cat[F.empty] = 4
    cat[F.committed & (F.label != w.labels)] = 5
    cat[~obs] = 2 * (w.labels[~obs] > 0)  # obstacle interiors drawn as occupied for readability
    cat[~obs & ~w.occ.ravel()] = 0
    return cat.reshape(w.ny, w.nx)

def draw(ax, sim, img, title, trajs=True, region=True):
    w = sim.world
    ax.imshow(img, cmap=CM, vmin=0, vmax=5, origin="lower", extent=[0, w.W, 0, w.H], interpolation="nearest")
    if region and sim.region is not None:
        x0, y0, x1, y1 = sim.region
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, hatch="///", edgecolor="#1976d2", lw=0.8))
    if trajs:
        for k, a in enumerate(sim.agents):
            tr = np.array(a.traj); unreliable = bool(sim.cfg.p_flip_agents) and sim.cfg.p_flip_agents[k] > 0
            ax.plot(tr[:, 0], tr[:, 1], "--" if unreliable else "-", lw=0.9, color=f"C{k}")
            ax.plot(tr[0, 0], tr[0, 1], "o", color=f"C{k}", ms=3.5, mec="k", mew=0.4)
    ax.set_title(title, fontsize=8, pad=2); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.6)

def gt_image(sim):
    w = sim.world; cat = np.where(w.occ.ravel(), 2, 1); return cat.reshape(w.ny, w.nx)

seed1 = int(sys.argv[1]) if len(sys.argv) > 1 else 2
seed2 = int(sys.argv[2]) if len(sys.argv) > 2 else 14
fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.35))
stats = []
# row 1
r1 = [("marginal, entropy route", dict(planner="entropy", mondrian=False)),
      ("classes, entropy route", dict(planner="entropy", mondrian=True)),
      ("classes, certified route", dict(planner="certified_m", mondrian=True))]
sims = []
for title, kw in r1:
    sim = Simulation(Config(seed=seed1, N=1, budget=100.0, sigma_r=0.05, p_flip_region=0.4, save_maps=True, **kw)); sim.run(); sims.append((title, sim))
draw(axes[0, 0], sims[0][1], gt_image(sims[0][1]), "ground truth", trajs=False)
for ax, (title, sim) in zip(axes[0, 1:], sims):
    a = sim.agents[0]; r = pd.DataFrame(sim.log if hasattr(sim, "log") else []).iloc[-1] if False else None
    draw(ax, sim, image(sim, a), title)
    F = a.fused; w = sim.world; obs = w.obs; x0, y0, x1, y1 = sim.region; Q = w.Q; ins = (Q[:, 0] >= x0) & (Q[:, 0] <= x1) & (Q[:, 1] >= y0) & (Q[:, 1] <= y1)
    stats.append("%s: c=%.2f c_in=%.2f wrong_in=%d wrong_out=%d" % (title, F.committed[obs].mean(), F.committed[ins & obs].mean(), (F.committed & (F.label != w.labels) & ins & obs).sum(), (F.committed & (F.label != w.labels) & ~ins & obs).sum()))
# row 2
r2 = [("certified, fused $\\mathcal{U}$", "certified"), ("entropy, coverage mask", "entropy_g"), ("entropy, exchanged plans", "entropy")]
sims2 = []
for title, pl in r2:
    sim = Simulation(Config(seed=seed2, N=3, topology="mesh", planner=pl, budget=100.0 / 3, sigma_r=0.05, p_flip_agents=[0.0, 0.0, 0.10], save_maps=True)); sim.run(); sims2.append((title, sim))
draw(axes[1, 0], sims2[0][1], gt_image(sims2[0][1]), "ground truth", trajs=False)
for ax, (title, sim) in zip(axes[1, 1:], sims2):
    draw(ax, sim, image(sim, sim.agents[0]), title)
    F = sim.agents[0].fused; obs = sim.world.obs
    stats.append("%s: c=%.2f undecided=%.2f nodata=%.2f" % (title, F.committed[obs].mean(), (F.undecided & ~F.nodata)[obs].mean(), F.nodata[obs].mean()))
handles = [Patch(facecolor=CM(i), edgecolor="k", lw=0.3, label=n) for i, n in enumerate(NAMES)] + \
          [plt.Line2D([], [], color="C0", lw=1, label="robot path"), plt.Line2D([], [], color="C2", lw=1, ls="--", label="unreliable robot")]
fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=6.5, columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(pad=0.2, rect=(0, 0.06, 1, 0.99)); plt.savefig(os.path.join(FIGS, "maps.pdf"))
print("\n".join(stats)); print("saved maps.pdf")
