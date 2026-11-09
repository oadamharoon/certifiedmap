"""Figure: Intel Research Lab log -- pseudo ground truth and the fused certified map of three robots."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from e15_real_lidar import prepare, run, RES
from certmap.viz import CMAP, NAMES, partition_image
from certmap.world import World
from certmap.features import LocalFeatures
from certmap.model import BayesLinearMap
from certmap.conformal import calibrate_agent, fuse, gate_from_samples
import e15_real_lidar as E
FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps" if "acm" in sys.argv else "paper", "figs")
plt.rcParams.update({"font.size": 8})
P, scans, occ, known = prepare("intel.clf")
world = World(occ, RES); obs = known.ravel() & world.obs
feats = LocalFeatures(world.W, world.H, 0.5, 0.4); PhiQ = feats.phi(world.Q)
rng = np.random.default_rng(0)
d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]; blk = (d // 5.0).astype(int)
def certify(N, weighted=True):
    seg = np.array_split(np.arange(len(P)), N); certs = []
    for i in range(N):
        model = BayesLinearMap(feats, 0.3, 1.0); Xtr, Xc, yc, fold = [], [], [], []
        for b in np.unique(blk[seg[i]]):
            ks = seg[i][blk[seg[i]] == b]; ntr = int(np.ceil(0.6 * len(ks)))
            for j, k in enumerate(ks):
                X, y = E.samples_from_scan(P[k], scans[k])
                if j < ntr: model.add(X, y); Xtr.append(X)
                else: Xc.append(X); yc.append(y); fold.append(rng.random(len(X)) < 0.5)
        Xtr = np.concatenate(Xtr); Xc = np.concatenate(Xc); yc = np.concatenate(yc); fold = np.concatenate(fold)
        mu, sd = model.predict(PhiQ); gate = gate_from_samples(Xtr, world.nx, world.ny, RES, 0.5)
        muc, sdc = model.predict(feats.phi(Xc)); sc, rc = ~fold, fold
        certs.append(calibrate_agent(mu, sd, gate, Xtr, Xc[sc], yc[sc], muc[sc], sdc[sc], Xc[rc], yc[rc], muc[rc], sdc[rc], beta=0.7 * 0.05 / N, weighted=weighted, nx=world.nx, ny=world.ny, res=RES, kde_bw=0.5))
    return fuse(certs, 0.05), seg
F1, _ = certify(1); F3, seg3 = certify(3); F0, _ = certify(1, weighted=False)
from matplotlib.colors import ListedColormap
CM = ListedColormap(["#e6e6e6", "#ffffff", "#111111", "#f6a800", "#8e24aa", "#e53935"])
NAMES2 = ["no data", "certified free", "certified occupied", "undecided", "empty", "wrong commitment"]
def image(F):
    cat = np.zeros(world.M, int)
    cat[F.committed & (F.label < 0)] = 1; cat[F.committed & (F.label > 0)] = 2
    cat[F.undecided & ~F.nodata] = 3; cat[F.empty] = 4
    cat[F.committed & (F.label != world.labels)] = 5
    cat[~obs] = 0
    return cat.reshape(world.ny, world.nx)
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.05))
gt = np.where(occ, 2, np.where(known, 1, 0))
axes[0].imshow(gt, origin="lower", cmap=CM, vmin=0, vmax=5, interpolation="nearest")
axes[0].plot(P[:, 0] / RES, P[:, 1] / RES, "-", lw=0.5, color="C0")
stats = []
for ax, F, title, segs in [(axes[1], F0, "sensor-law certificate", [np.arange(len(P))]), (axes[2], F1, "operator-law certificate", [np.arange(len(P))]), (axes[3], F3, "three robots, fused", seg3)]:
    ax.imshow(image(F), cmap=CM, vmin=0, vmax=5, origin="lower", interpolation="nearest")
    for i, sg in enumerate(segs):
        ax.plot(P[sg, 0] / RES, P[sg, 1] / RES, "-", lw=0.5, color=f"C{i}" if len(segs) > 1 else "C0")
    cov = F.Gamma[np.arange(world.M), (world.labels > 0).astype(int)][obs][~F.nodata[obs]].mean(); wrong = (F.committed & (F.label != world.labels) & obs).sum(); stats.append(f"{title}: c={F.committed[obs].mean():.3f} cov={cov:.3f} wrong={wrong} empty={F.empty[obs].mean():.3f}")
axes[0].set_title("pseudo ground truth", fontsize=8, pad=2); axes[1].set_title("sensor-law certificate", fontsize=8, pad=2); axes[2].set_title("operator-law certificate", fontsize=8, pad=2); axes[3].set_title("three robots, fused", fontsize=8, pad=2)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linewidth(0.6)
fig.legend(handles=[Patch(facecolor=CM(i), edgecolor="k", lw=0.3, label=n) for i, n in enumerate(NAMES2)] + [plt.Line2D([], [], color="C0", lw=1, label="robot path")], loc="lower center", ncol=7, frameon=False, fontsize=6.5, columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.5, 0.0))
plt.tight_layout(pad=0.3, rect=(0, 0.11, 1, 0.94)); plt.savefig(os.path.join(FIGS, "real_intel.pdf")); print("real_intel.pdf (single log; the paper uses fig_real2.py)"); print("\n".join(stats))
