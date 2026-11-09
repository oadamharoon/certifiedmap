"""Real LiDAR logs, two environments: pseudo ground truth, the certificate on the law the robot sampled, the
certificate on the operator's query law, and three contiguous segments treated as robots fusing e-values."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from e15_real_lidar import prepare, RES
from certmap.world import World
from certmap.features import LocalFeatures
from certmap.model import BayesLinearMap
from certmap.conformal import calibrate_agent, fuse, gate_from_samples
import e15_real_lidar as E
from matplotlib.colors import ListedColormap

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps", "figs"); os.makedirs(FIGS, exist_ok=True)
CM = ListedColormap(["#e6e6e6", "#ffffff", "#111111", "#f6a800", "#8e24aa", "#e53935"])
NAMES = ["no data", "certified free", "certified occupied", "undecided", "empty", "wrong commitment"]
plt.rcParams.update({"font.size": 8})
ALPHA, SAFE, RGATE = 0.05, 0.7, 0.5
TITLES = ["pseudo ground truth", "certificate on the sampled law", "certificate on the query law",
          "three robots, fused"]


def build(name):
    P, scans, occ, known = prepare(name)
    world = World(occ, RES); obs = known.ravel() & world.obs
    feats = LocalFeatures(world.W, world.H, 0.5, 0.4); PhiQ = feats.phi(world.Q)
    rng = np.random.default_rng(0)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]; blk = (d // 5.0).astype(int)

    def certify(N, weighted):
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
            mu, sd = model.predict(PhiQ); gate = gate_from_samples(Xtr, world.nx, world.ny, RES, RGATE)
            muc, sdc = model.predict(feats.phi(Xc)); sc, rc = ~fold, fold
            certs.append(calibrate_agent(mu, sd, gate, Xtr, Xc[sc], yc[sc], muc[sc], sdc[sc], Xc[rc], yc[rc],
                                         muc[rc], sdc[rc], beta=SAFE * ALPHA / N, weighted=weighted,
                                         nx=world.nx, ny=world.ny, res=RES, kde_bw=0.5))
        return fuse(certs, ALPHA), np.array_split(np.arange(len(P)), N)

    F_unw, _ = certify(1, False); F_w, _ = certify(1, True); F_3, seg3 = certify(3, True)
    return P, occ, known, world, obs, [F_unw, F_w, F_3], seg3


def crop_box(known, pad=3):
    """Tight box around the mapped area, so panels are not padded with empty grid."""
    ys, xs = np.where(known)
    return (max(xs.min() - pad, 0), min(xs.max() + pad, known.shape[1] - 1),
            max(ys.min() - pad, 0), min(ys.max() + pad, known.shape[0] - 1))


def panel(ax, world, obs, F):
    cat = np.zeros(world.M, int)
    cat[F.committed & (F.label < 0)] = 1; cat[F.committed & (F.label > 0)] = 2
    cat[F.undecided & ~F.nodata] = 3; cat[F.empty] = 4
    cat[F.committed & (F.label != world.labels)] = 5
    cat[~obs] = 0
    ax.imshow(cat.reshape(world.ny, world.nx), cmap=CM, vmin=0, vmax=5, origin="lower", interpolation="nearest")
    return


if __name__ == "__main__":
    logs = ["intel.clf", "fr079.clf"]
    built = [build(n) for n in logs]
    boxes = [crop_box(b[2]) for b in built]
    ratios = [(bx[3] - bx[2]) / (bx[1] - bx[0]) for bx in boxes]   # aspect of the mapped area, not of the grid
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 0.72 + 6.6 * sum(ratios) / 4),
                             gridspec_kw=dict(height_ratios=ratios))
    for r, (name, (P, occ, known, world, obs, Fs, seg3), bx) in enumerate(zip(logs, built, boxes)):
        gt = np.where(occ, 2, np.where(known, 1, 0))
        axes[r, 0].imshow(gt, origin="lower", cmap=CM, vmin=0, vmax=5, interpolation="nearest")
        axes[r, 0].plot(P[:, 0] / RES, P[:, 1] / RES, "-", lw=0.5, color="C0")
        for c, F in enumerate(Fs):
            panel(axes[r, c + 1], world, obs, F)
            segs = seg3 if c == 2 else [np.arange(len(P))]
            for i, sg in enumerate(segs):
                axes[r, c + 1].plot(P[sg, 0] / RES, P[sg, 1] / RES, "-", lw=0.5,
                                    color=f"C{i}" if len(segs) > 1 else "C0")
        axes[r, 0].set_ylabel(name.split(".")[0], fontsize=8)
        for ax in axes[r]:
            ax.set_xlim(bx[0], bx[1]); ax.set_ylim(bx[2], bx[3])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_linewidth(0.6)
        print(name, [round(float(F.committed[obs].mean()), 3) for F in Fs], flush=True)
    for c, t in enumerate(TITLES):
        axes[0, c].set_title(t, fontsize=8, pad=2)
    fig.legend(handles=[Patch(facecolor=CM(i), edgecolor="k", lw=0.3, label=n) for i, n in enumerate(NAMES)]
               + [plt.Line2D([], [], color="C0", lw=1, label="robot path")],
               loc="lower center", ncol=7, frameon=False, fontsize=6.5, columnspacing=0.8,
               handletextpad=0.4, bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(pad=0.2, h_pad=0.35, rect=(0, 0.065, 1, 1))
    plt.savefig(os.path.join(FIGS, "real.pdf")); print("real.pdf (two rows)")
