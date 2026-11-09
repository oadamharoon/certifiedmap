"""E34: the heterogeneity mechanism on real LiDAR logs (Intel, FR079).

E32/E33 are simulation. With no hardware, the honest real-data instantiation is to split one real log into
contiguous segments, treat each as a robot, degrade one of them, and ask what the team's shared state says about
the cells that robot observed. The trajectory is fixed, so this measures the *mechanism* behind the planning
result rather than the planning gain itself:

  a coverage mask treats an unreliable robot's observed cells as done;
  the certificate leaves exactly those cells undecided, and Corollary 2 says how many: the certified share of a
  robot's well-observed cells is min(1, beta_i / theta_i) with theta_i its contradiction rate.

Degradations of robot 3 (the others are untouched):
  flip   10 % of its labels flipped;
  drift  its scans are written at a pose that random-walks (0.05 m per block), rays still cast from the true pose.
Reported per condition: each robot's own certified share of its well-observed cells against the Corollary 2
forecast, the fused certified fraction, the error rate on committed cells, and the share of the map that a
coverage mask would call explored while the certificate leaves it undecided.
"""
import sys, os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from certmap.world import World
from certmap.features import LocalFeatures
from certmap.model import BayesLinearMap
from certmap.conformal import calibrate_agent, fuse, gate_from_samples
from e15_real_lidar import prepare, samples_from_scan, RES

ALPHA, SAFE, RGATE, BLOCK = 0.05, 0.7, 0.5, 5.0
CONDS = ["clean", "flip", "drift"]


def degrade(X, y, cond, i_agent, n_agents, blk_idx, rng, bias):
    """Robot `n_agents-1` is the unreliable one; the others are untouched."""
    if i_agent != n_agents - 1 or cond == "clean":
        return X, y
    if cond == "flip":
        return X, np.where(rng.random(len(y)) < 0.10, -y, y)
    return X + bias[None, :], y          # drift: samples written at a drifted pose


def run(name, N=3, seed=0):
    P, scans, occ, known = prepare(name)
    world = World(occ, RES); world.obs = known.ravel() & world.obs
    feats = LocalFeatures(world.W, world.H, 0.5, 0.4); PhiQ = feats.phi(world.Q)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    blk = (d // BLOCK).astype(int)
    seg = np.array_split(np.arange(len(P)), N)
    rows = []
    for cond in CONDS:
        rng = np.random.default_rng(seed)
        agents, gates = [], []
        for i in range(N):
            model = BayesLinearMap(feats, 0.3, 1.0); Xtr, Xc, yc, fold = [], [], [], []
            bias = np.zeros(2); last_b = None
            for b in np.unique(blk[seg[i]]):
                if cond == "drift" and i == N - 1:
                    bias = bias + rng.normal(0, 0.05, 2)        # random walk on the pose estimate, per block
                ks = seg[i][blk[seg[i]] == b]; ntr = int(np.ceil(0.6 * len(ks)))
                for j, k in enumerate(ks):
                    X, y = samples_from_scan(P[k], scans[k])
                    X, y = degrade(X, y, cond, i, N, b, rng, bias)
                    if j < ntr:
                        model.add(X, y); Xtr.append(X)
                    else:
                        Xc.append(X); yc.append(y); fold.append(rng.random(len(X)) < 0.5)
            Xtr = np.concatenate(Xtr); Xc = np.concatenate(Xc); yc = np.concatenate(yc); fold = np.concatenate(fold)
            mu, sd = model.predict(PhiQ); gate = gate_from_samples(Xtr, world.nx, world.ny, RES, RGATE)
            muc, sdc = model.predict(feats.phi(Xc)); sc, rc = ~fold, fold
            cert = calibrate_agent(mu, sd, gate, Xtr, Xc[sc], yc[sc], muc[sc], sdc[sc], Xc[rc], yc[rc], muc[rc], sdc[rc],
                                   beta=SAFE * ALPHA / N, weighted=True, nx=world.nx, ny=world.ny, res=RES, kde_bw=0.5)
            agents.append(cert); gates.append(gate)
            # this robot's own certificate, before fusion: the set it could defend alone at its own level beta
            own_G = cert.e < 1.0 / (SAFE * ALPHA / N)
            own_committed = own_G.sum(1) == 1
            # Corollary 2 is about *well-observed* cells, sigma at the posterior floor (e17_graded uses 1.02 x min)
            gated = gate & world.obs
            smin = cert.sigma[gated].min() if gated.any() else np.inf
            own = gated & np.isfinite(cert.q) & (cert.sigma <= 1.02 * smin)
            # theta is the weighted contradiction rate of *well-observed* recalibration labels, which is the
            # quantity Corollary 2 is stated in and which calibrate_agent already computes (AgentCert.p_hat).
            # An unweighted rate over all recalibration points is a different (larger) number and makes the
            # forecast spuriously conservative.
            theta = float(cert.p_hat)
            rows.append(dict(log=name, cond=cond, agent=i, unreliable=(i == N - 1 and cond != "clean"),
                             theta=round(theta, 4), beta=round(SAFE * ALPHA / N, 4),
                             forecast=round(min(1.0, (SAFE * ALPHA / N) / theta) if theta > 0 else 1.0, 3),
                             own_certified=round(float(own_committed[own].mean()), 3) if own.any() else np.nan,
                             own_cells=int(own.sum())))
        F = fuse(agents, ALPHA); obs = world.obs; lab = world.labels[obs]
        mask_explored = np.any(np.stack(gates), 0)[obs]          # what a coverage mask calls done
        committed = F.committed[obs]; correct = committed & (F.label[obs] == lab)
        rows.append(dict(log=name, cond=cond, agent="fused", unreliable=None,
                         certified=round(float(committed.mean()), 3),
                         acc_committed=round(float(correct.sum() / max(committed.sum(), 1)), 4),
                         mask_explored=round(float(mask_explored.mean()), 3),
                         mask_but_undecided=round(float((mask_explored & ~committed).mean()), 3)))
    return rows


if __name__ == "__main__":
    out = []
    for name in ["intel.clf", "fr079.clf"]:
        t0 = time.time(); out += run(name); print(f"{name.split(chr(46))[0]} done in {time.time()-t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(out)
    path = os.path.join(os.path.dirname(__file__), "..", "results", "e34_real_hetero.csv")
    df.to_csv(path, index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
