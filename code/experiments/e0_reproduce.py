"""E0 (4.2): reproduce Mahajan et al. (2026) under their protocol on our simulator.

Prescribed trajectories (5 robots sweeping overlapping sub-regions), K=2 blocks per robot split
into training/calibration halves, exact GP (Matern-3/2) or our Bayesian linear map fitted on the
block-1 training half, scores on the block-1 calibration half, recalibration (a_i, eps_i) on the
block-2 calibration half, test set = block-2 training fold (their test law), alpha=0.2, gate r=1 m,
s=0.7, lambda=0, unweighted. Reports their metrics on their test law, and the same certificate
evaluated on the uniform law over the explored observable grid.
"""
import sys, os, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.linalg import cho_factor, cho_solve
from certmap.world import make_world
from certmap.sensor import Lidar, LidarSpec
from certmap.features import LocalFeatures
from certmap.model import BayesLinearMap
from certmap.conformal import ScoreSet
from certmap.planner import lawnmower_waypoints, poses_along, GridGraph, traversable_from_belief

ALPHA, S_SAFE, R_GATE = 0.2, 0.7, 1.0


class GPMatern:
    """Exact GP regression, Matern-3/2, unit signal variance, fixed length-scale and noise."""
    def __init__(self, ell=0.5, noise_var=0.3, max_n=3000, rng=None):
        self.ell, self.sn2, self.max_n, self.rng = ell, noise_var, max_n, rng
    def k(self, A, B):
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)) * (np.sqrt(3) / self.ell)
        return (1 + d) * np.exp(-d)
    def fit(self, X, y):
        if len(X) > self.max_n:
            idx = self.rng.choice(len(X), self.max_n, replace=False); X, y = X[idx], y[idx]
        self.X = X
        K = self.k(X, X) + self.sn2 * np.eye(len(X))
        self.L = cho_factor(K, lower=True, check_finite=False)
        self.alpha = cho_solve(self.L, y, check_finite=False)
    def predict(self, Xq, chunk=2000):
        mu, var = np.empty(len(Xq)), np.empty(len(Xq))
        for s0 in range(0, len(Xq), chunk):
            Ks = self.k(Xq[s0:s0 + chunk], self.X)
            mu[s0:s0 + chunk] = Ks @ self.alpha
            V = cho_solve(self.L, Ks.T, check_finite=False)
            var[s0:s0 + chunk] = 1.0 - np.einsum("ij,ji->i", Ks, V)
        return np.clip(mu, -1, 1), np.sqrt(np.maximum(var, 0) + self.sn2)


class BLMWrap:
    def __init__(self, feats, noise_var=0.3):
        self.m = BayesLinearMap(feats, noise_var, 1.0); self.f = feats
    def fit(self, X, y):
        self.m.add(X, y); self.m.solve()
    def predict(self, Xq):
        return self.m.predict(self.f.phi(Xq))


def prescribed_trajectory(world, region, rng, length=100.0, lane=4.0):
    """Lawnmower over a rectangular sub-region, routed around obstacles with the true map."""
    x0, x1, y0, y1 = region
    gg = GridGraph(world.nx, world.ny, world.res)
    trav = ~world.occ.ravel()
    from scipy import ndimage
    trav = ~ndimage.binary_dilation(world.occ, iterations=1).ravel()
    ys = np.arange(y0 + 1, y1 - 0.5, lane)
    wps = []
    for k, y in enumerate(ys):
        wps += [(x0 + 1, y), (x1 - 1, y)] if k % 2 == 0 else [(x1 - 1, y), (x0 + 1, y)]
    # start at the nearest traversable cell to the first waypoint
    idx = np.where(trav)[0]; xy = gg.xy(idx)
    cur = idx[np.argmin(np.linalg.norm(xy - np.array(wps[0]), axis=1))]
    path = [cur]
    for wp in wps[1:]:
        dst = idx[np.argmin(np.linalg.norm(xy - np.array(wp), axis=1))]
        dist, pred = gg.shortest(trav, cur, limit=200.0)
        if not np.isfinite(dist[dst]):
            continue
        seg = gg.path(pred, cur, dst)
        path += seg[1:]; cur = dst
    poses = poses_along(gg.xy(path), 0.5)
    L = np.cumsum(np.r_[0, np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)])
    return poses[L <= length]


def run(seed, model_kind, topo, noise_var=0.3):
    rng = np.random.default_rng(seed)
    w = make_world(seed, wall_thick=0.25)
    lidar = Lidar(LidarSpec(40, np.pi, 5.0, 0.02, 0.003, 0.04, 0.003, 0.25))
    feats = LocalFeatures(w.W, w.H, 0.5, 0.4)
    N = 5
    regions = [(0, 14, 0, 12), (10, 24, 0, 12), (18, 30, 0, 12), (0, 16, 8, 20), (14, 30, 8, 20)]
    nb = {i: list(range(N)) for i in range(N)} if topo == "mesh" else {i: sorted({i, (i - 1) % N, (i + 1) % N}) for i in range(N)}
    d = {i: len(nb[i]) for i in range(N)}
    # ---- data collection along prescribed trajectories, K=2 blocks x (train, cal) halves ----
    data = {}
    for i in range(N):
        poses = prescribed_trajectory(w, regions[i], rng)
        n = len(poses); q = n // 4
        folds = {"tr1": poses[:q], "cal1": poses[q:2 * q], "tr2": poses[2 * q:3 * q], "cal2": poses[3 * q:]}
        D = {}
        for name, ps in folds.items():
            Xs, ys = [], []
            for p in ps:
                X, y = lidar.scan(w.occ, w.res, p, rng); Xs.append(X); ys.append(y)
            D[name] = (np.concatenate(Xs), np.concatenate(ys))
        data[i] = D
    # ---- block-1 certificate per agent ----
    certs = {}
    test_pts = []
    for i in range(N):
        D = data[i]
        model = GPMatern(0.5, noise_var, 3000, rng) if model_kind == "gp" else BLMWrap(feats, noise_var)
        model.fit(*D["tr1"])
        Xc, yc = D["cal1"]; sub = rng.choice(len(Xc), min(2000, len(Xc)), replace=False); Xc, yc = Xc[sub], yc[sub]
        mc, sc = model.predict(Xc); ss = ScoreSet.build(np.abs(yc - mc) / sc)
        Xr, yr = D["cal2"]; sub = rng.choice(len(Xr), min(2000, len(Xr)), replace=False); Xr, yr = Xr[sub], yr[sub]
        mr, sr = model.predict(Xr); p_rec = ss.pvalue(np.abs(yr - mr) / sr, np.ones(len(Xr)))
        beta = S_SAFE * ALPHA / d[i]
        o = np.argsort(p_rec); cum = np.arange(1, len(o) + 1) / len(o); ok = cum <= beta
        if ok.any():
            j = np.where(ok)[0][-1]; a, eps = float(p_rec[o][j]), max(float(cum[j]), 1 / (len(o) + 1))
        else:
            a, eps = 0.0, 1.0
        tree = cKDTree(D["tr1"][0])
        Xt, yt = D["tr2"]; sub = rng.choice(len(Xt), min(2000, len(Xt)), replace=False); Xt, yt = Xt[sub], yt[sub]
        test_pts.append((Xt, yt))
        certs[i] = dict(model=model, ss=ss, a=a, eps=eps, tree=tree)

    def evals(i, Xq):
        c = certs[i]; mu, sd = c["model"].predict(Xq)
        gate = c["tree"].query(Xq)[0] < R_GATE
        q = c["ss"].threshold(c["a"], np.ones(len(Xq))) if c["a"] > 0 else np.full(len(Xq), np.inf)
        e = np.zeros((len(Xq), 2))
        for k, y in enumerate((-1.0, 1.0)):
            e[:, k] = (np.abs(y - mu) / sd > q) / c["eps"] * gate
        return e, gate

    rows = []
    Xall = np.concatenate([t[0] for t in test_pts]); yall = np.concatenate([t[1] for t in test_pts])
    yall_clean = w.labels[w.flat_of(Xall)]                    # clean label at the test location
    for i in range(N):
        # their test law: the union of agents' block-2 training folds (noisy labels as sensed)
        E = [evals(j, Xall) for j in nb[i]]
        ef = sum(e for e, _ in E) / d[i]; g_any = np.any([g for _, g in E], 0)
        G = ef < 1 / ALPHA; n = G.sum(1)
        cov_noisy = G[np.arange(len(Xall)), (yall > 0).astype(int)].mean()
        cov_clean = G[np.arange(len(Xall)), (yall_clean > 0).astype(int)].mean()
        # uniform law over the explored observable grid
        Eg = [evals(j, w.Q) for j in nb[i]]
        efg = sum(e for e, _ in Eg) / d[i]; gg_any = np.any([g for _, g in Eg], 0) & w.obs
        Gg = efg < 1 / ALPHA; ng = Gg.sum(1)
        covg = Gg[np.arange(w.M), (w.labels > 0).astype(int)][gg_any].mean()
        rows.append(dict(seed=seed, model=model_kind, topo=topo, agent=i, a=certs[i]["a"], eps=certs[i]["eps"],
                         cov_test=cov_noisy, cov_test_clean=cov_clean, classified_test=(n == 1).mean(),
                         nodata_test=(~g_any).mean(), uncertain_test=((n == 2) & g_any).mean(), empty_test=(n == 0).mean(),
                         acc_test=((G[:, 1] & ~G[:, 0]) == (yall_clean > 0))[n == 1].mean() if (n == 1).any() else np.nan,
                         cov_grid=covg, classified_grid=(ng == 1)[gg_any].mean(), empty_grid=(ng == 0)[gg_any].mean(),
                         uncertain_grid=(ng == 2)[gg_any].mean(), explored_grid=gg_any.mean() / w.obs.mean()))
    return rows


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(s, m, t) for s in range(5) for m in ["gp", "blm"] for t in ["mesh", "ring"]]
    t0 = time.time()
    with Pool(8) as pool:
        res = pool.starmap(run, jobs)
    df = pd.DataFrame([r for rr in res for r in rr])
    out = os.path.join(os.path.dirname(__file__), "..", "results", "e0_reproduce.csv"); df.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    g = df.groupby(["model", "topo"])[["cov_test", "cov_test_clean", "classified_test", "nodata_test", "uncertain_test", "empty_test", "acc_test",
                                       "cov_grid", "classified_grid", "uncertain_grid", "empty_grid", "explored_grid", "eps"]].mean()
    print(g.round(3).to_string()); print(f"{time.time()-t0:.0f}s")
