"""E15: certificate replay on real 2D LiDAR logs (Freiburg SLAM benchmark, CARMEN format).

The trajectory is given (no planning). Poses are refined by scan-to-map correlative matching
starting from odometry; a pseudo ground truth is the occupancy grid built from all refined scans
(occupied: >=3 endpoint hits and hits > 0.3 * passes; free: >=3 passes and <=1 hit; else unknown and
excluded). The block protocol of the paper (train/cal time split, pooled weighted calibration,
alpha, e-value fusion) is then applied to the same scans, single robot and as three contiguous
'robots' fused on a complete graph, and evaluated on the uniform law over explored known cells.
"""
import sys, os, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from scipy import ndimage
from certmap.world import World
from certmap.features import LocalFeatures
from certmap.model import BayesLinearMap
from certmap.conformal import calibrate_agent, fuse, gate_from_samples, AgentCert
from certmap.planner import GridGraph

RES, RMAX, FREE_SP = 0.25, 8.0, 0.25


def parse_clf(path):
    poses, scans = [], []
    for line in open(path):
        if not line.startswith("FLASER"): continue
        t = line.split(); n = int(t[1]); r = np.array(t[2:2 + n], float); x, y, th = map(float, t[2 + n:5 + n])
        poses.append((x, y, th)); scans.append(r)
    return np.array(poses), scans


def endpoints(pose, r, fov=np.pi):
    n = len(r); ang = pose[2] + np.linspace(-fov / 2, fov / 2, n)
    ok = (r > 0.1) & (r < RMAX)
    return np.stack([pose[0] + r * np.cos(ang), pose[1] + r * np.sin(ang)], 1), ok, ang


def refine_poses(poses, scans, step=2, search=(0.15, 0.05, np.deg2rad(3), np.deg2rad(1)), res=0.1):
    """Incremental correlative scan-to-map matching on a hit-count grid (odometry increments as prior)."""
    W = 120.0; nx = int(W / res); origin = -W / 2
    hits = np.zeros((nx, nx), np.float32)
    out = [np.array(poses[0], float)]
    dxs = np.arange(-search[0], search[0] + 1e-9, search[1]); dths = np.arange(-search[2], search[2] + 1e-9, search[3])
    cur = np.array(poses[0], float)
    def score(p, r):
        e, ok, _ = endpoints(p, r); e = e[ok]
        ix = ((e[:, 0] - origin) / res).astype(int); iy = ((e[:, 1] - origin) / res).astype(int)
        m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < nx)
        return float(np.minimum(hits[iy[m], ix[m]], 3).sum())
    def add(p, r):
        e, ok, _ = endpoints(p, r); e = e[ok]
        ix = ((e[:, 0] - origin) / res).astype(int); iy = ((e[:, 1] - origin) / res).astype(int)
        m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < nx); np.add.at(hits, (iy[m], ix[m]), 1.0)
    add(cur, scans[0]); idxs = [0]
    for k in range(step, len(scans), step):
        # odometry increment in the previous frame
        p0, p1 = poses[k - step], poses[k]
        d = p1[:2] - p0[:2]; c, s = np.cos(cur[2] - p0[2]), np.sin(cur[2] - p0[2])
        pred = np.array([cur[0] + c * d[0] - s * d[1], cur[1] + s * d[0] + c * d[1], cur[2] + (p1[2] - p0[2])])
        best, bs = pred, -1
        for dth in dths:
            for dx in dxs:
                for dy in dxs:
                    p = pred + np.array([dx, dy, dth]); sc = score(p, scans[k])
                    if sc > bs: best, bs = p, sc
        cur = best; add(cur, scans[k]); out.append(cur.copy()); idxs.append(k)
    return np.array(out), idxs


def pseudo_gt(poses, scans, res=RES):
    e_all = []; 
    for p, r in zip(poses, scans):
        e, ok, _ = endpoints(p, r); e_all.append(e[ok])
    E = np.concatenate(e_all); lo = E.min(0) - 2; hi = E.max(0) + 2
    nx, ny = int(np.ceil((hi[0] - lo[0]) / res)), int(np.ceil((hi[1] - lo[1]) / res))
    hits = np.zeros((ny, nx)); passes = np.zeros((ny, nx))
    for p, r in zip(poses, scans):
        e, ok, ang = endpoints(p, r)
        for j in np.where(ok)[0]:
            ts = np.arange(FREE_SP, r[j] - FREE_SP / 2, FREE_SP)
            fx = p[0] + ts * np.cos(ang[j]) - lo[0]; fy = p[1] + ts * np.sin(ang[j]) - lo[1]
            ix = (fx / res).astype(int); iy = (fy / res).astype(int); m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            np.add.at(passes, (iy[m], ix[m]), 1)
        ix = ((e[ok, 0] - lo[0]) / res).astype(int); iy = ((e[ok, 1] - lo[1]) / res).astype(int); m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.add.at(hits, (iy[m], ix[m]), 1)
    occ = (hits >= 3) & (hits > 0.3 * passes)
    free = (passes >= 3) & (hits <= 1)
    known = occ | free
    return occ, known, lo, (nx, ny)


def samples_from_scan(p, r):
    e, ok, ang = endpoints(p, r); Xs = [e[ok]]; ys = [np.ones(ok.sum())]
    for j in np.where(ok)[0]:
        ts = np.arange(FREE_SP, r[j] - FREE_SP / 2, FREE_SP)
        if len(ts): Xs.append(np.stack([p[0] + ts * np.cos(ang[j]), p[1] + ts * np.sin(ang[j])], 1)); ys.append(-np.ones(len(ts)))
    return np.concatenate(Xs), np.concatenate(ys)


_CACHE = {}
def prepare(name):
    if name in _CACHE: return _CACHE[name]
    poses, scans = parse_clf(os.path.join(os.path.dirname(__file__), "..", "data", "real", name))
    t0 = time.time(); P, idxs = refine_poses(poses, scans); scans = [scans[i] for i in idxs]
    print(f"{name}: {len(P)} scans refined in {time.time()-t0:.0f}s; path {np.sum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1)):.0f} m", flush=True)
    occ, known, lo, (nx, ny) = pseudo_gt(P, scans); P = P.copy(); P[:, :2] -= lo
    _CACHE[name] = (P, scans, occ, known); return _CACHE[name]


def run(name, alpha=0.05, s_safe=0.7, r_gate=0.5, block_len_m=5.0, n_agents=(1, 3), weighted=True, seed=0):
    rng = np.random.default_rng(seed)
    P, scans, occ, known = prepare(name)
    world = World(occ, RES); world.obs = known.ravel() & world.obs      # evaluate on known observable cells only
    feats = LocalFeatures(world.W, world.H, 0.5, 0.4); PhiQ = feats.phi(world.Q)
    # blocks by travelled distance
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]; blk = (d // block_len_m).astype(int)
    rows = []
    for N in n_agents:
        seg = np.array_split(np.arange(len(P)), N)
        agents = []
        for i in range(N):
            model = BayesLinearMap(feats, 0.3, 1.0); Xtr, Xc, yc, fold = [], [], [], []
            for b in np.unique(blk[seg[i]]):
                ks = seg[i][blk[seg[i]] == b]; ntr = int(np.ceil(0.6 * len(ks)))
                for j, k in enumerate(ks):
                    X, y = samples_from_scan(P[k], scans[k])
                    if j < ntr: model.add(X, y); Xtr.append(X)
                    else: Xc.append(X); yc.append(y); fold.append(rng.random(len(X)) < 0.5)
            Xtr = np.concatenate(Xtr); Xc = np.concatenate(Xc); yc = np.concatenate(yc); fold = np.concatenate(fold)
            mu, sd = model.predict(PhiQ); gate = gate_from_samples(Xtr, world.nx, world.ny, RES, r_gate)
            muc, sdc = model.predict(feats.phi(Xc)); sc, rc = ~fold, fold
            beta = s_safe * alpha / N
            cert = calibrate_agent(mu, sd, gate, Xtr, Xc[sc], yc[sc], muc[sc], sdc[sc], Xc[rc], yc[rc], muc[rc], sdc[rc], beta=beta, weighted=weighted,
                                   nx=world.nx, ny=world.ny, res=RES, kde_bw=0.5)
            agents.append(cert)
        F = fuse(agents, alpha); obs = world.obs; lab = world.labels[obs]
        explored = ~F.nodata[obs]; inset = F.Gamma[np.arange(world.M), (world.labels > 0).astype(int)][obs]
        committed = F.committed[obs]; correct = committed & (F.label[obs] == lab)
        rows.append(dict(log=name, N=N, alpha=alpha, weighted=weighted, known_cells=int(obs.sum()), explored=float(explored.mean()),
                         cov_explored=float(inset[explored].mean()), c_explored=float(committed[explored].mean()), c_whole=float(committed.mean()),
                         acc=float(correct.sum() / max(committed.sum(), 1)), empty=float(F.empty[obs][explored].mean()),
                         undecided=float((F.undecided[obs] & explored).mean() / max(explored.mean(), 1e-9)),
                         p_hat=float(np.nanmean([a.p_hat for a in agents])), q=float(np.mean([a.q_scalar for a in agents])), eps=float(np.mean([a.eps for a in agents]))))
    return rows, (occ, known, P)


if __name__ == "__main__":
    out = []
    for name in ["intel.clf", "fr079.clf"]:
        for alpha in [0.05, 0.1, 0.2]:
            for w in [True, False]:
                rows, _ = run(name, alpha=alpha, weighted=w); out += rows
                print(pd.DataFrame(rows).round(3).to_string(index=False))
    df = pd.DataFrame(out); df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "e15_real_lidar.csv"), index=False)
    pd.set_option("display.width", 250); print(df.round(3).to_string(index=False))
