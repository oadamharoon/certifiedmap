"""Candidate segments, acquisition functions, and team allocation.

All adaptive planners share: the same traversability estimate, the same
candidate generator, the same predicted observation model, and the same
sequential-greedy team allocation. They differ only in the acquisition
function scored on the predicted observation set S of a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import dijkstra
from scipy.special import ndtr

from .conformal import AgentCert, gate_from_samples

SQRT2 = np.sqrt(2.0)


def _phibar(z):
    return 1.0 - ndtr(z)


# --------------------------------------------------------------------------
# grid graph and paths
# --------------------------------------------------------------------------
class GridGraph:
    def __init__(self, nx: int, ny: int, res: float):
        self.nx, self.ny, self.res = nx, ny, res
        self.M = nx * ny
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
        ix, iy = ix.ravel(), iy.ravel()
        rows, cols, w = [], [], []
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            ok = (ix + dx >= 0) & (ix + dx < nx) & (iy + dy >= 0) & (iy + dy < ny)
            a = iy[ok] * nx + ix[ok]
            b = (iy[ok] + dy) * nx + (ix[ok] + dx)
            rows.append(a); cols.append(b); w.append(np.full(ok.sum(), np.hypot(dx, dy) * res))
        self.rows, self.cols, self.w = np.concatenate(rows), np.concatenate(cols), np.concatenate(w)

    def graph(self, traversable: np.ndarray) -> sparse.csr_matrix:
        ok = traversable[self.rows] & traversable[self.cols]
        g = sparse.coo_matrix((self.w[ok], (self.rows[ok], self.cols[ok])), shape=(self.M, self.M))
        return (g + g.T).tocsr()

    def shortest(self, traversable: np.ndarray, src: int, limit: float):
        g = self.graph(traversable)
        dist, pred = dijkstra(g, indices=src, return_predecessors=True, limit=limit)
        return dist, pred

    @staticmethod
    def path(pred: np.ndarray, src: int, dst: int) -> list[int]:
        p, cur = [dst], dst
        while cur != src and cur >= 0:
            cur = pred[cur]
            p.append(cur)
        return p[::-1] if cur == src else []

    def xy(self, idx):
        idx = np.asarray(idx)
        return np.stack([(idx % self.nx + 0.5) * self.res, (idx // self.nx + 0.5) * self.res], 1)


def traversable_from_belief(mu: np.ndarray, gate: np.ndarray, nx: int, ny: int, thr: float = 0.0,
                            inflate: int = 1) -> np.ndarray:
    """Optimistic traversability: believed-occupied (gated & mu>thr) cells are blocked, inflated."""
    blocked = (gate & (mu > thr)).reshape(ny, nx)
    if inflate > 0:
        blocked = ndimage.binary_dilation(blocked, iterations=inflate)
    return ~blocked.ravel()


def poses_along(path_xy: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a polyline at ``spacing`` and attach headings (x, y, theta)."""
    if len(path_xy) < 2:
        return np.zeros((0, 3))
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    L = cum[-1]
    ts = np.arange(spacing, L + 1e-9, spacing)
    if len(ts) == 0:
        return np.zeros((0, 3))
    xs = np.interp(ts, cum, path_xy[:, 0]); ys = np.interp(ts, cum, path_xy[:, 1])
    xp = np.interp(ts - spacing / 2, cum, path_xy[:, 0]); yp = np.interp(ts - spacing / 2, cum, path_xy[:, 1])
    th = np.arctan2(ys - yp, xs - xp)
    return np.stack([xs, ys, th], 1)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    path_idx: list[int]                          # full path to the target (scored)
    exec_idx: list[int]                          # first seg_len metres (executed)
    path_xy: np.ndarray
    length: float                                # full path length (cost)
    poses: np.ndarray                            # poses at which observations are predicted
    S: np.ndarray = field(default=None)          # predicted sample locations
    cells: np.ndarray = field(default=None)      # affected query cells (flat idx)


def _truncate_idx(gg: GridGraph, idx, seg_len: float):
    xy = gg.xy(idx)
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    keep = cum <= seg_len + 1e-9
    return list(np.array(idx)[keep]), xy, float(cum[-1])


def _candidate(gg: GridGraph, idx, seg_len: float, pose_spacing: float, max_poses: int):
    exec_idx, xy, L = _truncate_idx(gg, idx, seg_len)
    if L < 0.5 or len(exec_idx) < 2:
        return None
    poses = poses_along(xy, pose_spacing)
    if len(poses) > max_poses:
        poses = poses[np.linspace(0, len(poses) - 1, max_poses).astype(int)]
    return Candidate(list(idx), exec_idx, xy, L, poses)


def frontier_candidates(gg: GridGraph, gate, mu, trav, nx, ny, dist, pred, src, n_front, seg_len,
                        pose_spacing, max_poses, d_min: float = 3.0, thr: float = 0.0):
    """Up to n_front nearest reachable frontier-cluster targets, scored over their full path."""
    free = (gate & (mu <= thr)).reshape(ny, nx)
    unknown = (~gate).reshape(ny, nx)
    front = free & ndimage.binary_dilation(unknown, structure=np.ones((3, 3))) & trav.reshape(ny, nx)
    lab, n = ndimage.label(front, structure=np.ones((3, 3)))
    targets = []
    for k in range(1, n + 1):
        idx = np.where((lab == k).ravel())[0]
        if len(idx) < 3:
            continue
        d = dist[idx]
        ok = np.isfinite(d) & (d >= d_min)
        if ok.any():
            j = idx[ok][np.argmin(d[ok])]
            targets.append((dist[j], j))
    targets.sort()
    out = []
    for _, dst in targets[:n_front]:
        idx = gg.path(pred, src, dst)
        if len(idx) >= 2:
            c = _candidate(gg, idx, seg_len, pose_spacing, max_poses)
            if c is not None:
                out.append(c)
    return out


def make_candidates(gg: GridGraph, trav: np.ndarray, src: int, rng: np.random.Generator,
                    n_cand: int, d_min: float, d_max: float, seg_len: float, pose_spacing: float,
                    max_poses: int = 30):
    dist, pred = gg.shortest(trav, src, limit=80.0)
    reach = np.where(np.isfinite(dist) & (dist >= d_min) & (dist <= d_max))[0]
    if len(reach) == 0:
        return [], dist, pred
    xy = gg.xy(reach); p0 = gg.xy([src])[0]
    ang = np.arctan2(xy[:, 1] - p0[1], xy[:, 0] - p0[0])
    bins = np.floor((ang + np.pi) / (2 * np.pi) * n_cand).astype(int) % n_cand
    chosen = []
    for b in range(n_cand):
        pool = reach[bins == b]
        if len(pool):
            chosen.append(pool[rng.integers(len(pool))])
    while len(chosen) < n_cand and len(reach) > len(chosen):
        c = reach[rng.integers(len(reach))]
        if c not in chosen:
            chosen.append(c)
    cands = []
    for dst in chosen:
        idx = gg.path(pred, src, dst)
        if len(idx) >= 2:
            c = _candidate(gg, idx, seg_len, pose_spacing, max_poses)
            if c is not None:
                cands.append(c)
    return cands, dist, pred


# --------------------------------------------------------------------------
# acquisition functions
# --------------------------------------------------------------------------
def rrt_candidates(gg: GridGraph, trav: np.ndarray, src: int, rng: np.random.Generator, gate: np.ndarray,
                   nx: int, ny: int, dist: np.ndarray, pred: np.ndarray, n_cand: int, seg_len: float,
                   pose_spacing: float, max_poses: int = 30, n_nodes: int = 400, step: float = 2.0,
                   claimed: np.ndarray | None = None, claim_r: float = 3.0):
    """Candidate segments from RRT-detected frontiers (Umari & Mukhopadhyay, IROS 2017).

    A tree is grown in the believed-free space by sampling the workspace and extending towards each sample; a
    node that lands on an unknown cell is a detected frontier point. As in the original, the tree *detects*
    frontiers and the ordinary grid planner *navigates* to them, so the candidate is the shortest believed-free
    path to the detected point, not the jagged tree branch. Targets are taken nearest-first with the team's
    claimed targets deconflicted. This is a different candidate generator, not a different acquisition: whatever
    acquisition the caller uses is then scored on these candidates.
    """
    W, H = nx * gg.res, ny * gg.res
    unknown = (~gate).reshape(ny, nx)
    p0 = gg.xy([src])[0]
    nodes = [p0]
    detected = []
    for _ in range(n_nodes):
        q = np.array([rng.uniform(0, W), rng.uniform(0, H)])
        a = np.asarray(nodes)
        i = int(np.argmin(np.linalg.norm(a - q[None, :], axis=1)))
        v = q - nodes[i]; n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        new = nodes[i] + v / n * min(step, n)
        if not (0 <= new[0] < W and 0 <= new[1] < H):
            continue
        m = max(int(np.ceil(np.linalg.norm(new - nodes[i]) / gg.res)), 1)
        pts = nodes[i][None, :] + np.linspace(0, 1, m + 1)[:, None] * (new - nodes[i])[None, :]
        idx = np.clip((pts[:, 1] / gg.res).astype(int), 0, ny - 1) * nx + np.clip((pts[:, 0] / gg.res).astype(int), 0, nx - 1)
        if not trav[idx].all():
            continue                              # the edge leaves the believed-free space
        nodes.append(new)
        iy, ix = min(int(new[1] / gg.res), ny - 1), min(int(new[0] / gg.res), nx - 1)
        if unknown[iy, ix]:
            detected.append(iy * nx + ix)
    if not detected:
        return []
    detected = [c for c in dict.fromkeys(detected) if np.isfinite(dist[c])]
    detected.sort(key=lambda c: dist[c])          # nearest detected frontier first, as the baseline assigns
    cands = []
    for c in detected:
        if claimed is not None and len(claimed) and np.min(np.linalg.norm(claimed - gg.xy([c])[0][None, :], axis=1)) < claim_r:
            continue
        idx = gg.path(pred, src, c)
        if len(idx) < 2:
            continue
        cd = _candidate(gg, idx, seg_len, pose_spacing, max_poses)
        if cd is not None:
            cands.append(cd)
        if len(cands) >= n_cand:
            break
    return cands


def commit_prob(mu, sig_now, sig_after, q):
    """P(cell committed after S) under the preposterior mu_S ~ N(mu, sig_now^2 - sig_after^2).

    With score |y - mu|/sigma and labels +-1, label y is rejected iff |y - mu| > q sigma.
    Exactly one label is rejected (a commitment, as opposed to an empty set) iff
        |mu_S| >= |q sigma_S - 1| =: t_S.
    sig_* are predictive stds (incl. noise); the preposterior variance of the mean
    equals the reduction in latent variance, which is the same difference.
    """
    t = np.abs(q * sig_after - 1.0)
    delta = np.sqrt(np.maximum(sig_now ** 2 - sig_after ** 2, 0.0))
    P = np.zeros_like(mu)
    pos = delta > 1e-9
    P[pos] = _phibar((t[pos] - mu[pos]) / delta[pos]) + _phibar((t[pos] + mu[pos]) / delta[pos])
    P[~pos] = (np.abs(mu[~pos]) >= t[~pos]).astype(float)
    return np.clip(P, 0.0, 1.0)


class Acquisition:
    """Scores a candidate for one agent. ``kind`` in {certified, entropy, lse, random}."""

    def __init__(self, kind: str, z_lse: float = 1.96, score: str = "residual"):
        self.kind = kind
        self.z = z_lse
        self.score_kind = score

    def score(self, cand: Candidate, model, feats, PhiQ, cert: AgentCert, undecided: np.ndarray,
              others_commit: np.ndarray | None, gate_r: float, nx: int, ny: int, res: float,
              rng: np.random.Generator):
        """Returns (gain, per-cell commit-probability array on cand.cells or None)."""
        if self.kind == "random":
            return rng.random(), None
        if cand.S is None or len(cand.S) == 0 or len(cand.cells) == 0:
            return 0.0, None
        cells = cand.cells
        PhiS = feats.phi(cand.S)
        var_after = model.var_after(PhiS, PhiQ[cells])
        sig_after = np.sqrt(np.maximum(var_after, 0.0) + model.sn2)
        sig_now = cert.sigma[cells]
        mu = cert.mu[cells]
        if self.kind == "entropy":
            return float(np.sum(np.log(np.maximum(sig_now, 1e-9) / np.maximum(sig_after, 1e-9)))), None
        if self.kind in ("entropy_u", "entropy_g"):
            # controls: entropy restricted to the fused undecided set ('entropy_u') or to cells no
            # neighbour has gated ('entropy_g', a coverage-mask baseline); 'undecided' carries the mask
            u = undecided[cells]
            return float(np.sum(u * np.log(np.maximum(sig_now, 1e-9) / np.maximum(sig_after, 1e-9)))), None
        if self.kind == "lse":
            amb_now = np.abs(mu) < self.z * sig_now
            P = commit_prob_lse(mu, sig_now, sig_after, self.z)
            return float(np.sum(P[amb_now])), None
        if self.kind in ("certified", "certified_r", "certified_m"):
            # credit only cells that the predicted samples will actually support
            # (within the gate radius of S): cells never sampled cannot be certified
            g_after = gate_from_samples(cand.S, nx, ny, res, gate_r)[cells]
            q_eff = cert.q_scalar
            if self.kind == "certified_m" and cert.group is not None:
                # Mondrian certificate: each cell's own threshold where calibrated, else its reliability class's
                qg = cert.q_group[cert.group[cells]]
                q_eff = np.where(np.isfinite(cert.q[cells]), cert.q[cells], qg)
                q_eff = np.where(np.isfinite(q_eff), q_eff, 1e6)
            if self.score_kind == "margin":
                P = commit_prob_lse(mu, sig_now, sig_after, q_eff) * g_after
            else:
                P = commit_prob(mu, sig_now, sig_after, q_eff) * g_after
            if self.kind in ("certified_r", "certified_m") and cert.theta_map is not None:
                # reliability-aware: with local contradiction rate theta, the attainable |mu| is at most 1 - 2 theta,
                # so a cell whose commitment threshold t_S exceeds it cannot be certified however often it is sampled
                t_S = np.abs(cert.q_scalar * sig_after - 1.0)
                P = P * ((1.0 - 2.0 * cert.theta_map[cells]) >= t_S)
            P = np.maximum(P, 0.0)
            u = undecided[cells]
            disc = 1.0 if others_commit is None else (1.0 - others_commit[cells])
            return float(np.sum(P * u * disc)), P * u
        raise ValueError(self.kind)


def commit_prob_lse(mu, sig_now, sig_after, z):
    """P(|mu_S| >= z sigma_S): the level-set (straddle) analogue with a fixed z."""
    t = z * sig_after
    delta = np.sqrt(np.maximum(sig_now ** 2 - sig_after ** 2, 0.0))
    P = np.zeros_like(mu)
    pos = delta > 1e-9
    P[pos] = _phibar((t[pos] - mu[pos]) / delta[pos]) + _phibar((t[pos] + mu[pos]) / delta[pos])
    P[~pos] = (np.abs(mu[~pos]) >= t[~pos]).astype(float)
    return np.clip(P, 0, 1)


def affected_cells(S: np.ndarray, nx: int, ny: int, res: float, radius: float) -> np.ndarray:
    """Flat indices of cells within ``radius`` of any predicted sample (where variance changes)."""
    ix = np.clip(np.floor(S[:, 0] / res).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(S[:, 1] / res).astype(int), 0, ny - 1)
    m = np.zeros((ny, nx), bool); m[iy, ix] = True
    m = ndimage.binary_dilation(m, iterations=int(np.ceil(radius / res)))
    return np.where(m.ravel())[0]


# --------------------------------------------------------------------------
# frontier and lawnmower (non-acquisition planners)
# --------------------------------------------------------------------------
def frontier_target(gate: np.ndarray, mu: np.ndarray, trav: np.ndarray, nx: int, ny: int,
                    dist: np.ndarray, d_min: float, thr: float = 0.0, unknown: np.ndarray | None = None,
                    claimed: list | None = None, claim_r: float = 3.0, res: float = 0.25) -> int | None:
    """Nearest reachable frontier cluster (believed-free gated cell adjacent to 'unknown').

    unknown  optional (M,) bool mask of cells counted as unexplored (default: not gated by this robot);
             a team passes the shared coverage mask (no neighbour gated) or the fused undecided set.
    claimed  flat indices of targets already chosen by teammates this block; clusters whose nearest
             cell lies within claim_r metres of a claimed target are skipped (target deconfliction).
    """
    free = (gate & (mu <= thr)).reshape(ny, nx)
    unk = (~gate if unknown is None else unknown).reshape(ny, nx)
    nb = ndimage.binary_dilation(unk, structure=np.ones((3, 3)))
    front = free & nb & trav.reshape(ny, nx)
    lab, n = ndimage.label(front, structure=np.ones((3, 3)))
    if n == 0:
        return None
    claimed_xy = [np.array([(c % nx + 0.5) * res, (c // nx + 0.5) * res]) for c in (claimed or [])]
    best, bestd = None, np.inf
    for k in range(1, n + 1):
        idx = np.where((lab == k).ravel())[0]
        d = dist[idx]
        ok = np.isfinite(d) & (d >= d_min)
        if ok.any() and len(idx) >= 3:
            j = idx[ok][np.argmin(d[ok])]
            xy = np.array([(j % nx + 0.5) * res, (j // nx + 0.5) * res])
            if any(np.linalg.norm(xy - c) < claim_r for c in claimed_xy):
                continue
            if dist[j] < bestd:
                best, bestd = j, dist[j]
    return best


def lawnmower_waypoints(W: float, H: float, strip_i: int, n_strips: int, lane_spacing: float,
                        margin: float = 1.0) -> np.ndarray:
    x0, x1 = strip_i * W / n_strips + margin, (strip_i + 1) * W / n_strips - margin
    ys = np.arange(margin, H - margin + 1e-9, lane_spacing)
    pts = []
    for k, y in enumerate(ys):
        pts += [(x0, y), (x1, y)] if k % 2 == 0 else [(x1, y), (x0, y)]
    return np.array(pts)
