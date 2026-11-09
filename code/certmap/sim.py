"""Block-structured multi-robot simulation: certify -> plan -> execute -> repeat.

Protocol per block k (Assumption "open-loop blocks" of the paper):
  1. every agent solves its model, predicts on the query grid, builds its
     certificate (gate, weights, recalibration, e-values) and fuses its
     neighbourhood's e-values into a set-valued map;
  2. metrics are recorded on that map;
  3. every agent chooses its next segment (sequential greedy in rotating order);
  4. all agents execute their segments; the first ``train_frac`` of each
     segment's scans go to the model, the rest to the calibration pool
     (randomly split into a score fold and a recalibration fold).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import ndimage

from .world import make_world, sample_starts, World
from .sensor import Lidar, LidarSpec
from .features import LocalFeatures
from .model import BayesLinearMap
from .conformal import calibrate_agent, calibrate_agent_grouped, fuse, gate_from_samples, AgentCert, FusedMap, local_contradiction_map
from .planner import (GridGraph, traversable_from_belief, make_candidates, frontier_candidates, rrt_candidates, Acquisition,
                      affected_cells, frontier_target, lawnmower_waypoints, poses_along)


NB = 6   # staleness buckets of the persistence certificate: 1, 2-3, 4-7, 8-15, 16-31, >= 32 / never


@dataclass
class Config:
    """Everything that defines a run. Runs are cached by the hash of the *given* fields
    (experiments/runner.py), so defaults may be reordered or documented but never renamed
    or silently changed: that would orphan every cached result."""

    # ---- seeds ---------------------------------------------------------------
    seed: int = 0                   # world + start poses
    noise_seed: int = -1            # sensor noise + planner randomness (defaults to seed)
    start_seed: int = -1            # start poses resampled from this seed on the same world (defaults to world seed)

    # ---- world ---------------------------------------------------------------
    W: float = 30.0
    H: float = 20.0
    res: float = 0.25
    wall_thick: float = 0.25
    n_walls: int = 8
    n_boxes: int = 6
    world_file: str = ""            # optional .npy boolean occupancy grid (HouseExpo floor plan) instead of the procedural world

    # ---- team ----------------------------------------------------------------
    N: int = 1
    topology: str = "mesh"          # mesh | ring | star (who fuses e-values with whom)
    planner: str = "certified"      # certified | certified_r | certified_m | entropy | entropy_u | entropy_g | lse | frontier | frontier_g | frontier_u | lawnmower | random
    alloc: str = "greedy"           # greedy | joint (joint search over candidate pairs, N=2 only)
    generator: str = "sectors"      # candidate generator: sectors (bearing-binned shortest paths + frontier
                                    # targets) | rrt (RRT-exploration style tree, Umari & Mukhopadhyay 2017)
    lookahead: int = 1              # segments of receding-horizon lookahead (1 = myopic, the default everywhere)
    beam: int = 3                   # beam width of that search

    # ---- certificate (static; Sec. III-V of the paper) -----------------------
    alpha: float = 0.05             # target miscoverage of the certified map
    safety: float = 0.7             # budget share s: per-neighbourhood level beta_i = s*alpha/d_i
    gate_r: float = 0.5             # a cell is gated (certifiable) if a training sample lies within gate_r
    lam: float = 0.0                # attenuation of the e-value fusion (0 = plain product, as in Mahajan et al.)
    weighted: bool = True           # covariate-shift weights that re-target the certificate to the uniform law (Tibshirani et al. 2019)
    kde_bw: float = 0.5             # box width of the calibration-density histogram (2 cells)
    score: str = "residual"         # residual (Mahajan et al.) | margin (nested sets)
    eps_ucb: float = 0.0            # >0: Hoeffding-type upper confidence margin on the recalibrated miscoverage
    window: int = 0                 # certificate window in blocks (0 = pool all blocks); model, gate and folds use the last `window` blocks
    log_combined: bool = False      # also evaluate the exact block-combined certificate (Corollary 1, first-gated-block weights)

    # ---- reliability classes (Mondrian certificate; E20-E22) -----------------
    mondrian: bool = False          # group-conditional certificate by reliability class (taxonomy from the training fold)
    mondrian_tau: float = 0.15      # contradiction rate above which a cell belongs to the unreliable class

    # ---- sensing reliability (label noise; heterogeneous teams) --------------
    p_flip: float = 0.0             # sensor label-flip probability (label noise)
    p_flip_range: float = 0.0       # range-dependent label flips: p(r) = p_flip_range * (r/max_range)^2
    p_flip_agents: tuple = ()       # heterogeneous team: per-agent label-flip probabilities (overrides p_flip)
    p_flip_region: float = 0.0      # label-flip probability inside a random unreliable region (glass/clutter); 0 = none
    region_size: tuple = (10.0, 8.0)  # size (m) of that unreliable region, placed at random by the world seed
    range_agents: tuple = ()        # heterogeneous team: per-agent max range (m)
    beams_agents: tuple = ()        # heterogeneous team: per-agent beam count
    sigma_e_agents: tuple = ()      # heterogeneous team: per-agent pose position noise (m) -- per-scan registration error
    drift_agents: tuple = ()        # heterogeneous team: per-agent pose drift (m per block, random walk on the
                                    # estimated pose). Unlike sigma_e this accumulates and is correlated in time:
                                    # rays are cast from the true pose, samples are written at the drifted one.
    sigma_theta_agents: tuple = ()  # heterogeneous team: per-agent pose heading noise (rad)

    # ---- moving hazards (dynamic environment; E23-E31) -----------------------
    n_hazards: int = 0              # boxes that move over the static map each block
    hazard_size: float = 1.5        # hazard box side (m)
    hazard_step: float = 1.0        # hazard displacement per block (m)
    hazard_motion: str = "walk"     # walk: random step each block | goal: move toward a random free goal at constant speed, re-sampled on arrival
    hazard_clear: float = 0.5       # hazards keep their footprint this far from a robot's position (they do not run it over)
    hazards_move_after_plan: bool = False  # dynamic loop: certify -> plan -> hazards move -> execute (traversal is one lag ahead)

    # ---- traversal guard: which cells the planner may drive through ----------
    guard: str = "none"             # none | inflate | fresh | prob | cp_track | cp_track_oracle | certified
                                    # (staleness-indexed persistence certificate) | ltt (the same evidence, but one
                                    # threshold on a graded risk score calibrated by Learn-then-Test) | oracle
    trav_thr: float = 0.3           # believed-occupied iff gated and mu > trav_thr (traversability and predicted rays)
    trav_inflate: int = 0           # inflation (cells) of believed-occupied cells for traversability
    gate_window: int = 0            # lapsing support: a cell is gated only if a training sample within r lies in the last `gate_window` blocks (0 = never lapses)
    prob_decay: float = 0.9         # prob guard: per-block decay of the log-odds occupancy grid (forgetting)
    track_window: int = 20          # cp_track guard: blocks of prediction errors used to calibrate the region radius

    # ---- persistence certificate of the `certified` guard (Props A-C) --------
    guard_alpha: float = -1.0       # level of the persistence certificate (-1 = alpha)
    obs_r: float = 0.5              # a cell counts as observed in a block if a sample fell within obs_r of it (staleness)
    horizon_window: int = 20        # evidence blocks of traversal pairs used for the per-class miss rate
    horizon_rho: float = 1.0        # age decay of those weights (Barber et al. nonexchangeable conformal; 1 = equal weights)
    change_tau: float = 0.1         # local miss-rate threshold defining the dynamic (vs stable) region class
    aci_gamma: float = 0.05         # adaptive conformal inference step on the per-class inner level (0 = off)
    share_pairs: bool = True        # teams: the certificate pools the neighbourhood's traversal pairs and miss maps
    warmup: int = 10                # blocks under the uncertified committed-free-and-fresh rule that seed the calibration
    probe_after: int = 3            # after this many consecutive blocks with no admissible segment, one uncertified probe block (0 = never)
    ltt_delta: float = 0.1          # ltt guard: the certificate may be wrong about the admitted set with probability delta
    ltt_bins: int = 4               # ltt guard: risk-score bins of the local miss rate (x NB staleness buckets)

    # ---- motion / blocks -----------------------------------------------------
    K: int = 100                    # max number of blocks (safety cap; runs end at the budget)
    budget: float = 100.0           # path budget per agent (m); run ends when all agents reach it
    seg_len: float = 5.0            # metres per block
    pose_spacing: float = 0.5
    n_cand: int = 12                # local candidate targets
    n_front: int = 4                # frontier candidate targets (scored over the full path)
    front_dmin: float = -1.0        # frontier planner: minimum target distance (default 0.6*seg_len)
    plan_pose_spacing: float = 1.0  # pose spacing for predicted observations when scoring
    plan_max_poses: int = 30
    train_frac: float = 0.6         # first fraction of a segment's scans go to the model, the rest to calibration
    split4: bool = False            # sample-split certificate: calibration pool -> 4 disjoint folds
    w_clip: float = 0.0             # >0: cap the covariate-shift weights at this quantile (tail control)
    trust_tau: float = 0.0          # >0: detector baseline, drop a teammate's gate from the shared coverage
                                    # mask when its self-estimated label-error rate p_hat exceeds this
    denom_delta: float = 0.0        # >0: divide the e-value by the Lemma-1 Chernoff bound at this delta
                                    # (exact-alpha variant) instead of by the measured miscoverage
                                    # (claim | recal-score | recal | planning); e-value denominator = a
    lane_spacing: float = 4.0       # lawnmower lane spacing

    # ---- sensor --------------------------------------------------------------
    sigma_r: float = 0.02
    sigma_phi: float = 0.003
    sigma_e: float = 0.04
    sigma_theta: float = 0.003
    n_beams: int = 40
    max_range: float = 5.0
    free_spacing: float = 0.25      # one free-space sample per cell along each beam

    # ---- model ---------------------------------------------------------------
    noise_var: float = 0.3
    prior_var: float = 1.0
    feat_spacing: float = 0.5
    ell: float = 0.4

    # ---- planning observation model -----------------------------------------
    plan_beams: int = 20
    plan_free_spacing: float = 0.75

    # ---- logging -------------------------------------------------------------
    save_maps: bool = False         # keep the final maps in the result (figures)
    log_comm: bool = False          # log per-block communication payload sizes
    log_feasibility: bool = False   # log believed distance to the nearest undecided cell each block (E4b)
    check_joint: bool = False       # N=2, certified: log greedy/joint objective ratio each block

class Agent:
    def __init__(self, i: int, pose: np.ndarray, feats: LocalFeatures, cfg: Config):
        self.i = i
        self.pose = pose.copy()
        self.model = BayesLinearMap(feats, cfg.noise_var, cfg.prior_var)
        self.X_train = []
        self.y_train = []
        self.blk_train = []
        self.cal = {"X": [], "y": [], "fold": [], "blk": [], "scan": []}
        self.n_scan = 0                 # per-agent scan counter: one scan is the independence unit of Asm. 2
        self.traj = [pose.copy()]
        self.path_len = 0.0
        self.cert: AgentCert | None = None
        self.fused: FusedMap | None = None
        self.lawn_wp = None
        self.lawn_i = 0
        self.blocked = set()            # cells the robot bumped into (used for traversability only)
        self.relaxed = 0                # traversability relaxation level used at the last plan
        self.e_first = None             # (M,2) e-values frozen at the block in which each cell was first gated
        self.gated_before = None
        self.logodds = None             # prob guard: decaying log-odds occupancy grid
        self.tracks = []                # cp_track guard: (centroid, velocity) of dynamic blobs at the last block
        self.track_err = []             # cp_track guard: (block, prediction error) pairs
        self.cell_hist = {}            # cell -> (block, majority label) from the latest block in which it was sampled
        self.last_obs = None            # block in which each cell was last sampled by this robot (-1 = never)
        self.pose_bias = None           # accumulated pose-estimate drift (x, y, theta), None = perfectly localised
        self.mpairs = []                # lidar miss-event pairs (block, cell, chi, staleness): chi = 1{observed label != committed label}; region map only
        self.tpairs = []                # traversal pairs (block, cell, chi, staleness): chi = 1{cell occupied when the robot entered it}; calibration
        self.miss_n = None; self.miss_c = None   # even-block lidar miss counts per cell (region map)
        self.label_prev = None; self.stale_prev = None; self._mpairs_done = -1
        self.plan_cells = np.zeros(6, int); self.plan_occ = np.zeros(6, int)   # admitted cells / occupied at traversal, by staleness bucket
        self.pairs = []                 # temporal pairs (block, cell, changed, fold) at lag 1; fold 0 = taxonomy, 1 = calibration
        self.change_n = None            # per-cell counts (observations, changes) at lag 1
        self.change_c = None
        self.aci_a = None               # per-class inner level adapted by ACI
        self.h_p = None; self.h_ok = None; self.h_class = None
        self.plan_S = None
        self.plan_P = None
        self.plan_gain = 0.0
        self.plan_gain_obs = 0.0
        self.plan_len = 0.0

    def train_xy(self):
        return np.concatenate(self.X_train, 0) if self.X_train else np.zeros((0, 2))

    def cal_arrays(self):
        if not self.cal["X"]:
            z = np.zeros((0, 2)); e = np.zeros(0)
            return z, e, e.astype(bool)
        f = np.concatenate(self.cal["fold"], 0)
        return (np.concatenate(self.cal["X"], 0), np.concatenate(self.cal["y"], 0),
                f if f.dtype.kind == "i" else f.astype(bool))

    def cal_scan(self):
        """Scan index of each calibration label; the independence unit of Assumption 2, since one scan's
        pose error is shared by all of its endpoints."""
        return np.concatenate(self.cal["scan"], 0) if self.cal["scan"] else np.zeros(0, int)

    def cal_block(self):
        return np.concatenate(self.cal["blk"], 0) if self.cal["blk"] else np.zeros(0, int)


class Simulation:
    def __init__(self, cfg: Config):
        self.k = -1
        self.cfg = cfg
        world_rng = np.random.default_rng(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed if cfg.noise_seed < 0 else 10_000 + cfg.noise_seed)
        self._pair_rng = np.random.default_rng(30_000 + cfg.seed)   # fold coin for temporal pairs (taxonomy / calibration)
        if cfg.world_file:
            self.world: World = World(np.load(cfg.world_file), cfg.res)
        else:
            self.world: World = make_world(cfg.seed, cfg.W, cfg.H, cfg.res, cfg.wall_thick, cfg.n_walls, cfg.n_boxes)
        w = self.world
        self.static_occ = w.occ.copy()
        self.hazards = []
        if cfg.n_hazards > 0:
            hr = np.random.default_rng(cfg.seed + 999)
            free = ndimage.distance_transform_edt(~self.static_occ) * w.res
            cand = np.argwhere(free > cfg.hazard_size / 2 + 0.25)
            for _ in range(cfg.n_hazards):
                iy, ix = cand[hr.integers(len(cand))]
                self.hazards.append(np.array([(ix + 0.5) * w.res, (iy + 0.5) * w.res]))
            self._hazard_rng = hr
            self._hazard_cand = cand
            self._hazard_goal = [None] * cfg.n_hazards
            self._apply_hazards()
        self.region = None
        if cfg.p_flip_region > 0:
            rr = np.random.default_rng(cfg.seed + 777)
            rw, rh = cfg.region_size
            x0 = rr.uniform(1.0, w.W - rw - 1.0); y0 = rr.uniform(1.0, w.H - rh - 1.0)
            self.region = (x0, y0, x0 + rw, y0 + rh)
        self.lidar = Lidar(LidarSpec(cfg.n_beams, np.pi, cfg.max_range, cfg.sigma_r, cfg.sigma_phi,
                                     cfg.sigma_e, cfg.sigma_theta, cfg.free_spacing))
        self.lidars = []
        for i in range(cfg.N):
            rng_ = cfg.range_agents[i] if cfg.range_agents else cfg.max_range
            nb_ = cfg.beams_agents[i] if cfg.beams_agents else cfg.n_beams
            se_ = cfg.sigma_e_agents[i] if cfg.sigma_e_agents else cfg.sigma_e
            st_ = cfg.sigma_theta_agents[i] if cfg.sigma_theta_agents else cfg.sigma_theta
            self.lidars.append(Lidar(LidarSpec(nb_, np.pi, rng_, cfg.sigma_r, cfg.sigma_phi, se_, st_, cfg.free_spacing)))
        self.feats = LocalFeatures(w.W, w.H, cfg.feat_spacing, cfg.ell)
        self.PhiQ = self.feats.phi(w.Q)
        self.gg = GridGraph(w.nx, w.ny, w.res)
        starts = sample_starts(w, cfg.N, world_rng if cfg.start_seed < 0 else np.random.default_rng(20_000 + cfg.start_seed))
        self.agents = [Agent(i, starts[i], self.feats, cfg) for i in range(cfg.N)]
        for a in self.agents:
            a.lawn_wp = lawnmower_waypoints(w.W, w.H, a.i, cfg.N, cfg.lane_spacing)
        self.acq = Acquisition(cfg.planner, score=cfg.score) if cfg.planner in ("certified", "certified_r", "certified_m", "entropy", "entropy_u", "entropy_g", "lse", "random") else None
        self.log = []
        self.d = {i: len(self.neighbourhood(i)) for i in range(cfg.N)}

    # ---- topology ----------------------------------------------------------
    def neighbourhood(self, i: int) -> list[int]:
        N = self.cfg.N
        if N == 1:
            return [0]
        if self.cfg.topology == "mesh":
            return list(range(N))
        if self.cfg.topology == "ring":
            return sorted({i, (i - 1) % N, (i + 1) % N})
        if self.cfg.topology == "star":             # agent 0 is the hub: d_hub = N, d_leaf = 2
            return list(range(N)) if i == 0 else sorted({0, i})
        raise ValueError(self.cfg.topology)

    # ---- moving hazards ---------------------------------------------------------
    def _apply_hazards(self):
        w = self.world; occ = self.static_occ.copy(); h = self.cfg.hazard_size / 2
        for c in self.hazards:
            x0, x1 = int((c[0] - h) / w.res), int(np.ceil((c[0] + h) / w.res)); y0, y1 = int((c[1] - h) / w.res), int(np.ceil((c[1] + h) / w.res))
            occ[max(y0, 0):min(y1, w.ny), max(x0, 0):min(x1, w.nx)] = True
        w.set_occ(occ)

    def move_hazards(self):
        """Each hazard takes a random step of `hazard_step` metres, staying in static free space and not running over a robot."""
        if not self.hazards:
            return
        w = self.world; free = ndimage.distance_transform_edt(~self.static_occ) * w.res
        for hi, c in enumerate(self.hazards):
            for attempt in range(8):
                if self.cfg.hazard_motion == "goal":
                    g = self._hazard_goal[hi]
                    if g is None or np.linalg.norm(g - c) < self.cfg.hazard_step or attempt > 0:
                        iy, ix = self._hazard_cand[self._hazard_rng.integers(len(self._hazard_cand))]
                        g = np.array([(ix + 0.5) * w.res, (iy + 0.5) * w.res]); self._hazard_goal[hi] = g
                    d = g - c; step = min(self.cfg.hazard_step, np.linalg.norm(d))
                    n = c + step * d / max(np.linalg.norm(d), 1e-9)
                else:
                    ang = self._hazard_rng.uniform(0, 2 * np.pi)
                    n = c + self.cfg.hazard_step * np.array([np.cos(ang), np.sin(ang)])
                if not (0 < n[0] < w.W and 0 < n[1] < w.H):
                    continue
                ix, iy = w.cell_of(n)
                if free[iy[0], ix[0]] <= self.cfg.hazard_size / 2 + 0.25:
                    continue
                if any(np.linalg.norm(n - a.pose[:2]) < self.cfg.hazard_size / 2 + self.cfg.hazard_clear for a in self.agents):
                    continue                              # a hazard does not run over a robot: footprint clearance hazard_clear
                c[:] = n
                break
        self._apply_hazards()

    # ---- sensing -------------------------------------------------------------
    def scan_at(self, a: Agent, pose: np.ndarray, to_train: bool):
        if not hasattr(self, "_block_samples"):
            self._block_samples = {}
        place = None if a.pose_bias is None else pose + a.pose_bias
        X, y = self.lidars[a.i].scan(self.world.occ, self.world.res, pose, self.rng, place_pose=place)
        if len(X) == 0:
            return
        pf = self.cfg.p_flip_agents[a.i] if self.cfg.p_flip_agents else self.cfg.p_flip
        if pf > 0:
            flip = self.rng.random(len(y)) < pf
            y = np.where(flip, -y, y)
        if self.region is not None:
            x0, y0, x1, y1 = self.region
            inside = (X[:, 0] >= x0) & (X[:, 0] <= x1) & (X[:, 1] >= y0) & (X[:, 1] <= y1)
            flip = inside & (self.rng.random(len(y)) < self.cfg.p_flip_region)
            y = np.where(flip, -y, y)
        if self.cfg.p_flip_range > 0:
            r = np.linalg.norm(X - pose[None, :2], axis=1) / self.cfg.max_range
            flip = self.rng.random(len(y)) < self.cfg.p_flip_range * np.clip(r, 0, 1) ** 2
            y = np.where(flip, -y, y)
        self._block_samples[a.i] = self._block_samples.get(a.i, 0) + len(X)
        if a.last_obs is None:
            a.last_obs = np.full(self.world.M, -1, int)
        a.last_obs[self.world.flat_of(X)] = self.k
        if self.cfg.guard == "prob":
            if a.logodds is None:
                a.logodds = np.zeros(self.world.M)
            flat = self.world.flat_of(X)
            np.add.at(a.logodds, flat, np.where(y > 0, 0.85, -0.4))   # standard inverse-sensor log-odds increments
            a.logodds = np.clip(a.logodds, -6.0, 6.0)
        if to_train:
            a.model.add(X, y); a.X_train.append(X); a.y_train.append(y); a.blk_train.append(np.full(len(X), self.k))
        else:
            a.cal["X"].append(X); a.cal["y"].append(y)
            a.cal["fold"].append(self.rng.integers(4, size=len(X)) if self.cfg.split4
                                  else (self.rng.random(len(X)) < 0.5))
            a.cal["blk"].append(np.full(len(X), self.k))
            a.cal["scan"].append(np.full(len(X), a.n_scan))
        a.n_scan += 1

    def bootstrap(self):
        """Turn in place: four scans (2 train, 2 cal) so the first certificate exists."""
        for a in self.agents:
            for j, dth in enumerate([0, np.pi / 2, np.pi, 3 * np.pi / 2]):
                p = a.pose.copy(); p[2] += dth
                self.scan_at(a, p, to_train=(j < 2))

    # ---- certification -------------------------------------------------------
    def _update_temporal_pairs(self, a: Agent):
        """Majority label per cell from this block's samples; a cell seen in the previous block yields a lag-1 pair."""
        w = self.world
        X, y, blk = a.cal_arrays()[0], a.cal_arrays()[1], a.cal_block()
        Xt = a.train_xy(); yt = np.concatenate(a.y_train, 0) if a.y_train else np.zeros(0)
        bt = np.concatenate(a.blk_train, 0) if a.blk_train else np.zeros(0, int)
        cur = self.k - 1                                   # the newest completed block
        if cur < 0 or getattr(a, "_pairs_done", -1) >= cur:
            return
        a._pairs_done = cur
        Xa = np.concatenate([X[blk == cur], Xt[bt == cur]], 0); ya = np.concatenate([y[blk == cur], yt[bt == cur]], 0)
        if len(Xa) == 0:
            return
        flat = w.flat_of(Xa)
        if a.change_n is None:
            a.change_n = np.zeros(w.M); a.change_c = np.zeros(w.M); a.tax_n = np.zeros(w.M); a.tax_c = np.zeros(w.M)
        for c in np.unique(flat):
            lab = 1.0 if ya[flat == c].mean() > 0 else -1.0
            prev = a.cell_hist.get(int(c))
            if prev is not None and prev[0] == cur - 1:
                changed = float(lab != prev[1])
                # each pair goes to the taxonomy fold (change map -> classes) or the calibration fold (change rate), never
                # both: the class of a calibration pair must not depend on its own outcome (as in the static taxonomy)
                fold = int(self._pair_rng.integers(2))
                a.pairs.append((cur, int(c), changed, fold))
                a.change_n[c] += 1; a.change_c[c] += changed          # all pairs (used by the cp_track baseline)
                if fold == 0:
                    a.tax_n[c] += 1; a.tax_c[c] += changed            # taxonomy fold only (classes of the certificate)
            a.cell_hist[int(c)] = (cur, lab)

    def _track_hazards(self, a: Agent):
        """Lindemann-style guard: dynamic blobs (currently believed occupied cells in the dynamic class; or the true
        hazard positions for cp_track_oracle) are tracked, predicted one block ahead with constant velocity, and a
        conformal region of radius equal to the (1-alpha) quantile of past per-block maximum prediction errors (max over
        agents, so the region covers all tracked agents jointly, as in Lindemann et al. 2023) is forbidden around each
        prediction."""
        cfg, w = self.cfg, self.world
        if cfg.guard == "cp_track_oracle":
            # detections solved: the true hazard positions (identity unknown; association as below)
            cents = [c.copy() for c in self.hazards]
        else:
            if a.change_n is None:
                a.forbid = np.zeros(w.M, bool); return
            k = max(int(round(3.0 / w.res)), 1)
            n = ndimage.uniform_filter(a.change_n.reshape(w.ny, w.nx), size=k, mode="constant") * k * k
            c = ndimage.uniform_filter(a.change_c.reshape(w.ny, w.nx), size=k, mode="constant") * k * k
            dyn = (((c + 0.1) / (n + 2.0)) > cfg.change_tau).ravel()
            occ_now = (a.cert.gate & (a.cert.mu > cfg.trav_thr) & dyn).reshape(w.ny, w.nx)
            lab, nb = ndimage.label(occ_now, structure=np.ones((3, 3)))
            cents = [np.array(ndimage.center_of_mass(occ_now, lab, j)) [::-1] * w.res + w.res / 2 for j in range(1, nb + 1)]
            cents = [ce for ce, j in zip(cents, range(1, nb + 1)) if (lab == j).sum() >= 4]
        # association with previous tracks (nearest within 6 m), prediction error bookkeeping
        new_tracks = []; block_err = []
        for ce in cents:
            best = None; bd = 6.0
            for (pc, pv) in a.tracks:
                d = np.linalg.norm(ce - (pc + pv))
                if d < bd: bd, best = d, (pc, pv)
            if best is not None:
                block_err.append(bd); vel = ce - best[0]
            else:
                vel = np.zeros(2)
            new_tracks.append((ce, vel))
        if block_err:
            a.track_err.append((self.k, max(block_err)))   # nonconformity = max over tracked agents (Lindemann et al. 2023): joint region
        a.tracks = new_tracks
        errs = np.array([e for (b, e) in a.track_err if b >= self.k - cfg.track_window])
        if len(errs) >= 5:
            q = int(np.ceil((len(errs) + 1) * (1 - cfg.alpha))); r = float(np.sort(errs)[min(q, len(errs)) - 1])
        else:
            r = 3.0
        forbid = np.zeros(w.M, bool)
        for (ce, vel) in a.tracks:
            pred = ce + vel
            forbid |= np.linalg.norm(w.Q - pred[None, :], axis=1) <= r + cfg.hazard_size / 2
        a.forbid = forbid; a.track_r = r

    @staticmethod
    def _stale_bucket(st):
        """Staleness (blocks since the neighbourhood last sampled the cell) -> bucket 0: 1, 1: 2-3, 2: 4-7, 3: 8-15, 4: 16-31, 5: >= 32 or never."""
        return np.clip(np.floor(np.log2(np.maximum(st, 1))).astype(int), 0, NB - 1)

    def _update_miss_pairs(self, a: Agent):
        """Lidar miss-event pairs: every cell sampled during block k-1 whose certificate of block k-1 committed a label
        gives chi = 1{observed majority label != committed label}, tagged with the cell's staleness at block k-1. Used
        for the region taxonomy only: a lidar under-samples occupied cells (a blob shows its front face), so these pairs
        cannot calibrate the miss rate the robot meets when it drives; traversal pairs (execute) do that."""
        w = self.world
        cur = self.k - 1
        if cur < 0 or a._mpairs_done >= cur or a.label_prev is None:
            return
        a._mpairs_done = cur
        X, y, blk = a.cal_arrays()[0], a.cal_arrays()[1], a.cal_block()
        Xt = a.train_xy(); yt = np.concatenate(a.y_train, 0) if a.y_train else np.zeros(0)
        bt = np.concatenate(a.blk_train, 0) if a.blk_train else np.zeros(0, int)
        Xa = np.concatenate([X[blk == cur], Xt[bt == cur]], 0); ya = np.concatenate([y[blk == cur], yt[bt == cur]], 0)
        if len(Xa) == 0:
            return
        if a.miss_n is None:
            a.miss_n = np.zeros(w.M); a.miss_c = np.zeros(w.M)
        flat = w.flat_of(Xa)
        for c in np.unique(flat):
            L = a.label_prev[c]
            if L == 0:
                continue
            lab = 1.0 if ya[flat == c].mean() > 0 else -1.0
            chi = float(lab != L)
            a.mpairs.append((cur, int(c), chi, int(a.stale_prev[c])))
            if cur % 2 == 0:
                # region map from even blocks only; calibration uses odd blocks' traversal pairs (a bump never classifies itself)
                a.miss_n[c] += 1; a.miss_c[c] += chi

    def _last_obs(self, a: Agent):
        """Block in which each cell was last observed by the neighbourhood: a sample within obs_r of the cell (the gate
        construction of the static certificate; free samples are free_spacing apart along the beams)."""
        w = self.world
        lo = np.max(np.stack([self.agents[j].last_obs if self.agents[j].last_obs is not None else np.full(w.M, -1)
                              for j in self.neighbourhood(a.i)], 0), 0)
        r = max(int(round(self.cfg.obs_r / w.res)), 1)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        return ndimage.maximum_filter(lo.reshape(w.ny, w.nx), footprint=(xx ** 2 + yy ** 2) <= r * r, mode="constant", cval=-1).ravel()

    def _staleness(self, a: Agent):
        """Blocks since each cell was last observed by the neighbourhood (10**6 = never)."""
        lo = self._last_obs(a)
        return np.where(lo < 0, 10 ** 6, self.k - lo)

    def _ltt_certificate(self, a: Agent):
        """Learn-then-Test admission of the traversable set (the construction of Haroon & Fleming, certified safety
        curation, transplanted from training-set composition to the set of cells a robot may drive through).

        Risk score: each cell is binned by (local lidar-miss rate, staleness bucket) and scored with the weighted
        miss rate of the traversal pairs in its bin, shrunk towards the pooled rate. A threshold lambda admits
        S(lambda) = {cells whose score <= lambda}. Thresholds are tested in a fixed sequence from the safest
        upwards: for each, the one-sided binomial p-value of "the miss rate of S(lambda) exceeds alpha" is
        computed on the calibration pairs that fall inside S(lambda), and the procedure stops at the first failure
        to reject, returning the last rejected threshold. If the first test fails, nothing is certified and the
        robot falls back immediately to the uncertified fresh rule (committed free and observed in the last
        block), counted and reported separately, exactly as the warm-up and probe blocks are. Holding position
        instead would also be sound, but the original returns a conservative selection rather than nothing.

        Guarantee (same conditioning as Prop. A): with probability at least 1 - delta over the calibration pairs,
        the admitted set's miss rate is at most alpha. It is a high-probability statement about the *selected set*
        rather than a marginal expectation per class, which is what the per-class guard provides. The pairs are
        generated sequentially rather than drawn i.i.d., so this inherits Prop. A's within-class exchangeability
        assumption; it does not inherit the finite-pool exactness of the original hypergeometric test.
        """
        from scipy.stats import binom
        cfg, w = self.cfg, self.world
        st = self._staleness(a); a.stale_prev = st
        team = [self.agents[j] for j in self.neighbourhood(a.i)] if cfg.share_pairs else [a]
        team = [b for b in team if b.miss_n is not None and b.tpairs]
        alpha = cfg.guard_alpha if cfg.guard_alpha > 0 else cfg.alpha
        if not team:
            a.h_class = np.zeros(w.M, int); a.h_ok = np.zeros(w.M, bool); a.ltt_lambda = np.nan; a.ltt_n = 0
            return
        pairs = [pr for b in team for pr in b.tpairs]
        miss_n = sum(b.miss_n for b in team); miss_c = sum(b.miss_c for b in team)
        k = max(int(round(3.0 / w.res)), 1)
        n = ndimage.uniform_filter(miss_n.reshape(w.ny, w.nx), size=k, mode="constant") * k * k
        c = ndimage.uniform_filter(miss_c.reshape(w.ny, w.nx), size=k, mode="constant") * k * k
        rate = ((c + 0.1) / (n + 2.0)).ravel()                       # local miss rate from the taxonomy fold
        edges = np.linspace(0.0, max(cfg.change_tau * 2, 1e-3), cfg.ltt_bins + 1)[1:-1]
        rbin = np.digitize(rate, edges)                              # 0 .. ltt_bins-1
        cell_bin = rbin * NB + self._stale_bucket(st)
        nb_tot = cfg.ltt_bins * NB

        ev_blocks = sorted({b for (b, _, _, _) in pairs if b % 2 == 1}, reverse=True)[:cfg.horizon_window]
        rank = {b: i for i, b in enumerate(ev_blocks)}
        wsum = np.zeros(nb_tot); num = np.zeros(nb_tot); cal = []
        for (b, cell, chi, s_pair) in pairs:
            if b not in rank:
                continue
            g = int(rbin[cell]) * NB + int(self._stale_bucket(np.array([s_pair]))[0])
            wgt = cfg.horizon_rho ** rank[b]
            wsum[g] += wgt; num[g] += wgt * chi
            cal.append((g, chi))
        if not cal:
            a.h_class = cell_bin; a.h_ok = np.zeros(w.M, bool); a.ltt_lambda = np.nan; a.ltt_n = 0
            return
        pooled = float(np.sum(num) / max(np.sum(wsum), 1e-9))
        m0 = 2.0                                                     # shrinkage towards the pooled rate
        score_bin = (num + m0 * pooled) / (wsum + m0)                # risk score of each bin
        a.h_class = cell_bin
        cal_g = np.array([g for g, _ in cal]); cal_chi = np.array([chi for _, chi in cal])
        # fixed-sequence testing over thresholds, safest first
        lam_grid = np.unique(score_bin[np.isfinite(score_bin)])
        lam_hat = np.nan
        for lam in lam_grid:
            sel = score_bin[cal_g] <= lam + 1e-12
            nj = int(sel.sum())
            if nj < 10:
                continue                                             # too few calibration pairs to test this threshold
            kj = int(cal_chi[sel].sum())
            pval = float(binom.cdf(kj, nj, alpha))                   # H0: the admitted set's miss rate exceeds alpha
            if pval <= cfg.ltt_delta:
                lam_hat = lam
            else:
                break                                                # fixed sequence: stop at the first failure
        a.ltt_lambda = lam_hat; a.ltt_n = len(cal)
        a.h_ok = (score_bin[cell_bin] <= lam_hat) if np.isfinite(lam_hat) else np.zeros(w.M, bool)

    def _horizon_certificate(self, a: Agent):
        """Staleness-indexed persistence certificate. Classes = region (stable/dynamic, from the even-block lidar miss
        map, box 3 m) x staleness bucket (1, 2-3, 4-7, 8-15, 16-31, >= 32 blocks). Per class, the weighted miss rate of the
        odd-block traversal pairs over the last horizon_window blocks (weights rho^age; Barber et al. 2023) is the p-value
        of "the committed label is wrong when the robot enters the cell"; the label is admitted for traversal iff that
        p-value is at or below the class's inner level, adapted by one-sided ACI on all traversal pairs of the last
        block. Teams pool pairs and counts over the neighbourhood."""
        cfg, w = self.cfg, self.world
        st = self._staleness(a); a.stale_prev = st
        team = [self.agents[j] for j in self.neighbourhood(a.i)] if cfg.share_pairs else [a]
        team = [b for b in team if b.miss_n is not None and b.tpairs]
        target = cfg.guard_alpha if cfg.guard_alpha > 0 else cfg.alpha
        if a.aci_a is None:
            a.aci_a = np.full(2 * NB, target)
            a.aci_n = np.zeros(2 * NB, int)      # T_g of Prop. 8.1: blocks at which class g updated
        if not team:
            a.h_class = np.zeros(w.M, int); a.h_ok = np.zeros(w.M, bool); a.h_p = np.ones(2 * NB); return
        pairs = [pr for b in team for pr in b.tpairs]
        miss_n = sum(b.miss_n for b in team); miss_c = sum(b.miss_c for b in team)
        k = max(int(round(3.0 / w.res)), 1)
        n = ndimage.uniform_filter(miss_n.reshape(w.ny, w.nx), size=k, mode="constant") * k * k
        c = ndimage.uniform_filter(miss_c.reshape(w.ny, w.nx), size=k, mode="constant") * k * k
        region = (((c + 0.1) / (n + 2.0)).ravel() > cfg.change_tau).astype(int)
        cls = NB * region + self._stale_bucket(st)
        p = np.ones(2 * NB); wsum = np.zeros(2 * NB); num = np.zeros(2 * NB)
        # window = the last horizon_window odd (calibration) blocks in which the team generated traversal pairs; age is
        # the rank of the block among those, weights rho^age (rho = 1: equal weights, the stationary case)
        ev_blocks = sorted({b for (b, _, _, _) in pairs if b % 2 == 1}, reverse=True)[:cfg.horizon_window]
        rank = {b: i for i, b in enumerate(ev_blocks)}
        for (b, cell, chi, s_pair) in pairs:
            if b not in rank:
                continue
            g = NB * region[cell] + int(self._stale_bucket(np.array([s_pair]))[0])
            wgt = cfg.horizon_rho ** rank[b]; wsum[g] += wgt; num[g] += wgt * chi
        for g in range(2 * NB):
            p[g] = (num[g] + 1.0) / (wsum[g] + 1.0) if wsum[g] > 0 else (a.h_p[g] if a.h_p is not None and len(a.h_p) == 2 * NB else 1.0)
        if cfg.aci_gamma > 0 and hasattr(a, "h_ok_prev2"):
            # one-sided ACI on the inner level: a_{t+1} = min(a_t + gamma (alpha - err_t), alpha); err_t = miss rate among
            # re-observed cells the block-(k-2) certificate admitted in class g, 0 if the class admitted nothing, no update
            # if admitted cells were not re-observed. Not clipped from below: a_t <= 0 admits nothing, which is
            # what keeps a_t >= -gamma(1-alpha) and so bounds the long-run miss rate (Prop. 8.1 in the paper).
            for g in range(2 * NB):
                cert_g = a.h_ok_prev2 & (a.h_class_prev2 == g)
                if not cert_g.any():
                    err = 0.0
                else:
                    errs = [chi for (b, cell, chi, s_pair) in pairs if b == self.k - 1 and cert_g[cell]]
                    if not errs:
                        continue
                    err = float(np.mean(errs))
                a.aci_a[g] = min(a.aci_a[g] + cfg.aci_gamma * (target - err), target)
                a.aci_n[g] += 1
        a.h_class = cls; a.h_p = p; a.h_ok = p[cls] <= a.aci_a[cls]

    def certify(self):
        cfg, w = self.cfg, self.world
        if cfg.guard in ("certified", "ltt"):
            for a in self.agents:
                self._update_miss_pairs(a)
            for a in self.agents:
                self._horizon_certificate(a) if cfg.guard == "certified" else self._ltt_certificate(a)
        for a in self.agents:
            if cfg.guard == "prob" and a.logodds is not None and self.k > 0:
                a.logodds *= cfg.prob_decay
            if cfg.guard in ("cp_track", "cp_track_oracle"):
                self._update_temporal_pairs(a)
            if cfg.window > 0:
                # windowed certificate: model, gate and calibration folds from the last `window` blocks only
                kmin = self.k - cfg.window
                keep = [j for j, b in enumerate(a.blk_train) if b[0] >= kmin]
                model = BayesLinearMap(self.feats, cfg.noise_var, cfg.prior_var)
                for j in keep:
                    model.add(a.X_train[j], a.y_train[j])
                a.model = model
                Xtr = np.concatenate([a.X_train[j] for j in keep], 0) if keep else np.zeros((0, 2))
                mu, sd = a.model.predict(self.PhiQ)
                gate = gate_from_samples(Xtr, w.nx, w.ny, w.res, cfg.gate_r)
                Xc, yc, fold = a.cal_arrays(); cb = a.cal_block()
                sel = cb >= kmin; Xc, yc, fold = Xc[sel], yc[sel], fold[sel]
            else:
                mu, sd = a.model.predict(self.PhiQ)
                Xtr = a.train_xy()
                if cfg.gate_window > 0:
                    keep = [j for j, b in enumerate(a.blk_train) if b[0] >= self.k - cfg.gate_window]
                    Xg = np.concatenate([a.X_train[j] for j in keep], 0) if keep else np.zeros((0, 2))
                    gate = gate_from_samples(Xg, w.nx, w.ny, w.res, cfg.gate_r)
                else:
                    gate = gate_from_samples(Xtr, w.nx, w.ny, w.res, cfg.gate_r)
                Xc, yc, fold = a.cal_arrays()
            beta = cfg.safety * cfg.alpha / self.d[a.i]
            if len(Xc) < 20:
                a.cert = AgentCert(mu, sd, gate, np.ones(w.M), 0.0, 1.0, np.full(w.M, np.inf), np.inf,
                                   np.zeros((w.M, 2)), 0, 0)
                continue
            Phic = self.feats.phi(Xc)
            muc, sdc = a.model.predict(Phic)
            sc, rc = ~fold, fold
            if cfg.mondrian:
                # reliability taxonomy from the TRAINING fold only (G_k-measurable): contradiction rate of training
                # labels against the model mean at their location, box-smoothed; class 1 = unreliable
                ytr = np.concatenate(a.y_train, 0)
                mutr, _ = a.model.predict(self.feats.phi(Xtr))
                theta_tr = local_contradiction_map(Xtr, ytr, mutr, w.nx, w.ny, w.res)
                grp_cell = (theta_tr > cfg.mondrian_tau).astype(int)
                def grp_at(X):
                    ix = np.clip(np.floor(X[:, 0] / w.res).astype(int), 0, w.nx - 1); iy = np.clip(np.floor(X[:, 1] / w.res).astype(int), 0, w.ny - 1)
                    return grp_cell[iy * w.nx + ix]
                a.cert = calibrate_agent_grouped(mu, sd, gate, Xtr, Xc[sc], yc[sc], muc[sc], sdc[sc],
                                                 Xc[rc], yc[rc], muc[rc], sdc[rc], group_cell=grp_cell, group_cal=grp_at(Xc[sc]),
                                                 group_rec=grp_at(Xc[rc]), n_groups=2, beta=beta, weighted=cfg.weighted,
                                                 nx=w.nx, ny=w.ny, res=w.res, kde_bw=cfg.kde_bw, lam=cfg.lam,
                                                 score=cfg.score, eps_ucb=cfg.eps_ucb)
                a.cert.theta_map = theta_tr
            elif cfg.split4:
                # folds: 0 claim, 1 recal-score, 2 recal, 3 planning. The claim certificate is scored on fold 0
                # and recalibrated on fold 2 against fold 1, so no fold 0 label enters the level it is judged at;
                # the acquisition reads a separate certificate built from fold 3 only.
                f = fold
                A_, B_, C_, D_ = (f == 0), (f == 1), (f == 2), (f == 3)
                a.cert = calibrate_agent(mu, sd, gate, Xtr, Xc[A_], yc[A_], muc[A_], sdc[A_],
                                         Xc[C_], yc[C_], muc[C_], sdc[C_], beta=beta, weighted=cfg.weighted,
                                         nx=w.nx, ny=w.ny, res=w.res, kde_bw=cfg.kde_bw, lam=cfg.lam,
                                         score=cfg.score, eps_ucb=cfg.eps_ucb,
                                         X_rs=Xc[B_], y_rs=yc[B_], mu_rs=muc[B_], sd_rs=sdc[B_], denom_a=True)
                a.cert_plan = calibrate_agent(mu, sd, gate, Xtr, Xc[D_], yc[D_], muc[D_], sdc[D_],
                                              Xc[D_], yc[D_], muc[D_], sdc[D_], beta=beta, weighted=cfg.weighted,
                                              nx=w.nx, ny=w.ny, res=w.res, kde_bw=cfg.kde_bw, lam=cfg.lam,
                                              score=cfg.score, eps_ucb=cfg.eps_ucb)
            else:
                scn = a.cal_scan()
                a.cert = calibrate_agent(mu, sd, gate, Xtr, Xc[sc], yc[sc], muc[sc], sdc[sc],
                                         Xc[rc], yc[rc], muc[rc], sdc[rc], beta=beta, weighted=cfg.weighted,
                                         nx=w.nx, ny=w.ny, res=w.res, kde_bw=cfg.kde_bw, lam=cfg.lam,
                                         score=cfg.score, eps_ucb=cfg.eps_ucb, denom_delta=cfg.denom_delta,
                                         scan_rec=(scn[rc] if len(scn) == len(Xc) else None), w_clip=cfg.w_clip)
            if cfg.planner == "certified_r":
                a.cert.theta_map = local_contradiction_map(Xc, yc, muc, w.nx, w.ny, w.res)
            if cfg.log_combined:
                # exact block-combined certificate (Corollary 1): block-k calibration fold only, its own density,
                # frozen for the cells first gated in block k (the gate is G_k-measurable: training locations only)
                if a.e_first is None:
                    a.e_first = np.zeros((w.M, 2)); a.gated_before = np.zeros(w.M, bool)
                cb_all = a.cal_block(); blk = cb_all == cb_all.max()   # the most recent block's calibration fold
                new = a.cert.gate & ~a.gated_before
                if blk.sum() >= 20 and new.any():
                    sb, rb = blk & ~fold, blk & fold
                    # the target law for block k is uniform over the cells it is *selected* for: Cor. 2 needs
                    # u_k proportional to omega_k * u, not the full gate, so `new` is passed as the gate here
                    cb = calibrate_agent(mu, sd, new, Xtr, Xc[sb], yc[sb], muc[sb], sdc[sb],
                                         Xc[rb], yc[rb], muc[rb], sdc[rb], beta=beta, weighted=cfg.weighted,
                                         nx=w.nx, ny=w.ny, res=w.res, kde_bw=cfg.kde_bw, lam=cfg.lam,
                                         score=cfg.score, eps_ucb=cfg.eps_ucb)
                    a.e_first[new] = cb.e[new]
                a.gated_before |= new
        for a in self.agents:
            certs = [self.agents[j].cert for j in self.neighbourhood(a.i)]
            gated = np.any(np.stack([c.gate for c in certs], 0), 0)
            # e-BH at level 2*alpha over the 2|G| (cell,label) hypotheses -> FCR <= alpha (Sec. III)
            a.fused = self._fuse_fcr(certs, gated)
            if cfg.split4 and all(getattr(self.agents[j], "cert_plan", None) is not None for j in self.neighbourhood(a.i)):
                cp = [self.agents[j].cert_plan for j in self.neighbourhood(a.i)]
                a.fused_plan = fuse(cp, cfg.alpha)
            if cfg.guard in ("cp_track", "cp_track_oracle"):
                self._track_hazards(a)
            if cfg.guard == "certified":
                if hasattr(a, "h_ok_prev"):
                    a.h_ok_prev2 = a.h_ok_prev; a.h_class_prev2 = a.h_class_prev
                a.h_ok_prev = a.h_ok.copy(); a.h_class_prev = a.h_class.copy()
                a.label_prev = np.where(a.fused.committed, a.fused.label, 0.0)   # the labels this block's guard acts on
            if cfg.guard == "ltt":
                a.label_prev = np.where(a.fused.committed, a.fused.label, 0.0)

    def _fuse_fcr(self, certs, gated):
        from .conformal import ebh_threshold
        F = fuse(certs, self.cfg.alpha)
        if self.cfg.trust_tau > 0:
            # Detector baseline: a teammate contributes to the shared coverage mask only while its own
            # estimated label-error rate clears a trust threshold. p_hat is each robot's weighted fraction of
            # well-observed recalibration labels contradicting its model, so no oracle names the bad robot.
            # A robot with no estimate yet (p_hat nan, early blocks) is trusted, and if the rule would drop
            # everyone the mask falls back to the full union rather than going empty.
            keep = [c for c in certs if not np.isfinite(c.p_hat) or c.p_hat <= self.cfg.trust_tau]
            if keep:
                F.nodata = ~np.any(np.stack([c.gate for c in keep], 0), 0)
        thr = ebh_threshold(F.e_fused[gated].ravel(), 2 * self.cfg.alpha) if gated.any() else np.inf
        G2 = ~(F.e_fused >= thr); n2 = G2.sum(1)
        F.Gamma_fcr, F.committed_fcr, F.thr_fcr = G2, n2 == 1, thr
        F.label_fcr = np.where(F.committed_fcr, np.where(G2[:, 1], 1.0, -1.0), 0.0)
        return F

    def _dist_undecided(self, a: Agent) -> float:
        """Believed path distance from the robot to the nearest observable undecided cell (feasibility signal)."""
        w = self.world
        src = int(w.flat_of(a.pose[:2])[0])
        trav = self._traversable(a, src)
        dist, _ = self.gg.shortest(trav, src, limit=200.0)
        tgt = a.fused.undecided & w.obs
        d = dist[tgt]
        d = d[np.isfinite(d)]
        return float(d.min()) if len(d) else np.inf

    def _region_stats(self, a: Agent, committed: np.ndarray, obs: np.ndarray) -> dict:
        if self.region is None:
            return {}
        w = self.world; x0, y0, x1, y1 = self.region
        Q = w.Q; inside = (Q[:, 0] >= x0) & (Q[:, 0] <= x1) & (Q[:, 1] >= y0) & (Q[:, 1] <= y1)
        tr = np.array(a.traj)
        seg = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1) if len(tr) > 1 else np.zeros(0)
        mid = (tr[1:, :2] + tr[:-1, :2]) / 2 if len(tr) > 1 else np.zeros((0, 2))
        ins = (mid[:, 0] >= x0) & (mid[:, 0] <= x1) & (mid[:, 1] >= y0) & (mid[:, 1] <= y1) if len(mid) else np.zeros(0, bool)
        ok = (a.fused.label == w.labels)
        expl = np.any(np.stack([self.agents[j].cert.gate for j in self.neighbourhood(a.i)], 0), 0) & obs
        wrong = committed & ~ok
        return dict(region_frac=float((inside & obs).sum() / max(obs.sum(), 1)),
                    cov_in=float(1.0 - wrong[inside & expl].sum() / max((inside & expl).sum(), 1)),
                    cov_out=float(1.0 - wrong[~inside & expl].sum() / max((~inside & expl).sum(), 1)),
                    acc_in=float(ok[committed & inside & obs].mean()) if (committed & inside & obs).any() else np.nan,
                    acc_out=float(ok[committed & ~inside & obs].mean()) if (committed & ~inside & obs).any() else np.nan,
                    c_in=float(committed[inside & obs].mean()) if (inside & obs).any() else np.nan,
                    c_out=float(committed[~inside & obs].mean()),
                    path_in_frac=float(seg[ins].sum() / max(seg.sum(), 1e-9)))

    def _bands(self, a: Agent, F: FusedMap, obs: np.ndarray) -> dict:
        """Where do calibrated gated cells sit relative to the commitment bands (residual score)?"""
        c = a.cert
        g = c.gate & obs
        fin = g & np.isfinite(c.q)
        out = dict(nocal_frac=float((g & ~np.isfinite(c.q)).sum() / max(g.sum(), 1)))
        ue = F.undecided & g
        out["undec_share_nocal"] = float((ue & ~np.isfinite(c.q)).sum() / max(ue.sum(), 1))
        if self.cfg.score == "residual":
            lo = fin & (c.sigma < 1.0 / c.q); hi = fin & (c.sigma > 2.0 / c.q); mid = fin & ~lo & ~hi
            out.update(band_lo=float(lo.sum() / max(fin.sum(), 1)), band_mid=float(mid.sum() / max(fin.sum(), 1)),
                       band_hi=float(hi.sum() / max(fin.sum(), 1)),
                       committed_in_lo=float(F.committed[lo].mean()) if lo.any() else np.nan,
                       committed_in_mid=float(F.committed[mid].mean()) if mid.any() else np.nan,
                       undec_share_mid=float((ue & mid).sum() / max(ue.sum(), 1)))
        else:
            amb = fin & (np.abs(c.mu) <= c.q * c.sigma)
            out.update(band_lo=np.nan, band_mid=float(amb.sum() / max(fin.sum(), 1)), band_hi=np.nan,
                       committed_in_lo=np.nan, committed_in_mid=np.nan,
                       undec_share_mid=float((ue & amb).sum() / max(ue.sum(), 1)))
        return out

    # ---- metrics -------------------------------------------------------------
    def record(self, k: int, t0: float):
        w = self.world
        obs = w.obs                       # all fractions are over the observable query set
        lab = w.labels[obs]
        rows = []
        # team-level metrics: mean pairwise robot separation; fraction of gated cells gated by >= 2 robots
        P = np.array([a.pose[:2] for a in self.agents])
        team_sep = float(np.mean([np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i + 1, len(P))])) if len(P) > 1 else 0.0
        G = np.stack([a.cert.gate for a in self.agents], 0).sum(0)
        team_overlap = float((G >= 2).sum() / max((G >= 1).sum(), 1))
        for a in self.agents:
            F = a.fused
            explored = ~F.nodata[obs]
            inset = F.Gamma[np.arange(w.M), (w.labels > 0).astype(int)][obs]
            committed = F.committed[obs]
            correct = committed & (F.label[obs] == lab)
            # marginal rule: rejections = (committed + 2*empty); false rejections = wrong committed + empty
            rej = committed.sum() + 2 * F.empty[obs].sum(); false_rej = (committed & ~correct).sum() + F.empty[obs].sum()
            cf = F.committed_fcr[obs]; correct_f = cf & (F.label_fcr[obs] == lab)
            n2 = F.Gamma_fcr[obs].sum(1); rej_f = cf.sum() + 2 * (n2 == 0).sum(); false_rej_f = (cf & ~correct_f).sum() + (n2 == 0).sum()
            comb = {}
            if self.cfg.log_combined:
                ef = sum(self.agents[j].e_first for j in self.neighbourhood(a.i)) / len(self.neighbourhood(a.i))
                G2 = ef < 1.0 / self.cfg.alpha; n2 = G2.sum(1); comm_c = (n2 == 1)[obs]
                inset2 = G2[np.arange(w.M), (w.labels > 0).astype(int)][obs]
                lab2 = np.where(G2[:, 1], 1.0, -1.0)[obs]
                comb = dict(c_comb=float(comm_c.mean()), cov_comb=float(inset2[explored].mean()) if explored.any() else 1.0,
                            acc_comb=float((comm_c & (lab2 == lab)).sum() / max(comm_c.sum(), 1)), empty_comb=float((n2 == 0)[obs].mean()))
            comm = {}
            if self.cfg.log_comm:
                comm = dict(comm_gate_cells=int(a.cert.gate.sum()), comm_e_nonzero=int((a.cert.e > 0).sum()),
                            comm_plan_pts=int(len(a.plan_S)) if a.plan_S is not None else 0,
                            comm_samples_block=int(self._block_samples.get(a.i, 0)), comm_cells=int(w.M))
            rows.append(dict(
                block=k, agent=a.i, path_len=a.path_len, collisions=int(getattr(a, "collisions", 0)),
                idle=int(getattr(a, "idle", 0)), bumps=int(getattr(a, "bumps", 0)), steps=int(getattr(a, "steps", 0)), h_ok_frac=float(a.h_ok.mean()) if a.h_ok is not None else 1.0, **{f"pc{b}": int(a.plan_cells[b]) for b in range(6)}, **{f"po{b}": int(a.plan_occ[b]) for b in range(6)}, **{f"tc{b}": int(getattr(a, "trav_cells", np.zeros(6, int))[b]) for b in range(6)}, **{f"to{b}": int(getattr(a, "trav_occ", np.zeros(6, int))[b]) for b in range(6)}, ltt_lambda=float(getattr(a, "ltt_lambda", np.nan)), ltt_n=int(getattr(a, "ltt_n", 0)), unc_cells=int(getattr(a, "unc_cells", 0)), unc_occ=int(getattr(a, "unc_occ", 0)), unc_blocks=int(getattr(a, "unc_blocks", 0)), **{f"aci{g}": float(a.aci_a[g]) if getattr(a, "aci_a", None) is not None else np.nan for g in range(2 * NB)}, **{f"acin{g}": int(a.aci_n[g]) if getattr(a, "aci_n", None) is not None else 0 for g in range(2 * NB)}, **{f"hp{g}": float(a.h_p[g]) if getattr(a, "h_p", None) is not None and len(a.h_p) == 2 * NB else np.nan for g in range(2 * NB)}, **comm, **comb, **self._region_stats(a, a.fused.committed, obs),
                fdr_marginal=float(false_rej / max(rej, 1)), c_fcr=float(cf.mean()), acc_fcr=float(correct_f.sum() / max(cf.sum(), 1)),
                fdr_fcr=float(false_rej_f / max(rej_f, 1)), thr_fcr=float(F.thr_fcr),
                **self._bands(a, F, obs),
                cov_explored=float(inset[explored].mean()) if explored.any() else 1.0,
                cov_whole=float(inset.mean()),
                c_whole=float(committed.mean()),
                c_explored=float(committed[explored].mean()) if explored.any() else 0.0,
                acc_committed=float(correct.sum() / max(committed.sum(), 1)),
                err_committed=float((committed & ~correct).mean()),   # P(committed and wrong)
                empty=float(F.empty[obs].mean()),
                undecided_whole=float(F.undecided[obs].mean()),
                nodata=float(F.nodata[obs].mean()),
                ambiguous=float((F.undecided[obs] & explored).mean()),
                explored=float(explored.mean()),
                a_i=a.cert.a, eps_i=a.cert.eps, q_i=a.cert.q_scalar,
                n_cal=a.cert.n_cal, n_recal=a.cert.n_recal, n_train=a.model.n_data,
                n_eff=a.cert.n_eff, n_w=a.cert.n_w, eps_prov=a.cert.eps_prov,
                n_ind=a.cert.n_ind, n_scan=a.cert.n_scan, m_bar=a.cert.m_bar,
                gain_pred=a.plan_gain, gain_pred_obs=a.plan_gain_obs, plan_len=a.plan_len,
                greedy_over_joint=getattr(self, "_ratio_log", np.nan), relaxed=a.relaxed,
                p_hat=a.cert.p_hat, dist_undecided=self._dist_undecided(a) if self.cfg.log_feasibility else np.nan,
                team_sep=team_sep, team_overlap=team_overlap,
                n_committed=int(committed.sum()), wall=time.time() - t0,
            ))
        self.log += rows

    # ---- planning -------------------------------------------------------------
    def plan_agent(self, a: Agent, order_pos: int, others_commit: np.ndarray | None,
                   prev_S: list[np.ndarray], undecided: np.ndarray, return_all: bool = False):
        cfg, w = self.cfg, self.world
        src = int(w.flat_of(a.pose[:2])[0])
        a.relaxed = 0
        trav = self._traversable(a, src)
        a.plan_gain = a.plan_gain_obs = a.plan_len = 0.0
        if cfg.planner == "lawnmower":
            return self._plan_lawn(a, trav, src)
        if cfg.planner in ("frontier", "frontier_g", "frontier_u"):
            dist, pred = self.gg.shortest(trav, src, limit=60.0)
            dmin = cfg.front_dmin if cfg.front_dmin > 0 else 0.6 * cfg.seg_len
            # frontier_g: coordinated frontier on the shared coverage mask (cells no neighbour has gated);
            # frontier_u: the same rule reading the fused undecided set; both deconflict targets in allocation order
            unknown = None if cfg.planner == "frontier" else undecided
            claimed = None if cfg.planner == "frontier" else self._claimed
            tgt = frontier_target(a.cert.gate, a.cert.mu, trav, w.nx, w.ny, dist, d_min=dmin, thr=cfg.trav_thr,
                                  unknown=unknown, claimed=claimed, res=w.res)
            if tgt is None:
                return self._random_segment(a, trav, src)
            if cfg.planner != "frontier":
                self._claimed.append(int(tgt))
            return self._truncate(self.gg.path(pred, src, tgt))
        cands, dist, pred = make_candidates(self.gg, trav, src, self.rng, cfg.n_cand, 0.6 * cfg.seg_len,
                                            1.2 * cfg.seg_len, cfg.seg_len, cfg.plan_pose_spacing, cfg.plan_max_poses)
        if cfg.generator == "rrt":
            # RRT-exploration-style generator: frontiers detected by a tree, navigated by the grid planner
            cands = rrt_candidates(self.gg, trav, src, self.rng, a.cert.gate, w.nx, w.ny, dist, pred,
                                   cfg.n_cand + cfg.n_front, cfg.seg_len, cfg.plan_pose_spacing, cfg.plan_max_poses,
                                   claimed=(self.gg.xy(self._claimed) if cfg.N > 1 and self._claimed else None)) or cands
        else:
            cands += frontier_candidates(self.gg, a.cert.gate, a.cert.mu, trav, w.nx, w.ny, dist, pred, src,
                                         cfg.n_front, cfg.seg_len, cfg.plan_pose_spacing, cfg.plan_max_poses,
                                         d_min=0.6 * cfg.seg_len, thr=cfg.trav_thr)
        if not cands:
            return self._random_segment(a, trav, src)
        est_occ = (a.cert.gate & (a.cert.mu > cfg.trav_thr)).reshape(w.ny, w.nx)
        model = a.model
        cert = getattr(a, "cert_plan", None) if self.cfg.split4 else a.cert
        if cert is None: cert = a.cert
        if cfg.planner in ("entropy", "entropy_u", "entropy_g", "lse") and prev_S:
            # virtual pooling: plan on own model conditioned on teammates' planned observations
            model = a.model.copy(); model.add(np.concatenate(prev_S, 0), np.zeros(sum(len(s) for s in prev_S)))
            mu2, sd2 = model.predict(self.PhiQ)
            cert = AgentCert(mu2, sd2, a.cert.gate, a.cert.wx, a.cert.a, a.cert.eps, a.cert.q, a.cert.q_scalar,
                             a.cert.e, a.cert.n_cal, a.cert.n_recal)
        best, best_score, best_P = None, -np.inf, None
        scored = []
        for c in cands:
            S = [self.lidars[a.i].predict_samples(est_occ, w.res, p, min(cfg.plan_beams, self.lidars[a.i].s.n_beams), cfg.plan_free_spacing) for p in c.poses]
            c.S = np.concatenate(S, 0) if S else np.zeros((0, 2))
            c.cells = affected_cells(c.S, w.nx, w.ny, w.res, 3 * cfg.ell) if len(c.S) else np.zeros(0, int)
            g, P = self.acq.score(c, model, self.feats, self.PhiQ, cert, undecided, others_commit,
                                  cfg.gate_r, w.nx, w.ny, w.res, self.rng)
            scored.append((c, g, P))
            s = g / max(c.length, 0.5) if cfg.planner != "random" else g
            if s > best_score:
                best, best_score, best_P = c, s, (P, c.cells, g)
        if return_all:
            return scored
        if cfg.lookahead > 1 and cfg.planner not in ("random", "lawnmower"):
            best = self._lookahead(a, scored, trav, undecided, others_commit, model, cert, est_occ) or best
            best_P = (None, best.cells, 0.0)
        a.plan_S = best.S
        a.plan_len = best.length
        a.plan_gain = best_P[2] if best_P is not None else 0.0
        a.plan_gain_obs = 0.0
        if best_P is not None and best_P[0] is not None:
            a.plan_P = best_P[:2]
            P, cells = best_P[:2]
            disc = 1.0 if others_commit is None else (1.0 - others_commit[cells])
            a.plan_gain_obs = float(np.sum(P * disc * w.obs[cells]))
        return best.exec_idx

    def _lookahead(self, a: Agent, scored, trav, undecided, others_commit, model, cert, est_occ):
        """Receding-horizon beam search over `lookahead` segments; only the first segment is executed.

        Each surviving beam carries a model conditioned on the observations its segments are predicted to collect,
        so a cell already covered earlier in the sequence contributes little later: the double counting is handled
        by the posterior itself rather than by bookkeeping. Candidates at depth > 1 are generated from the end of
        the previous segment with the same generator as depth 1. Scores are cumulative gain per metre, as in the
        myopic planner. Everything else (acquisition, candidate generator, execution) is unchanged, so this
        isolates the horizon.
        """
        cfg, w = self.cfg, self.world
        beams = [(g / max(c.length, 0.5), g, c.length, c, model, cert, c) for c, g, _ in scored]
        beams.sort(key=lambda b: -b[0]); beams = beams[:cfg.beam]
        for depth in range(1, cfg.lookahead):
            nxt = []
            for _, gsum, lsum, tail, m, ct, first in beams:
                m2 = m.copy(); m2.add(tail.S, np.zeros(len(tail.S)))          # labels are irrelevant to variance
                mu2, sd2 = m2.predict(self.PhiQ)
                gate2 = ct.gate | gate_from_samples(tail.S, w.nx, w.ny, w.res, cfg.gate_r)
                ct2 = AgentCert(mu2, sd2, gate2, ct.wx, ct.a, ct.eps, ct.q, ct.q_scalar, ct.e, ct.n_cal, ct.n_recal)
                src2 = int(tail.exec_idx[-1])
                cs, dist2, pred2 = make_candidates(self.gg, trav, src2, self.rng, cfg.n_cand, 0.6 * cfg.seg_len,
                                                   1.2 * cfg.seg_len, cfg.seg_len, cfg.plan_pose_spacing, cfg.plan_max_poses)
                if cfg.generator == "rrt":
                    cs = rrt_candidates(self.gg, trav, src2, self.rng, gate2, w.nx, w.ny, dist2, pred2,
                                        cfg.n_cand + cfg.n_front, cfg.seg_len, cfg.plan_pose_spacing, cfg.plan_max_poses) or cs
                else:
                    cs += frontier_candidates(self.gg, gate2, mu2, trav, w.nx, w.ny, dist2, pred2, src2, cfg.n_front,
                                              cfg.seg_len, cfg.plan_pose_spacing, cfg.plan_max_poses,
                                              d_min=0.6 * cfg.seg_len, thr=cfg.trav_thr)
                for c in cs:
                    S = [self.lidars[a.i].predict_samples(est_occ, w.res, p, min(cfg.plan_beams, self.lidars[a.i].s.n_beams),
                                                          cfg.plan_free_spacing) for p in c.poses]
                    c.S = np.concatenate(S, 0) if S else np.zeros((0, 2))
                    c.cells = affected_cells(c.S, w.nx, w.ny, w.res, 3 * cfg.ell) if len(c.S) else np.zeros(0, int)
                    g, _ = self.acq.score(c, m2, self.feats, self.PhiQ, ct2, undecided, others_commit,
                                          cfg.gate_r, w.nx, w.ny, w.res, self.rng)
                    nxt.append(((gsum + g) / max(lsum + c.length, 0.5), gsum + g, lsum + c.length, c, m2, ct2, first))
            if not nxt:
                break
            nxt.sort(key=lambda b: -b[0]); beams = nxt[:cfg.beam]
        return beams[0][6] if beams else None

    def _traversable(self, a: Agent, src: int, inflate: int | None = None):
        w = self.world
        if inflate is None:
            inflate = self.cfg.trav_inflate
        if self.cfg.guard in ("certified", "ltt"):
            # traverse only cells the fused certificate commits free now whose (region, staleness) class is admitted;
            # uncertified blocks (warm-up, and a probe after probe_after consecutive blocks without an admissible segment)
            # admit committed-free cells observed in the last block, and are accounted separately
            fresh = (a.stale_prev <= 1) if a.stale_prev is not None else np.zeros(w.M, bool)
            refused = self.cfg.guard == "ltt" and not np.isfinite(getattr(a, "ltt_lambda", np.nan))
            a.uncertified = (self.k < self.cfg.warmup or refused
                             or (self.cfg.probe_after > 0 and getattr(a, "stuck", 0) >= self.cfg.probe_after))
            h = fresh if a.uncertified else a.h_ok
            trav = (a.fused.committed & (a.fused.label < 0) & h)
        elif self.cfg.guard == "fresh":
            # heuristic: committed free and observed by the neighbourhood in the last block (the warm-up rule, uncalibrated)
            st = self._staleness(a); a.stale_prev = st
            trav = (a.fused.committed & (a.fused.label < 0) & (st <= 1))
        elif self.cfg.guard == "oracle":
            trav = ~self.world.occ.ravel()
        elif self.cfg.guard == "prob":
            pocc = 1.0 / (1.0 + np.exp(-(a.logodds if a.logodds is not None else np.zeros(w.M))))
            trav = a.cert.gate & (pocc <= self.cfg.alpha)
        elif self.cfg.guard in ("cp_track", "cp_track_oracle"):
            trav = a.cert.gate & (a.cert.mu <= self.cfg.trav_thr) & ~getattr(a, "forbid", np.zeros(w.M, bool))
        elif self.cfg.guard == "inflate":
            trav = a.cert.gate & traversable_from_belief(a.cert.mu, a.cert.gate, w.nx, w.ny, thr=self.cfg.trav_thr, inflate=max(inflate, 1))
        else:
            trav = traversable_from_belief(a.cert.mu, a.cert.gate, w.nx, w.ny, thr=self.cfg.trav_thr, inflate=inflate)
        if a.blocked:
            trav[list(a.blocked)] = False
        trav[src] = True
        # if the belief encloses the robot, relax: no inflation, then belief-free (bumps only)
        if inflate > 0 and not self._has_exit(trav, src):
            a.relaxed = 1
            return self._traversable(a, src, inflate=0)
        if inflate == 0 and not self._has_exit(trav, src):
            a.relaxed = 2
            if self.cfg.guard in ("certified", "ltt", "oracle", "inflate", "prob", "cp_track", "cp_track_oracle", "fresh"):
                return trav                       # no admissible segment: the robot idles this block
            trav = np.ones(w.M, bool)
            if a.blocked:
                trav[list(a.blocked)] = False
        return trav

    def _has_exit(self, trav: np.ndarray, src: int, d: float = 3.0) -> bool:
        dist, _ = self.gg.shortest(trav, src, limit=d * 1.5)
        return bool(np.any(np.isfinite(dist) & (dist >= d)))

    def _truncate(self, idx):
        if len(idx) < 2:
            return idx
        xy = self.gg.xy(idx)
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        return list(np.array(idx)[cum <= self.cfg.seg_len + 1e-9])

    def _random_segment(self, a, trav, src):
        cands, _, _ = make_candidates(self.gg, trav, src, self.rng, 4, 1.0, 1.2 * self.cfg.seg_len,
                                      self.cfg.seg_len, self.cfg.plan_pose_spacing, self.cfg.plan_max_poses)
        return cands[self.rng.integers(len(cands))].exec_idx if cands else [src]

    def _plan_lawn(self, a, trav, src):
        w = self.world
        for _ in range(len(a.lawn_wp)):
            wp = a.lawn_wp[a.lawn_i % len(a.lawn_wp)]
            dst = int(w.flat_of(wp)[0])
            dist, pred = self.gg.shortest(trav, src, limit=80.0)
            if np.isfinite(dist[dst]) and dist[dst] > 0.5:
                idx = self.gg.path(pred, src, dst)
                if dist[dst] <= self.cfg.seg_len:
                    a.lawn_i += 1
                return self._truncate(idx)
            a.lawn_i += 1
        return self._random_segment(a, trav, src)

    def plan_joint2(self, k: int):
        """N=2, certified: evaluate greedy and joint allocation on the same candidate sets.
        Logs the ratio of the joint objective (product-form gain per total metre) achieved by
        greedy to the joint optimum; executes greedy or joint according to cfg.alloc."""
        w = self.world
        order = [k % 2, (k + 1) % 2]
        a0, a1 = self.agents[order[0]], self.agents[order[1]]
        s0 = self.plan_agent(a0, 0, None, [], a0.fused.undecided, return_all=True)
        s1 = self.plan_agent(a1, 1, None, [], a1.fused.undecided, return_all=True)
        if not isinstance(s0, list) or not isinstance(s1, list):
            return None
        def dense(c, P):
            v = np.zeros(w.M)
            if P is not None: v[c.cells] = P
            return v
        D0 = [(c, g, dense(c, P)) for c, g, P in s0]; D1 = [(c, g, dense(c, P)) for c, g, P in s1]
        def joint_val(x, y):
            c0, g0, v0 = x; c1, g1, v1 = y
            return (g0 + g1 - float(np.sum(v0 * v1))) / max(c0.length + c1.length, 1.0)
        # greedy
        i0 = int(np.argmax([g / max(c.length, 0.5) for c, g, _ in D0]))
        v0 = D0[i0][2]
        i1 = int(np.argmax([(g - float(np.sum(v0 * v))) / max(c.length, 0.5) for c, g, v in D1]))
        greedy = (i0, i1); gv = joint_val(D0[i0], D1[i1])
        # joint
        best, bv = None, -np.inf
        for i, x in enumerate(D0):
            for j, y in enumerate(D1):
                v = joint_val(x, y)
                if v > bv: best, bv = (i, j), v
        self._ratio_log = gv / bv if bv > 0 else 1.0
        pick = greedy if self.cfg.alloc == "greedy" else best
        plans = {}
        for a, D, i in ((a0, D0, pick[0]), (a1, D1, pick[1])):
            c, g, _ = D[i]
            a.plan_S, a.plan_len, a.plan_gain, a.plan_gain_obs = c.S, c.length, g, float(np.sum(D[i][2] * w.obs))
            plans[a.i] = c.exec_idx
        return plans

    def plan(self, k: int):
        cfg, w = self.cfg, self.world
        if cfg.N == 2 and cfg.planner == "certified" and cfg.alloc in ("greedy", "joint") and getattr(cfg, "check_joint", False):
            p = self.plan_joint2(k)
            if p is not None:
                return p
        order = [(k + j) % cfg.N for j in range(cfg.N)]
        others = np.zeros(w.M) if cfg.planner in ("certified", "certified_r", "certified_m") else None
        prev_S = []
        plans = {}
        self._claimed = []
        for pos, i in enumerate(order):
            a = self.agents[i]
            a.plan_P = None
            fp = getattr(a, "fused_plan", None) if cfg.split4 else None
            fsrc = fp if fp is not None else a.fused
            target = fsrc.nodata if cfg.planner in ("entropy_g", "frontier_g") else fsrc.undecided
            plans[i] = self.plan_agent(a, pos, others if pos > 0 else None, prev_S, target)
            if cfg.planner in ("certified", "certified_r", "certified_m") and a.plan_P is not None:
                P, cells = a.plan_P
                others[cells] = 1.0 - (1.0 - others[cells]) * (1.0 - P)
            if a.plan_S is not None and len(a.plan_S):
                prev_S.append(a.plan_S)
        return plans

    # ---- execution --------------------------------------------------------------
    def execute(self, a: Agent, path_idx: list[int], remaining: float = np.inf):
        cfg, w = self.cfg, self.world
        if len(path_idx) < 2:
            # idle block: turn a quarter and scan in place (train/calibration alternating) so the robot looks around
            a.stuck = getattr(a, "stuck", 0) + 1
            a.pose = a.pose.copy(); a.pose[2] += np.pi / 2
            self.scan_at(a, a.pose, to_train=(self.k % 2 == 0))
            return
        xy = self.gg.xy(path_idx)
        # stop before entering a truly occupied cell; remember the bump
        occ = w.occ.ravel()[np.array(path_idx)]
        # per-cell accounting of the admitted segment (Prop. A quantity): occupied at traversal time, by staleness =
        # blocks since the cell was last sampled by any robot in the neighbourhood (buckets 1, 2-3, 4-7, 8-15, 16+, never)
        lo = self._last_obs(a)[np.array(path_idx)]
        st = np.where(lo < 0, 10 ** 6, self.k - lo)
        bk = np.clip(np.floor(np.log2(np.maximum(st, 1))).astype(int), 0, 4); bk[lo < 0] = 5
        if not getattr(a, "uncertified", False):
            np.add.at(a.plan_cells, bk, 1); np.add.at(a.plan_occ, bk, occ.astype(int))
        # traversal pairs: cells the robot enters this block, in order, up to and including the first occupied one
        # (chi = 1 there); cells beyond a stop or beyond the budget are not entered and give no pair
        stop_i = int(np.argmax(occ)) if occ.any() else len(path_idx)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(np.vstack([a.pose[None, :2], xy]), axis=0), axis=1))])
        n_reach = int(np.searchsorted(cum[1:], remaining, side="right"))          # cells reachable within the budget
        n_enter = min(stop_i + 1, n_reach, len(path_idx))
        for j in range(n_enter):
            a.tpairs.append((self.k, int(path_idx[j]), float(occ[j]), int(st[j])))
        if n_enter and occ[:n_enter].any():
            a.miss_xy = getattr(a, "miss_xy", []) + [tuple(q) for q, o in zip(xy[:n_enter], occ[:n_enter]) if o]
        unc = getattr(a, "uncertified", False)
        if unc:
            a.unc_cells = getattr(a, "unc_cells", 0) + n_enter; a.unc_occ = getattr(a, "unc_occ", 0) + int(occ[:n_enter].sum())
            a.unc_blocks = getattr(a, "unc_blocks", 0) + 1
        else:
            a.trav_cells = getattr(a, "trav_cells", np.zeros(6, int)); a.trav_occ = getattr(a, "trav_occ", np.zeros(6, int))
            np.add.at(a.trav_cells, bk[:n_enter], 1); np.add.at(a.trav_occ, bk[:n_enter], occ[:n_enter].astype(int))
        a.stuck = 0
        if occ.any():
            stop = int(np.argmax(occ))
            a.blocked.add(int(path_idx[stop]))
            if self.hazards:
                # entering a cell occupied now: a hazard collision if the cell is a hazard cell, else a static bump
                if self.world.occ.ravel()[path_idx[stop]] and not self.static_occ.ravel()[path_idx[stop]]:
                    a.collisions = getattr(a, "collisions", 0) + 1
                else:
                    a.bumps = getattr(a, "bumps", 0) + 1
            xy = xy[:max(stop, 1)]
        poses = poses_along(np.vstack([a.pose[None, :2], xy]), cfg.pose_spacing)
        if len(poses) == 0:
            # blocked in place: still take a scan so the belief can update
            self.scan_at(a, a.pose, to_train=True)
            return
        n_train = int(np.ceil(cfg.train_frac * len(poses)))
        start = a.path_len
        for j, p in enumerate(poses):
            if a.path_len - start + np.linalg.norm(p[:2] - a.pose[:2]) > remaining + 1e-6 and j > 0:
                break
            self.scan_at(a, p, to_train=(j < n_train))
            a.steps = getattr(a, "steps", 0) + 1
            a.path_len += np.linalg.norm(p[:2] - a.pose[:2])
            a.pose = p.copy()
            a.traj.append(p.copy())

    # ---- main loop -------------------------------------------------------------------
    def run(self):
        t0 = time.time()
        self.bootstrap()
        for k in range(self.cfg.K + 1):
            self.k = k
            if self.cfg.drift_agents:
                # pose-estimate drift: a random walk, so the error is correlated in time and accumulates
                for a in self.agents:
                    d = self.cfg.drift_agents[a.i] if a.i < len(self.cfg.drift_agents) else 0.0
                    if d > 0:
                        if a.pose_bias is None:
                            a.pose_bias = np.zeros(3)
                        a.pose_bias += self.rng.normal(0, [d, d, d / 10.0])
            if k > 0 and not self.cfg.hazards_move_after_plan:
                self.move_hazards()
            self.certify()
            self.record(k, t0)
            active = [a for a in self.agents if a.path_len < self.cfg.budget - 1e-6]
            if k == self.cfg.K or not active:
                break
            plans = self.plan(k)
            if self.cfg.hazards_move_after_plan:
                self.move_hazards()
            self._block_samples = {}
            for i, idx in plans.items():
                a = self.agents[i]
                if a.path_len < self.cfg.budget - 1e-6:
                    if len(idx) < 2:
                        a.idle = getattr(a, "idle", 0) + 1
                    self.execute(a, idx, self.cfg.budget - a.path_len)
        out = dict(config=asdict(self.cfg), log=self.log,
                   traj=[np.array(a.traj) for a in self.agents], occ_frac=float(self.world.occ.mean()))
        if self.cfg.save_maps:
            out["maps"] = [dict(label=a.fused.label, committed=a.fused.committed, undecided=a.fused.undecided,
                                empty=a.fused.empty, nodata=a.fused.nodata, mu=a.cert.mu, sigma=a.cert.sigma,
                                gate=a.cert.gate, e=a.cert.e,
                                pose_bias=(None if a.pose_bias is None else a.pose_bias.copy()))
                           for a in self.agents]
            out["occ"] = self.world.occ
        return out
