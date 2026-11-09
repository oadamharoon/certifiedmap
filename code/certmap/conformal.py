"""Split conformal calibration, e-values, and neighbourhood fusion.

Implements Algorithm 1 of Mahajan et al. (2026) on a query grid, with an
optional covariate-shift weighting (Tibshirani et al. 2019) that re-targets
each agent's certificate from its calibration law to the uniform law over its
gated region.

Notation (matches the paper):
  A(x,y) = |y - mu(x)| / sigma(x)        nonconformity score
  p(x,y)                                  (weighted) split-conformal p-value
  a_i, eps_i                              recalibrated inner level / measured miscoverage
  e_i(x,y) = g_i(x) 1{p<=a_i}/eps_i gate_i(x)
  e^f(x,y) = mean_{j in N[i]} e_j(x,y);  Gamma(x) = {y : e^f(x,y) < 1/alpha}
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------
# weighted split conformal on a finite score set
# --------------------------------------------------------------------------
@dataclass
class ScoreSet:
    """Sorted calibration scores with weights; supports vectorised p-values/thresholds."""
    s: np.ndarray       # ascending scores
    w: np.ndarray       # weights aligned with s
    tail: np.ndarray    # tail[j] = sum_{m>=j} w_m  (weight of scores >= s[j])
    W: float

    @classmethod
    def build(cls, scores: np.ndarray, weights: np.ndarray | None = None):
        o = np.argsort(scores)
        s = scores[o]
        w = np.ones_like(s) if weights is None else weights[o]
        tail = np.cumsum(w[::-1])[::-1]
        return cls(s, w, tail, float(w.sum()))

    def pvalue(self, A: np.ndarray, wx: np.ndarray) -> np.ndarray:
        """p(x,y) = (sum_{A_m >= A} w_m + w_x) / (W + w_x)."""
        j = np.searchsorted(self.s, A, side="left")      # first index with s >= A
        tw = np.where(j < len(self.s), self.tail[np.minimum(j, len(self.s) - 1)], 0.0)
        wx = np.asarray(wx, float)
        return np.where(np.isfinite(wx), (tw + wx) / (self.W + wx), 1.0)

    def threshold(self, a: float, wx: np.ndarray) -> np.ndarray:
        """q(x) such that p(x,y) <= a  <=>  A(x,y) > q(x). Returns +inf where impossible.

        p(A) <= a  <=>  tail(A) <= a (W + w_x) - w_x =: c(x). With j* the smallest
        index with tail[j*] <= c(x), the condition holds for all A > s[j*-1].
        """
        wx = np.asarray(wx, float)
        c = np.where(np.isfinite(wx), a * (self.W + wx) - wx, -np.inf)
        # tail is non-increasing; find smallest j with tail[j] <= c
        j = np.searchsorted(-self.tail, -c, side="left")
        q = np.full(len(wx), np.inf)
        ok = c >= 0                       # otherwise even A = +inf gives p > a
        jj = np.clip(j, 1, len(self.s))   # j == len(s): flag iff A > max score
        q[ok] = self.s[jj[ok] - 1]
        return q


# --------------------------------------------------------------------------
# per-agent certificate
# --------------------------------------------------------------------------
@dataclass
class AgentCert:
    """Everything agent i broadcasts / needs for fusion, on the query grid."""
    mu: np.ndarray            # (M,)   clipped predictive mean
    sigma: np.ndarray         # (M,)   predictive std (incl. noise)
    gate: np.ndarray          # (M,)   bool, observation gate
    wx: np.ndarray            # (M,)   covariate-shift weight of a test point at each cell
    a: float                  # recalibrated inner level
    eps: float                # measured local miscoverage at a  (denominator of e-value)
    q: np.ndarray             # (M,)   per-cell score threshold: flag y iff A(x,y) > q(x)
    q_scalar: float           # threshold at the median gated weight (used by the planner)
    e: np.ndarray             # (M,2)  e-values for labels [-1,+1]
    n_cal: int
    n_recal: int
    p_hat: float = np.nan     # weighted fraction of well-observed recalibration labels contradicting sign(mu)
    theta_map: np.ndarray = None  # (M,) local contradiction-rate estimate from the calibration fold (reliability-aware planning)
    group: np.ndarray = None      # (M,) int reliability class of each cell (Mondrian certificate); None = single class
    q_group: np.ndarray = None    # (G,) per-class scalar thresholds at the median weight (planner default for unexplored cells)
    n_eff: float = np.nan         # self-normalized effective recalibration size (sum w)^2 / sum w^2
    n_w: float = np.nan           # Bernstein effective size (sum w)/max w, the n the weighted proof needs
    eps_prov: float = np.nan      # upper confidence bound on the true miscoverage at level a
    n_ind: float = np.nan         # (sum w)/max scan weight: the n of the proposition, scan-level summands
    n_scan: float = np.nan        # number of scans represented in the recalibration fold
    m_bar: float = np.nan         # mean number of recalibration labels per scan


def density_on_grid(X: np.ndarray, nx: int, ny: int, res: float, bandwidth: float) -> np.ndarray:
    """Histogram of sample locations on the grid, smoothed by a box of width ``bandwidth``
    (bandwidth <= res gives the raw per-cell histogram), returned as a density."""
    ix = np.clip(np.floor(X[:, 0] / res).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(X[:, 1] / res).astype(int), 0, ny - 1)
    h = np.zeros((ny, nx))
    np.add.at(h, (iy, ix), 1.0)
    k = int(round(bandwidth / res))
    if k > 1:
        h = ndimage.uniform_filter(h, size=k, mode="constant")
    return (h / max(h.sum(), 1e-12) / res ** 2).ravel()


def gate_from_samples(X: np.ndarray, nx: int, ny: int, res: float, r: float) -> np.ndarray:
    """gate(x) = 1{dist(x, X) < r} evaluated at cell centres."""
    if len(X) == 0:
        return np.zeros(nx * ny, bool)
    ix = np.clip(np.floor(X[:, 0] / res).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(X[:, 1] / res).astype(int), 0, ny - 1)
    occ = np.zeros((ny, nx), bool)
    occ[iy, ix] = True
    d = ndimage.distance_transform_edt(~occ) * res
    return (d < r).ravel()


def score_fn(kind: str, y, mu, sd):
    """Nonconformity score. 'residual': |y-mu|/sigma (Mahajan et al.).
    'margin': -y*mu/sigma; with the threshold clipped at 0 the induced sets are nested
    (never empty) and a cell is committed iff |mu| > q sigma."""
    if kind == "residual":
        return np.abs(y - mu) / sd
    if kind == "margin":
        return -y * mu / sd
    raise ValueError(kind)


def calibrate_agent(mu_grid, sigma_grid, gate, X_train, X_cal, y_cal, mu_cal, sd_cal,
                    X_rec, y_rec, mu_rec, sd_rec, *, beta: float, weighted: bool,
                    nx: int, ny: int, res: float, kde_bw: float = 0.5, lam: float = 0.0,
                    p_floor: float | None = None, score: str = "residual",
                    eps_ucb: float = 0.0, X_rs=None, y_rs=None, mu_rs=None, sd_rs=None,
                    denom_a: bool = False, denom_delta: float = 0.0, scan_rec=None,
                    w_clip: float = 0.0) -> AgentCert:
    """Phase 1-2 of Algorithm 1 for one agent.

    beta      per-neighbourhood miscoverage budget s*alpha/d_i
    weighted  if True, weights w(x) = u(x)/rho(x) re-target the certificate to the
              uniform law u over the gated region; rho is a KDE of the calibration
              locations. If False, all weights are 1 (Mahajan et al.'s protocol).
    """
    M = nx * ny
    # ---- weights --------------------------------------------------------
    if weighted and len(X_cal) > 0:
        rho = density_on_grid(np.concatenate([X_cal, X_rec], 0), nx, ny, res, kde_bw)
        wx = np.zeros(M)
        g = gate
        u = 1.0 / max(g.sum(), 1)
        with np.errstate(divide="ignore"):
            wx[g] = u / rho[g]                               # +inf where there is no calibration support
        fin = g & np.isfinite(wx)
        wx[fin] /= np.median(wx[fin]) if fin.any() else 1.0  # scale-free; keeps numbers O(1)
        if w_clip > 0 and fin.any():
            # Cap the importance-weight tail at a quantile. n_tilde = (sum w)/max w is set by the largest
            # weight, so the tail alone decides the finite-sample term of Prop. 4.3; capping it trades a
            # slightly stronger tilt of the certified law (Sec. 4.1) for a much smaller correction.
            wx[fin] = np.minimum(wx[fin], np.quantile(wx[fin], w_clip))
        def w_at(X):
            ix = np.clip(np.floor(X[:, 0] / res).astype(int), 0, nx - 1)
            iy = np.clip(np.floor(X[:, 1] / res).astype(int), 0, ny - 1)
            return wx[iy * nx + ix]
        w_cal, w_rec = w_at(X_cal), w_at(X_rec)
        # calibration points outside the gated region carry no weight (target law has no mass there)
        w_cal = np.where(np.isfinite(w_cal), np.maximum(w_cal, 1e-9), 1e-9)
        w_rec = np.where(np.isfinite(w_rec), np.maximum(w_rec, 1e-9), 1e-9)
    else:
        wx = np.ones(M)
        w_cal, w_rec = np.ones(len(X_cal)), np.ones(len(X_rec))

    # ---- scores and p-values on the recalibration fold --------------------
    A_cal = score_fn(score, y_cal, mu_cal, sd_cal)
    ss = ScoreSet.build(A_cal, w_cal)
    A_rec = score_fn(score, y_rec, mu_rec, sd_rec)
    # Sample-split recalibration: with a separate score set (fold B) the recalibrated level does not depend on
    # the fold that carries the claim, which is what lets the claim fold stay fresh under conditioning.
    if X_rs is not None and len(X_rs):
        if weighted and len(X_cal) > 0:
            w_rs = w_at(X_rs)
            w_rs = np.where(np.isfinite(w_rs), np.maximum(w_rs, 1e-9), 1e-9)   # same clamp as the other folds
        else:
            w_rs = np.ones(len(X_rs))
        ss_r = ScoreSet.build(score_fn(score, y_rs, mu_rs, sd_rs), w_rs)
    else:
        ss_r = ss
    p_rec = ss_r.pvalue(A_rec, w_rec)

    # ---- recalibrate a and eps on the held-out fold (Remark 2) -------------
    # a = sup{a : eps_hat(a) <= beta}, eps_hat(a) = weighted fraction of recal points with p <= a
    o = np.argsort(p_rec)
    Wr = max(w_rec.sum(), 1e-12)
    cum = np.cumsum(w_rec[o]) / Wr
    n_eff = Wr ** 2 / max(np.sum(w_rec ** 2), 1e-12)
    if eps_ucb > 0:
        # Additive Hoeffding margin. Superseded by denom_delta (Lemma 1): additive margins do not shrink with
        # beta, so at beta = s*alpha/d_i they exceed the whole budget once d_i >= 3. Kept for the E7 ablation.
        cum = cum + eps_ucb / np.sqrt(max(n_eff, 1.0))
    ok = cum <= beta
    if ok.any():
        j = np.where(ok)[0][-1]
        a = float(p_rec[o][j])
        eps = float(cum[j])
    else:
        a, eps = 0.0, 1.0
    n_rec = len(X_rec)
    eps = max(eps, 1.0 / (n_rec + 1))           # eps > 0 (a p-value cannot be below 1/(T+1))
    if p_floor is not None:
        eps = max(eps, p_floor)

    # Effective size the recalibration bound is entitled to. Two corrections separate it from the nominal fold.
    # (i) The weighted Bernstein step needs Var <= m / n with n = (sum w)/(max summand weight), not the
    #     self-normalized n_eff = (sum w)^2/sum w^2, because bounding sum (w_c/W)^2 m_c needs the largest weight.
    # (ii) One scan's pose error is shared by all of its endpoints, so the independence unit of Assumption 2 is
    #      the scan, not the label. Summing each scan's weighted misses into one bounded summand keeps the
    #      summands independent without any model for the within-scan dependence, and the max is then taken over
    #      scan totals. This is exact, unlike a cluster design effect, which would need a random-effects model.
    if scan_rec is not None and len(scan_rec) == len(w_rec) and len(w_rec):
        uq, inv, cnt = np.unique(scan_rec, return_inverse=True, return_counts=True)
        Wc = np.bincount(inv, weights=w_rec)       # total recalibration weight of each scan
        m_bar, n_scan = float(cnt.mean()), float(len(uq))
    else:
        Wc, m_bar, n_scan = w_rec, 1.0, float(len(w_rec))
    n_w = Wr / max(float(np.max(w_rec)) if len(w_rec) else 1.0, 1e-12)
    n_ind = Wr / max(float(np.max(Wc)) if len(Wc) else 1.0, 1e-12)

    # With probability 1-delta the true weighted average miscoverage at the selected level is at most z/n_ind,
    # z the larger root of (z - j*)^2 = 2 z log(1/delta) and j* = eps * n_ind. The multiplicative Chernoff lower
    # tail holds for independent summands bounded in [0, b], not only Bernoulli ones, so scan totals need no
    # extra Bernstein term: rescaling by b = 1/n_ind gives exactly this root. Dividing the e-value by z/n_ind
    # makes E[e] <= 1 outright, at the cost of the factor z/j*; dividing by the measured eps leaves that factor
    # on the coverage level instead.
    L = float(np.log(1.0 / denom_delta)) if denom_delta > 0 else 0.0
    js = eps * n_ind
    bq = 2.0 * js + 2.0 * L
    eps_prov = min(1.0, 0.5 * (bq + np.sqrt(max(bq * bq - 4.0 * js * js, 0.0))) / max(n_ind, 1.0))

    # ---- per-cell thresholds and e-values ----------------------------------
    q = ss.threshold(a, wx) if a > 0 else np.full(M, np.inf)
    gm = gate & np.isfinite(wx) & (wx > 0)
    q_scalar = float(ss.threshold(a, np.array([np.median(wx[gm])]))[0]) if (a > 0 and gm.any()) else np.inf
    if score == "margin":
        # clipping the threshold at 0 enlarges the sets (validity preserved) and makes them nested
        q = np.maximum(q, 0.0); q_scalar = max(q_scalar, 0.0)
    g_att = np.exp(-lam * sigma_grid)
    e = np.zeros((M, 2))
    # denom_a: divide by the recalibrated level itself. Under the sample split the conditional miscoverage at
    # level a is at most a, so this makes E[e] <= 1 without Assumption 3; dividing by the measured eps needs it.
    den = max(a, 1.0 / (len(X_rec) + 1)) if denom_a else (eps_prov if denom_delta > 0 else eps)
    for k, y in enumerate((-1.0, 1.0)):
        A = score_fn(score, y, mu_grid, sigma_grid)
        e[:, k] = g_att * (A > q) / den * gate
    # certifiability forecast: attainable label-error rate at well-observed points (sigma near its floor)
    well = sd_rec <= np.quantile(sd_rec, 0.5) if len(sd_rec) else np.zeros(0, bool)
    if well.any():
        dis = (np.sign(mu_rec[well]) != np.sign(y_rec[well])).astype(float)
        p_hat = float(np.sum(w_rec[well] * dis) / max(np.sum(w_rec[well]), 1e-12))
    else:
        p_hat = np.nan
    return AgentCert(mu_grid, sigma_grid, gate, wx, a, eps, q, q_scalar, e, len(X_cal), n_rec, p_hat,
                     n_eff=float(n_eff), n_w=float(n_w), eps_prov=float(eps_prov),
                     n_ind=float(n_ind), n_scan=float(n_scan), m_bar=float(m_bar))


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------
@dataclass
class FusedMap:
    Gamma: np.ndarray       # (M,2) bool: label -1 / +1 admissible
    committed: np.ndarray   # (M,) bool |Gamma|=1
    undecided: np.ndarray   # (M,) bool Gamma = {-1,+1}
    empty: np.ndarray       # (M,) bool Gamma = {}
    nodata: np.ndarray      # (M,) bool no active neighbour
    label: np.ndarray       # (M,) committed label (0 where not committed)
    e_fused: np.ndarray     # (M,2)
    Gamma_fcr: np.ndarray = None    # (M,2) e-BH set with false-commitment-rate control
    committed_fcr: np.ndarray = None
    label_fcr: np.ndarray = None
    thr_fcr: float = np.inf         # e-BH rejection threshold on the fused e-value


def ebh_threshold(e_all: np.ndarray, alpha: float) -> float:
    """e-BH (Wang & Ramdas 2022) over K = len(e_all) hypotheses: reject the k* largest
    e-values, k* = max{k : e_(k) >= K/(alpha k)}. Returns the rejection threshold (inf if none)."""
    K = len(e_all)
    es = np.sort(e_all)[::-1]
    k = np.arange(1, K + 1)
    ok = es >= K / (alpha * k)
    if not ok.any():
        return np.inf
    kstar = int(np.max(k[ok]))
    return K / (alpha * kstar)


def fuse(certs: list[AgentCert], alpha: float, mask: np.ndarray | None = None) -> FusedMap:
    d = len(certs)
    ef = sum(c.e for c in certs) / d
    Gamma = ef < 1.0 / alpha
    n = Gamma.sum(1)
    committed = n == 1
    label = np.where(committed, np.where(Gamma[:, 1], 1.0, -1.0), 0.0)
    nodata = ~np.any(np.stack([c.gate for c in certs], 0), 0)
    F = FusedMap(Gamma, committed, n == 2, n == 0, nodata, label, ef)
    # e-BH over the 2M (cell, label) hypotheses "the true label of x is y" restricted to the
    # query domain 'mask': controls E[#true labels rejected / #rejections] <= alpha/2 on the
    # uniform law (the conformal e-values are valid on average over cells).
    m = np.ones(ef.shape[0], bool) if mask is None else mask
    thr = ebh_threshold(ef[m].ravel(), alpha)
    G2 = ~(ef >= thr)
    n2 = G2.sum(1)
    F.Gamma_fcr, F.committed_fcr = G2, n2 == 1
    F.label_fcr = np.where(F.committed_fcr, np.where(G2[:, 1], 1.0, -1.0), 0.0)
    F.thr_fcr = thr
    return F


def local_contradiction_map(X: np.ndarray, y: np.ndarray, mu_at: np.ndarray, nx: int, ny: int, res: float,
                            box_m: float = 3.0, prior_a: float = 0.1, prior_n: float = 2.0) -> np.ndarray:
    """Local contradiction rate theta_hat(x): the fraction of calibration labels that contradict the sign of the
    model mean at their own location, counted on the grid and box-smoothed over ``box_m`` metres, with a weak
    Beta prior. In a region whose labels are uninformative the rate approaches 1/2; in clean regions it is the
    model's own boundary sign-error rate."""
    from scipy import ndimage
    M = nx * ny
    ix = np.clip(np.floor(X[:, 0] / res).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(X[:, 1] / res).astype(int), 0, ny - 1)
    flat = iy * nx + ix
    tot = np.bincount(flat, minlength=M).astype(float).reshape(ny, nx)
    con = np.bincount(flat[y * np.sign(mu_at) < 0], minlength=M).astype(float).reshape(ny, nx)
    k = max(int(round(box_m / res)), 1)
    tot = ndimage.uniform_filter(tot, size=k, mode="constant") * k * k
    con = ndimage.uniform_filter(con, size=k, mode="constant") * k * k
    return ((con + prior_a) / (tot + prior_n)).ravel()


def calibrate_agent_grouped(mu_grid, sigma_grid, gate, X_train, X_cal, y_cal, mu_cal, sd_cal,
                            X_rec, y_rec, mu_rec, sd_rec, *, group_cell: np.ndarray, group_cal: np.ndarray,
                            group_rec: np.ndarray, n_groups: int, min_n: int = 20, **kw) -> AgentCert:
    """Mondrian (group-conditional) certificate: split conformal is run separately within each reliability
    class, each with the full budget beta, so the miscoverage bound holds within every class. The taxonomy
    (group_*) must be measurable with respect to the training data only. A class with fewer than ``min_n``
    calibration or recalibration samples commits nothing."""
    M = len(mu_grid)
    q = np.full(M, np.inf); e = np.zeros((M, 2)); wx = np.zeros(M); q_group = np.full(n_groups, np.inf)
    a0 = eps0 = None; p_hat = np.nan; n_cal = n_rec = 0
    for g in range(n_groups):
        cells = group_cell == g
        sc, sr = group_cal == g, group_rec == g
        if sc.sum() < min_n or sr.sum() < min_n or not (gate & cells).any():
            continue
        c = calibrate_agent(mu_grid, sigma_grid, gate & cells, X_train, X_cal[sc], y_cal[sc], mu_cal[sc], sd_cal[sc],
                            X_rec[sr], y_rec[sr], mu_rec[sr], sd_rec[sr], **kw)
        q[cells] = c.q[cells]; e[cells] = c.e[cells]; wx[cells] = c.wx[cells]; q_group[g] = c.q_scalar
        n_cal += c.n_cal; n_rec += c.n_recal
        if g == 0:
            a0, eps0, p_hat = c.a, c.eps, c.p_hat
    a0 = 0.0 if a0 is None else a0; eps0 = 1.0 if eps0 is None else eps0
    return AgentCert(mu_grid, sigma_grid, gate, wx, a0, eps0, q, float(q_group[0]), e, n_cal, n_rec, p_hat,
                     None, group_cell.astype(int), q_group)
