"""E48: the clip-level objective curve, computed offline from captured calibration inputs.

The self-selecting rule picks the clip quantile q that minimises the bound it can certify,
    bound(q) = alpha * (1 + eta(q)) * (1 + zeta_c(q)) / (1 - zeta_c(q)) + delta,
where eta(q) is Prop. 4.3's finite-sample correction and zeta_c(q) is the tilt clipping induces by
re-targeting the certified law from u to one proportional to w_c * rho. Both are functions of the weights
the robot already holds, so no sweep over simulations is needed to choose q: the estimation error of
rho-hat enters every candidate as the same q-independent factor and cannot move the argmin.

This script captures the final-block calibration inputs of a few runs and replays calibrate_agent over a
grid of q, so the whole curve costs a handful of runs rather than a sweep.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from multiprocessing import Pool
from e44_provable_denom import CONDS

GRID = [0.0, 0.995, 0.99, 0.98, 0.97, 0.95, 0.925, 0.90, 0.875, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40]
DELTA, ALPHA = 0.01, 0.05
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def eta_exact(js, L):
    js = float(js)
    if js <= 0:
        return np.inf
    a = js + L
    return (a + np.sqrt(max(a * a - js * js, 0.0))) / js - 1


def tilt(w0, wq, gate):
    """Relative deviation of the law clipping actually certifies from the query law u.

    The certified law is proportional to w_c * rho = (w_c/w) * u * (rho/rho-hat); the observable factor is
    r = w_c/w, and u is uniform on the gated region, so E_u[r] is its unweighted mean there.
    """
    m = gate & np.isfinite(w0) & (w0 > 0) & np.isfinite(wq)
    if not m.any():
        return np.nan, np.nan
    r = wq[m] / w0[m]
    Z = float(r.mean())
    if Z <= 0:
        return np.nan, np.nan
    dev = np.abs(r / Z - 1.0)
    return float(dev.max()), float(np.quantile(dev, 0.99))


def one(cfg):
    import certmap.sim as S
    from certmap.conformal import calibrate_agent as real_cal
    from certmap.sim import Simulation, Config
    cap = collections.deque(maxlen=cfg["N"])

    def wrapper(*a, **k):
        if "w_clip" in k:
            cap.append((a, dict(k)))
        return real_cal(*a, **k)

    S.calibrate_agent = wrapper
    try:
        Simulation(Config(**cfg)).run()
    finally:
        S.calibrate_agent = real_cal

    rows = []
    for ai, (a, k) in enumerate(cap):
        base = real_cal(*a, **{**k, "w_clip": 0.0})
        for q in GRID:
            c = base if q == 0.0 else real_cal(*a, **{**k, "w_clip": q})
            js = c.eps * c.n_w
            e = eta_exact(js, np.log(cfg["N"] / DELTA))
            zs, z99 = tilt(base.wx, c.wx, c.gate)
            def bnd(z):
                if not np.isfinite(z) or z >= 1:
                    return np.inf
                return ALPHA * (1 + e) * (1 + z) / (1 - z) + DELTA
            rows.append(dict(seed=cfg["seed"], N=cfg["N"], agent=ai, q=q, n_w=c.n_w, eps=c.eps,
                             jstar=js, eta1=1 + e, zeta_sup=zs, zeta_p99=z99,
                             bound_notilt=ALPHA * (1 + e) + DELTA,
                             bound_sup=bnd(zs), bound_p99=bnd(z99)))
    return rows


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    B3, _ = CONDS[0]
    cfgs = [dict(B3, seed=s, planner="certified", weighted=True) for s in range(seeds)]
    print(f"[e48] {len(cfgs)} runs x {len(GRID)} clip levels, {workers} workers", flush=True)
    with Pool(workers) as p:
        out = [r for rs in p.map(one, cfgs) for r in rs]
    df = pd.DataFrame(out)
    os.makedirs(RESULTS, exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "e48_clip_curve.csv"), index=False)
    print(f"[e48] wrote results/e48_clip_curve.csv ({len(df)} rows)")
