"""E49: is the max-weight functional in Prop. 4.3 loose, or worst-case tight?

Prop. 4.3's Chernoff step needs the summands bounded by b = max_g p_g = 1/n~. A Bernstein step would instead
need V = sum_g p_g^2 Var(m_g) <= sum_g p_g^2 E[m_g]. Bounding that by (max p_g) * sum p_g E[m_g] = eps/n~
recovers the current form, and is tight exactly when the misses sit on the heaviest-weight points. If instead
the misses are spread, sum p_g^2 E[m_g] is nearer eps/n_eff, which is 8x smaller.

This measures the unbiased estimator V_hat = sum_g p_g^2 m_g at the selected level a, against both proxies.
No behaviour changes: the correction is charged on the level, not in the denominator, so nothing is re-run.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from multiprocessing import Pool
from e44_provable_denom import CONDS

DELTA, ALPHA = 0.01, 0.05
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def probe(a, k, N):
    """Recompute the recalibration fold's weight/miss structure for one captured call."""
    from certmap.conformal import score_fn, ScoreSet, density_on_grid
    import certmap.conformal as C
    cert = C.calibrate_agent(*a, **{**k, "w_clip": 0.0})
    # rebuild w_rec and p_rec exactly as calibrate_agent does
    (mu_grid, sigma_grid, gate, X_train, X_cal, y_cal, mu_cal, sd_cal, X_rec, y_rec, mu_rec, sd_rec) = a
    nx, ny, res = k["nx"], k["ny"], k["res"]
    def w_at(X):
        ix = np.clip(np.floor(X[:, 0] / res).astype(int), 0, nx - 1)
        iy = np.clip(np.floor(X[:, 1] / res).astype(int), 0, ny - 1)
        return cert.wx[iy * nx + ix]
    if k["weighted"]:
        w_cal = np.maximum(np.nan_to_num(w_at(X_cal), nan=1e-9, posinf=1e-9), 1e-9)
        w_rec = np.maximum(np.nan_to_num(w_at(X_rec), nan=1e-9, posinf=1e-9), 1e-9)
    else:
        w_cal = np.ones(len(X_cal)); w_rec = np.ones(len(X_rec))
    ss = ScoreSet.build(score_fn(k["score"], y_cal, mu_cal, sd_cal), w_cal)
    p_rec = ss.pvalue(score_fn(k["score"], y_rec, mu_rec, sd_rec), w_rec)
    W = w_rec.sum(); p = w_rec / W
    m = (p_rec <= cert.a).astype(float)          # miss indicator at the selected level
    eps_hat = float((p * m).sum())
    Vhat = float((p * p * m).sum())              # unbiased for sum p^2 E[m] (>= Var)
    bmax = float(p.max())
    n_w, n_eff = 1.0 / bmax, 1.0 / float((p * p).sum())
    return dict(N=N, eps=eps_hat, cert_eps=cert.eps, n_w=n_w, n_eff=n_eff, Vhat=Vhat,
                V_worst=eps_hat * bmax, V_spread=eps_hat / n_eff,
                frac_miss_weight=float((p * m).sum()), n_rec=len(w_rec))


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
    out = []
    for a, k in cap:
        r = probe(a, k, cfg["N"]); r["seed"] = cfg["seed"]; out.append(r)
    return out


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    B3, _ = CONDS[0]
    cfgs = [dict(B3, seed=s, planner="certified", weighted=True) for s in range(seeds)]
    print(f"[e49] {len(cfgs)} runs, {workers} workers", flush=True)
    with Pool(workers) as p:
        rows = [r for rs in p.map(one, cfgs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e49_variance.csv"), index=False)
    print(f"[e49] wrote results/e49_variance.csv ({len(df)} rows)")
