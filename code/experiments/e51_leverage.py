"""F1a / E51: can the acquisition move the quantity Prop. 4.3's bound depends on?

E48/E49 showed the 2.95 correction factor at N=3 is set by n~ = W/max w, and that it cannot be recovered after
the fact. A paired comparison of the two existing acquisitions on the same world shows the planner moves n~ by
about 13.5%, but neither was trying. This screen asks how much is available *per decision*: at a planning step,
recompute the n~ that each candidate segment would produce if it were the one driven.

For a candidate with predicted sample locations S, the sampled density becomes
density_on_grid([X_cal, X_rec, S]), so the weights w = u/rho and hence n~ = sum(w_rec)/max(w_rec) can be
recomputed exactly, with no simulation of the future.

Pre-registered rule: R1 proceeds only if the median across blocks of p90/p10 of predicted n~ exceeds 1.5.
"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from multiprocessing import Pool
from e44_provable_denom import CONDS

KEEP = 40                     # planning steps captured per run, spread over the mission
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")


def ntilde(X_cal, X_rec, gate, nx, ny, res, bw, extra=None, want_w=False):
    """n~ = sum(w_rec)/max(w_rec) for the sampled density implied by these locations."""
    from certmap.conformal import density_on_grid
    pts = [X_cal, X_rec] + ([extra] if extra is not None and len(extra) else [])
    rho = density_on_grid(np.concatenate(pts, 0), nx, ny, res, bw)
    wx = np.zeros(nx * ny)
    u = 1.0 / max(gate.sum(), 1)
    with np.errstate(divide="ignore"):
        wx[gate] = u / rho[gate]
    fin = gate & np.isfinite(wx)
    if not fin.any():
        return np.nan
    wx[fin] /= np.median(wx[fin])
    ix = np.clip(np.floor(X_rec[:, 0] / res).astype(int), 0, nx - 1)
    iy = np.clip(np.floor(X_rec[:, 1] / res).astype(int), 0, ny - 1)
    w_rec = np.maximum(np.nan_to_num(wx[iy * nx + ix], nan=1e-9, posinf=1e-9), 1e-9)
    if want_w:
        return w_rec
    return float(w_rec.sum() / max(w_rec.max(), 1e-12))


def knockout_curve(w, ks=(1, 2, 5, 10, 25, 50, 100, 250)):
    """n~ if the top-k weights were each pulled down to the (k+1)-th largest, i.e. the best case for a planner
    that goes and samples the k least-sampled gated cells. Bounds what targeted sampling can buy."""
    v = np.sort(w)[::-1]; W = v.sum(); out = {}
    for k in ks:
        if k + 1 >= len(v):
            continue
        cap = v[k]
        vv = v.copy(); vv[:k] = cap
        out[k] = float(vv.sum() / max(vv.max(), 1e-12))
    return out


def one(cfg):
    import certmap.sim as S
    import certmap.planner as P
    from certmap.sim import Simulation, Config
    cap = collections.deque(maxlen=KEEP)
    seen = []
    real_affected = P.affected_cells
    def affected_spy(Sarr, *a, **k):
        seen.append(np.asarray(Sarr).copy())
        return real_affected(Sarr, *a, **k)
    real_plan = Simulation.plan_agent
    def plan_spy(self, a, pos, others, prev_S, target, *rest, **kw):
        del seen[:]
        out = real_plan(self, a, pos, others, prev_S, target, *rest, **kw)
        try:
            X, y, fold = a.cal_arrays()
            if len(X) and a.cert is not None and len(seen) >= 3:
                sc, rc = ~fold, fold
                cap.append(dict(X_cal=X[sc].copy(), X_rec=X[rc].copy(), gate=a.cert.gate.copy(),
                                cands=[s for s in seen if len(s)]))
        except Exception:
            pass
        return out
    S.affected_cells = affected_spy; P.affected_cells = affected_spy
    Simulation.plan_agent = plan_spy
    try:
        sim = Simulation(Config(**cfg)); sim.run()
        nx, ny, res, bw = sim.world.nx, sim.world.ny, sim.world.res, cfg.get("kde_bw", 0.5)
    finally:
        S.affected_cells = real_affected; P.affected_cells = real_affected
        Simulation.plan_agent = real_plan

    rows = []
    for bi, c in enumerate(cap):
        base = ntilde(c["X_cal"], c["X_rec"], c["gate"], nx, ny, res, bw)
        vals = [ntilde(c["X_cal"], c["X_rec"], c["gate"], nx, ny, res, bw, extra=s) for s in c["cands"]]
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) < 3:
            continue
        p10, p50, p90 = np.percentile(vals, [10, 50, 90])
        wv = ntilde(c["X_cal"], c["X_rec"], c["gate"], nx, ny, res, bw, want_w=True)
        ko = knockout_curve(wv)
        rows.append(dict(seed=cfg["seed"], N=cfg["N"], step=bi, n_cands=len(vals), base=base,
                         n_rec=len(wv), **{f"ko{k}": v for k, v in ko.items()},
                         p10=p10, p50=p50, p90=p90, spread=p90 / max(p10, 1e-9),
                         best_over_base=vals.max() / max(base, 1e-9)))
    return rows


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    B3, _ = CONDS[0]
    cfgs = [dict(B3, seed=s, planner="certified", weighted=True) for s in range(seeds)]
    print(f"[e51] {len(cfgs)} runs, {workers} workers", flush=True)
    with Pool(workers) as p:
        rows = [r for rs in p.map(one, cfgs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e51_leverage.csv"), index=False)
    print(f"[e51] wrote results/e51_leverage.csv ({len(df)} planning steps)")
