"""Why the objectives coincide: how much of the commitment decision is variance-driven."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from certmap.sim import Simulation, Config

def one(seed):
    sim = Simulation(Config(seed=seed, N=1, planner="certified", budget=75, save_maps=True)); out = sim.run()
    w = sim.world; a = sim.agents[0]; c = a.cert; F = a.fused; obs = w.obs
    g = c.gate & obs
    q = c.q                      # per-cell threshold (inf without calibration support)
    fin = g & np.isfinite(q)
    sig, mu = c.sigma, np.abs(c.mu)
    band_lo = sig < 1.0 / q       # sigma below 1/q: the wrong label is rejected for any mu
    band_hi = sig > 2.0 / q       # sigma above 2/q: no label can be rejected for any mu
    mid = fin & ~band_lo & ~band_hi
    return dict(seed=seed,
        gated=int(g.sum()), no_cal_support=float((g & ~np.isfinite(q)).mean() / max(g.mean(), 1e-9)),
        frac_sigma_below_1q=float((fin & band_lo).sum() / fin.sum()), frac_sigma_above_2q=float((fin & band_hi).sum() / fin.sum()),
        frac_mid=float(mid.sum() / fin.sum()),
        committed_in_lo=float(F.committed[fin & band_lo].mean()), committed_in_mid=float(F.committed[mid].mean()) if mid.any() else np.nan,
        undecided_in_mid=float(F.undecided[mid].mean()) if mid.any() else np.nan,
        undec_explored_total=int((F.undecided & g).sum()),
        undec_share_nocal=float((F.undecided & g & ~np.isfinite(q)).sum() / max((F.undecided & g).sum(), 1)),
        undec_share_mid=float((F.undecided & mid).sum() / max((F.undecided & g).sum(), 1)),
        undec_share_hi=float((F.undecided & fin & band_hi).sum() / max((F.undecided & g).sum(), 1)),
        empty_in_lo_mu=float(mu[F.empty & fin].mean()) if (F.empty & fin).any() else np.nan,
        q_scalar=float(c.q_scalar), sigma_floor=float(np.sqrt(sim.cfg.noise_var)), one_over_q=float(1 / c.q_scalar), two_over_q=float(2 / c.q_scalar),
        nodata=float(F.nodata[obs].mean()), undecided_total=float(F.undecided[obs].mean()))
if __name__ == "__main__":
    with Pool(10) as p: rows = p.map(one, range(10))
    df = pd.DataFrame(rows); pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False)); print(); print(df.drop(columns="seed").mean().round(3).to_string())
    df.to_csv(os.path.join(os.path.dirname(__file__), "..", "results", "mechanism.csv"), index=False)
