"""E17: graded dichotomy check. Own-certificate commit share among well-observed cells vs min(1, beta_i/p).
Single robot at flip p in {0.02,0.03,0.05,0.10} (beta = s*alpha = 0.035) and the unreliable robot of a
three-robot team at p=0.10 (beta_i = s*alpha/3 = 0.0117)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from certmap.sim import Simulation, Config
from runner import RESULTS

def own_commit(a, obs, tol):
    c = a.cert
    rej = c.e > 0                                   # own rejection of label y at x: p(x,y) <= a
    comm = (rej.sum(1) == 1)
    smin = c.sigma[c.gate & obs].min()
    wo = c.gate & obs & np.isfinite(c.q) & (c.sigma <= tol * smin)
    return float(comm[wo].mean()), int(wo.sum()), float(c.a), float(c.q_scalar)

def one(args):
    kind, seed, p = args
    if kind == "single":
        sim = Simulation(Config(seed=seed, N=1, planner="certified", budget=100.0, sigma_r=0.05, p_flip=p)); beta = 0.7 * 0.05
    else:
        sim = Simulation(Config(seed=seed, N=3, topology="mesh", planner="certified", budget=100.0 / 3, sigma_r=0.05, p_flip_agents=[0.0, 0.0, p])); beta = 0.7 * 0.05 / 3
    sim.run()
    obs = sim.world.obs
    a = sim.agents[-1]
    rows = []
    for tol in [1.05, 1.2, 1.5]:
        share, n, a_i, q = own_commit(a, obs, tol)
        rows.append(dict(kind=kind, seed=seed, p=p, tol=tol, share=share, n_wo=n, a_i=a_i, q=q, pred=min(1.0, beta / p)))
    return rows

if __name__ == "__main__":
    jobs = [("single", s, p) for s in range(20) for p in [0.02, 0.03, 0.05, 0.10]] + [("team", s, 0.10) for s in range(20)] + [("team", s, 0.05) for s in range(20)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 20) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    df = pd.DataFrame(rows); os.makedirs(os.path.join(RESULTS, "e17_analysis"), exist_ok=True)
    df.to_csv(os.path.join(RESULTS, "e17_analysis", "runs.csv"), index=False)
    g = df.groupby(["kind", "p", "tol"]).agg(share=("share", "mean"), share_sd=("share", "std"), pred=("pred", "first"), a_i=("a_i", "mean"), q=("q", "mean"), n_wo=("n_wo", "mean")).reset_index()
    g["share_ci"] = 1.96 * g.share_sd / np.sqrt(20)
    print(g.round(3).to_string()); g.to_csv(os.path.join(RESULTS, "e17_analysis", "summary.csv"), index=False)
