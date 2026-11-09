"""E17b: is the graded dichotomy acquisition-independent? Same measurement as E17 (own-certificate commit share
among well-observed cells) for a single robot driven by the entropy acquisition instead of the certified one."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from multiprocessing import Pool
from certmap.sim import Simulation, Config
from runner import RESULTS
from e17_graded import own_commit

def one(args):
    seed, p = args
    sim = Simulation(Config(seed=seed, N=1, planner="entropy", budget=100.0, sigma_r=0.05, p_flip=p)); beta = 0.7 * 0.05
    sim.run()
    a = sim.agents[-1]; obs = sim.world.obs
    rows = []
    for tol in [1.05, 1.2, 1.5]:
        share, n, a_i, q = own_commit(a, obs, tol)
        rows.append(dict(kind="entropy", seed=seed, p=p, tol=tol, share=share, n_wo=n, a_i=a_i, q=q, pred=min(1.0, beta / p)))
    return rows

if __name__ == "__main__":
    jobs = [(s, p) for s in range(20) for p in [0.02, 0.03, 0.05, 0.10]]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 10) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "e17_analysis", "runs_entropy.csv"), index=False)
    g = df.groupby(["p", "tol"]).agg(share=("share", "mean"), share_sd=("share", "std"), pred=("pred", "first")).reset_index()
    g["share_ci"] = 1.96 * g.share_sd / np.sqrt(20)
    print(g.round(3).to_string())
