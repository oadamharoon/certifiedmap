"""E33b: where does drift stop being detectable?

E33 found that a drifting robot costs the team much less than a label-flipping one (+0.012-0.015 against the
coverage mask, versus +0.075 for flips), and that committed accuracy barely moves (0.996 -> 0.992). The reason
is that drift produces a map that is *wrong but self-consistent*: the certificate detects contradictions, and a
uniformly shifted wall contradicts nothing. That is a limit of the guarantee and it should be measured, not
assumed: at 0.02-0.05 m per block the accumulated shift is well under a cell, so this sweeps up to 0.20 m per
block (about a metre of accumulated shift) and reports committed accuracy, not only certified area.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

if __name__ == "__main__":
    cfgs = [dict(N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120, seed=s, planner=p,
                 drift_agents=(0.0, 0.0, d))
            for s in range(20) for p in ["certified", "entropy_g"] for d in [0.10, 0.20]]
    run_all("e33", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
