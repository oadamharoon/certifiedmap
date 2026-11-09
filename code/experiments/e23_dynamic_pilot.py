"""E23 pilot: moving hazards. Does the pooled certificate stay valid against the current map as hazard volume grows,
does a windowed certificate restore validity and at what cost, and do planners differ in time-averaged certified area?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = []
    for s in range(10):
        for nh, hs in [(4, 1.5), (8, 3.0), (12, 3.0)]:
            for win in [0, 10]:
                for p in ["certified", "entropy", "frontier"]:
                    cfgs.append(dict(seed=s, N=1, planner=p, budget=200.0, sigma_r=0.05, n_hazards=nh, hazard_size=hs, hazard_step=1.0, window=win, K=60))
    run_all("e23", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
