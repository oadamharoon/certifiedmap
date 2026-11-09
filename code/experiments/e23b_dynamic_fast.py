"""E23b: fast/large hazards and a long horizon. Where does the pooled certificate lose validity against the current map?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = []
    for s in range(10):
        for nh, hs, st in [(12, 3.0, 3.0), (20, 3.0, 3.0), (12, 4.0, 5.0)]:
            for p in ["certified", "entropy"]:
                cfgs.append(dict(seed=s, N=1, planner=p, budget=400.0, sigma_r=0.05, n_hazards=nh, hazard_size=hs, hazard_step=st, window=0, K=100))
    run_all("e23b", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
