"""E6 (4.3): label-flip sweep across the miscoverage budget beta = s*alpha (= 0.035)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
SEEDS = list(range(20))
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, p_flip=pf)
            for s in SEEDS for p in ["certified", "entropy", "lse"] for pf in [0.01, 0.02, 0.03, 0.05, 0.10]]
    cfgs += [dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, p_flip=pf, score="margin")
             for s in SEEDS for p in ["certified", "entropy"] for pf in [0.02, 0.05, 0.10]]
    run_all("e6", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
