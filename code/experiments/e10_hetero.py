"""E10 (4.6): heteroscedastic sensing -- label reliability decays with range."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
SEEDS = list(range(20))
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner=p, budget=150.0, sigma_r=0.05, p_flip_range=pr)
            for s in SEEDS for p in ["certified", "entropy", "lse", "frontier"] for pr in [0.1, 0.3]]
    run_all("e10", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
