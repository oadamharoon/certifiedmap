"""E9 (Q2/Q3): team size at fixed per-robot budget (parallel speed-up), spread and overlap."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
SEEDS = list(range(10))
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=N, topology="mesh", planner=p, budget=40.0, sigma_r=0.05)
            for s in SEEDS for N in [1, 2, 3, 5, 8] for p in ["certified", "entropy_g", "entropy", "entropy_u"]]
    run_all("e9", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
