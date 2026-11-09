"""E8 (4.4): greedy vs joint allocation, N=2 certified."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
SEEDS = list(range(20))
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=2, planner="certified", budget=50.0, sigma_r=0.05, check_joint=True, alloc=al) for s in SEEDS for al in ["greedy", "joint"]]
    run_all("e8", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
