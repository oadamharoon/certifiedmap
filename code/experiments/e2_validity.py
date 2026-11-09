"""E2: does the certificate hold on the uniform explored law under adaptive planning?
Factors: planner (non-adaptive lawnmower vs adaptive certified/entropy), weighting on/off, alpha sweep."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

SEEDS = list(range(20))
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner=p, budget=100.0, alpha=a, weighted=w, sigma_r=0.05)
            for s in SEEDS for p in ["lawnmower", "certified", "entropy"] for a in [0.02, 0.05, 0.1, 0.2] for w in [True, False]]
    run_all("e2", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
