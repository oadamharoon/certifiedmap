"""E16: exact block-combined certificate (Corollary 1) vs the pooled certificate, certified planner, alpha sweep."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner="certified", budget=100.0, alpha=a, sigma_r=0.05, log_combined=True) for s in range(20) for a in [0.02, 0.05, 0.1, 0.2]]
    cfgs += [dict(seed=s, N=3, topology="mesh", planner="certified", budget=100.0 / 3, alpha=0.05, sigma_r=0.05, log_combined=True) for s in range(20)]
    run_all("e16", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
