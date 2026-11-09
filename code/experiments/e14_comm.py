"""E14: communication payload accounting per block (homogeneous team, N=3 mesh)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0 / 3, sigma_r=0.05, log_comm=True) for s in range(10) for p in ["certified", "entropy"]]
    run_all("e14", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
