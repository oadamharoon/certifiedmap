"""E3: team size and topology at fixed total budget (100 m)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

SEEDS = list(range(20))
if __name__ == "__main__":
    cfgs = []
    for s in SEEDS:
        for p in ["certified", "entropy", "entropy_u", "lawnmower"]:
            for N, topo in [(1, "mesh"), (2, "mesh"), (3, "mesh"), (5, "mesh"), (5, "ring")]:
                cfgs.append(dict(seed=s, N=N, topology=topo, planner=p, budget=100.0 / N, sigma_r=0.05))
    run_all("e3", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
