"""E13: heterogeneity in reach (range) and in sampling density (beams) rather than reliability."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
SEEDS = list(range(20))
if __name__ == "__main__":
    cfgs = []
    for s in SEEDS:
        for p in ["certified", "entropy_u", "entropy_g", "entropy"]:
            cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0 / 3, sigma_r=0.05, range_agents=[5.0, 5.0, 2.5]))
            cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0 / 3, sigma_r=0.05, beams_agents=[40, 40, 8]))
    run_all("e13", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
