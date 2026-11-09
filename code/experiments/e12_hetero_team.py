"""E12: heterogeneous team -- one robot's labels are unreliable, so its observed cells are not certified.
Does reading the fused undecided set (certified, entropy-U) beat a coverage mask (entropy-G)?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
SEEDS = list(range(20))
if __name__ == "__main__":
    cfgs = []
    for s in SEEDS:
        for flips in [[0.0, 0.0, 0.05], [0.0, 0.0, 0.10], [0.0, 0.05, 0.10]]:
            for p in ["certified", "entropy_u", "entropy_g", "entropy"]:
                cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0 / 3, sigma_r=0.05, p_flip_agents=flips))
    run_all("e12", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
