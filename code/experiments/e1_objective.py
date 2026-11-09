"""E1: does the certified-area objective change trajectories and certified fraction? (single agent)"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

PLANNERS = ["certified", "entropy", "lse", "frontier", "lawnmower", "random"]
SEEDS = list(range(20))
CONDS = [dict(sigma_r=0.02, wall_thick=0.25), dict(sigma_r=0.10, wall_thick=0.25),
         dict(sigma_r=0.02, wall_thick=0.75), dict(sigma_r=0.10, wall_thick=0.75)]

if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner=p, budget=150.0, **c) for c in CONDS for s in SEEDS for p in PLANNERS]
    run_all("e1", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
