"""E25 (Gate 1): traversal guards under moving hazards, hazard-speed sweep. Misses = planned-through cells that are
occupied at traversal time (the robot stops there). Guards: none, static inflation, lag-1 horizon certificate at
per-cell level alpha and alpha/20 (segment-level union bound), oracle of the current map."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = []
    for s in range(10):
        for st in [0.5, 1.0, 2.0, 3.0]:
            for g, ga in [("none", -1.0), ("inflate", -1.0), ("certified", -1.0), ("certified", 0.0025), ("oracle", -1.0)]:
                cfgs.append(dict(seed=s, N=1, planner="entropy", budget=300.0, sigma_r=0.05, n_hazards=20, hazard_size=3.0, hazard_step=st,
                                 hazards_move_after_plan=True, guard=g, guard_alpha=ga, K=120))
    run_all("e25", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
