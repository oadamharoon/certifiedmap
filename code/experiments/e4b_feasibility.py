"""E4b: certified runs of E1 with the feasibility signal logged (distance to nearest undecided cell)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e1_objective import CONDS, SEEDS
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner="certified", budget=150.0, log_feasibility=True, **c) for c in CONDS for s in SEEDS]
    run_all("e4b", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
