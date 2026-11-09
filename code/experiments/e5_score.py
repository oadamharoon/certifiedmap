"""E5 (4.1): E1 repeated under the nested 'margin' nonconformity score."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e1_objective import CONDS, SEEDS, PLANNERS
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner=p, budget=150.0, score="margin", **c) for c in CONDS for s in SEEDS for p in PLANNERS]
    run_all("e5", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
