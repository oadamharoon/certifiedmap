"""E1b: same planner, same world, different sensor/planner randomness -> reference divergence level."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e1_objective import CONDS, SEEDS

if __name__ == "__main__":
    cfgs = [dict(seed=s, noise_seed=1, N=1, planner=p, budget=150.0, **c)
            for c in CONDS for s in SEEDS for p in ["certified", "entropy", "lse"]]
    run_all("e1b", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
