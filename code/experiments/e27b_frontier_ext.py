"""E27b: extend the safety-progress frontier upward: log-odds guard at thresholds 0.2/0.3/0.5 (decay 0.9) and the
horizon certificate at 0.2; random-walk hazards, speeds 1 and 3, 20 worlds, entropy planner (same design as E27)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    base = dict(N=1, budget=300.0, sigma_r=0.05, n_hazards=20, hazard_size=3.0, hazards_move_after_plan=True, K=120, planner="entropy")
    cfgs = []
    for s in range(20):
        for st in [1.0, 3.0]:
            cfgs.append(dict(base, seed=s, hazard_step=st, guard="certified", guard_alpha=0.2))
            for al in [0.2, 0.3, 0.5]:
                cfgs.append(dict(base, seed=s, hazard_step=st, guard="prob", prob_decay=0.9, alpha=al))
    run_all("e27", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
