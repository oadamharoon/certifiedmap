"""E27: (c) safety-progress frontier at the main setting: certified guard at levels 0.1/0.2 (0.01/0.02/0.05 in E26),
log-odds guard at thresholds 0.01-0.5 (decay 0.9) and decays 0.8/0.95/0.99 (threshold 0.05); (d) goal-directed hazards
(persistent motion between re-sampled goals), seven guards plus certified at 0.02. Speeds 1 and 3, 20 worlds."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e26_gate2 import GUARDS, BASE
if __name__ == "__main__":
    base = dict(BASE, planner="entropy", n_hazards=10)
    cfgs = []
    for s in range(20):
        for st in [1.0, 3.0]:
            for ga in [0.1, 0.2]:
                cfgs.append(dict(base, seed=s, hazard_step=st, guard="certified", guard_alpha=ga))
            for al in [0.01, 0.02, 0.1, 0.2, 0.3, 0.5]:
                cfgs.append(dict(base, seed=s, hazard_step=st, guard="prob", prob_decay=0.9, alpha=al))
            for dec in [0.8, 0.95, 0.99]:
                cfgs.append(dict(base, seed=s, hazard_step=st, guard="prob", prob_decay=dec))
            for g in GUARDS:
                cfgs.append(dict(base, seed=s, hazard_step=st, guard=g, hazard_motion="goal"))
            cfgs.append(dict(base, seed=s, hazard_step=st, guard="certified", guard_alpha=0.02, hazard_motion="goal"))
    run_all("e27", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
