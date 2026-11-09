"""E26 (Gate 2, final design): (a) seven traversal guards under the entropy planner at the main setting (10 random-walk
hazards of 3 m, speeds 1 and 3 m/block) and a density sweep (5 and 20 hazards at speed 1), the certified guard also at
levels 0.01 and 0.02; (b) acquisitions under the certified guard at the main setting. 300 m, K=120, 20 worlds.
Validity: occupied cells among the cells entered on certified blocks (Prop. A), by staleness; strict = all admitted cells."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
GUARDS = ["none", "inflate", "fresh", "prob", "cp_track", "certified", "oracle"]
LEVELS = [0.01, 0.02]                     # extra certified levels (0.05 = alpha is the default)
BASE = dict(N=1, budget=300.0, sigma_r=0.05, hazard_size=3.0, hazards_move_after_plan=True, K=120)
if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for nh, st in [(10, 1.0), (10, 3.0), (5, 1.0), (20, 1.0)]:
            for g in GUARDS:
                cfgs.append(dict(BASE, seed=s, planner="entropy", n_hazards=nh, hazard_step=st, guard=g))
            for ga in LEVELS:
                cfgs.append(dict(BASE, seed=s, planner="entropy", n_hazards=nh, hazard_step=st, guard="certified", guard_alpha=ga))
        for st in [1.0, 3.0]:
            for p in ["certified", "frontier"]:
                cfgs.append(dict(BASE, seed=s, planner=p, n_hazards=10, hazard_step=st, guard="certified"))
    run_all("e26", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
