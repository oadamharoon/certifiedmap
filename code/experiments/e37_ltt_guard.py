"""E37 (item 5): two guarantees for the same evidence.

`certified` admits per (region x staleness) class when that class's weighted miss rate clears a level adapted by
ACI: a marginal statement, one class at a time, which can always be issued but is all-or-nothing per class.
`ltt` scores every cell by its bin's miss rate and admits one threshold chosen by Learn-then-Test: an (alpha,
delta) statement about the admitted *set*, which needs enough calibration pairs to issue at all (at alpha = 0.05,
delta = 0.1 the safest threshold needs >= 45 pairs with no misses) and otherwise refuses to the uncertified rule.
Levels 0.02 / 0.05 / 0.10 separate the two effects: conservatism and data cost.
Reported per run: miss rate over entered cells on certified blocks, the share of blocks that certified at all,
the uncertified blocks, and progress. The delta check is across worlds: the fraction of runs whose admitted set
exceeded alpha should be at most delta.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(N=1, budget=300.0, sigma_r=0.05, n_hazards=10, hazard_size=3.0, hazards_move_after_plan=True,
            K=120, planner="entropy")
if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for st in [1.0, 3.0]:
            for g in ["fresh", "oracle"]:
                cfgs.append(dict(BASE, seed=s, hazard_step=st, guard=g))
            for g in ["certified", "ltt"]:
                for al in [0.02, 0.05, 0.10]:
                    cfgs.append(dict(BASE, seed=s, hazard_step=st, guard=g, guard_alpha=al))
    run_all("e37", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 15)
