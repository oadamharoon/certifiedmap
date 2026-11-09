"""E39: the traversal certificate as a runtime safety filter, independent of the policy it filters.

The guard constrains which cells a planner may drive through; it never chooses the route. If that is a genuine
runtime-assurance layer then it should hold its level under *any* policy, including a deliberately reckless one.
This applies the same guard to four policies that differ in how much they care about safety -- a random walk, a
nearest-frontier rule, information gain, and the certified acquisition -- at two hazard speeds, and reports the
realised miss rate per entered cell with and without the guard.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(N=1, budget=300.0, sigma_r=0.05, n_hazards=10, hazard_size=3.0, hazards_move_after_plan=True, K=120)
if __name__ == "__main__":
    cfgs = [dict(BASE, seed=s, planner=p, hazard_step=st, guard=g)
            for s in range(20) for p in ["random", "frontier", "entropy", "certified"]
            for st in [1.0, 3.0] for g in ["none", "certified"]]
    run_all("e39", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 15)
