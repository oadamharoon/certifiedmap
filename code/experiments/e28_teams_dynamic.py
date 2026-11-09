"""E28 (Gate 3a): three-robot teams (mesh, 100 m per robot) among 10 random-walk hazards at speeds 1 and 3: seven guards,
the certified guard at 0.02, and the certified guard without pooled traversal pairs (share_pairs=False). 20 worlds."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e26_gate2 import GUARDS, BASE
if __name__ == "__main__":
    base = dict(BASE, N=3, topology="mesh", budget=100.0, planner="entropy", n_hazards=10)
    cfgs = []
    for s in range(20):
        for st in [1.0, 3.0]:
            for g in GUARDS:
                cfgs.append(dict(base, seed=s, hazard_step=st, guard=g))
            cfgs.append(dict(base, seed=s, hazard_step=st, guard="certified", guard_alpha=0.02))
            cfgs.append(dict(base, seed=s, hazard_step=st, guard="certified", share_pairs=False))
    run_all("e28", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
