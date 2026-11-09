"""E30: stress settings: faster (5 m/block), larger (4 m) and denser (30) hazards; guards fresh / certified (0.05 and
0.02) / inflate / oracle; 20 worlds, entropy planner, 300 m."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e26_gate2 import BASE
if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for nh, st, hs in [(20, 5.0, 3.0), (20, 3.0, 4.0), (30, 3.0, 3.0), (10, 5.0, 4.0)]:
            for g in ["fresh", "certified", "inflate", "oracle"]:
                cfgs.append(dict(BASE, seed=s, planner="entropy", n_hazards=nh, hazard_step=st, hazard_size=hs, guard=g))
            cfgs.append(dict(BASE, seed=s, planner="entropy", n_hazards=nh, hazard_step=st, hazard_size=hs, guard="certified", guard_alpha=0.02))
    run_all("e30", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
