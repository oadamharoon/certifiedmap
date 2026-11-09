"""E32b: the drop-the-untrusted-robot baseline under a *fixed per-robot* budget.

E32's drop baseline gave the two remaining robots the whole 300 m (150 m each), i.e. it assumed the third
robot's path budget is transferable. When each robot's budget is set by its own battery, dropping one costs a
third of the team's capacity: two robots at 100 m each. Both versions are reported."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e32_hetero_uncertainty import BASE

if __name__ == "__main__":
    cfgs = [dict(BASE, seed=s, planner=p, N=2, budget=100.0) for s in range(20) for p in ["certified", "entropy_g"]]
    run_all("e32", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
