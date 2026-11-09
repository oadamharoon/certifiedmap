"""E41: the sensor-capability boundary case at the budget the rest of the heterogeneity table uses.

E13 ran half-range and sparse-beam teams at 33 m per robot, which is a third of the 100 m that every row of the
ICCPS heterogeneity table uses, so its null is not directly comparable to them. This repeats the two capability
conditions under E32's exact configuration (N=3, mesh, 100 m per robot, sigma_r=0.05, K=120) so that
"capability inequality buys nothing" and "label-error inequality buys a lot" are measured in the same regime.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120)
CONDS = {"range": dict(range_agents=(5.0, 5.0, 2.5)), "beams": dict(beams_agents=(40, 40, 8))}
PLANNERS = ["certified", "entropy_g"]

if __name__ == "__main__":
    cfgs = [dict(BASE, seed=s, planner=p, **c)
            for s in range(20) for p in PLANNERS for c in CONDS.values()]
    run_all("e41", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
