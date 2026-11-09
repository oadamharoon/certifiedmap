"""E46: how many times does each traversal class update?

Prop. 8.1 bounds the long-run miss rate of class g by alpha + (alpha + gamma(1-alpha))/(gamma T_g), where T_g
counts the blocks at which class g updated its level. The bound is only as good as T_g, so the paper has to
report it. Same settings as the guard table (E26), certified guard only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(N=1, budget=300.0, sigma_r=0.05, hazard_size=3.0, hazards_move_after_plan=True, K=120,
            planner="entropy", guard="certified")
SETTINGS = [(5, 1.0), (10, 1.0), (10, 3.0), (20, 1.0)]

if __name__ == "__main__":
    cfgs = [dict(BASE, seed=s, n_hazards=h, hazard_step=st) for s in range(5) for h, st in SETTINGS]
    run_all("e46", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
