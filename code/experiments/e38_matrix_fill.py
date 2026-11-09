"""E38: fill the gaps in the heterogeneity matrix.

The programme so far is a star design around (N=3, our candidate generator): team size was varied at the flip and
drift faults only, and the RRT generator was checked at the clean and flip conditions only. This crosses the two
factors that carry the claim with the two fault types they were not crossed with, so "the effect grows with team
size" and "the effect does not depend on our candidate generator" rest on more than one fault each.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(topology="mesh", budget=100.0, sigma_r=0.05, K=120)
REG = dict(sigma_e_agents=(0.04,) * 4 + (0.60,), sigma_theta_agents=(0.003,) * 4 + (0.10,))
MIX = dict(p_flip_agents=(0.0, 0.0, 0.0, 0.10, 0.0), sigma_e_agents=(0.04,) * 4 + (0.60,),
           sigma_theta_agents=(0.003,) * 4 + (0.10,))
REG3 = dict(sigma_e_agents=(0.04, 0.04, 0.60), sigma_theta_agents=(0.003, 0.003, 0.10))
MIX3 = dict(p_flip_agents=(0.0, 0.10, 0.0), sigma_e_agents=(0.04, 0.04, 0.60),
            sigma_theta_agents=(0.003, 0.003, 0.10))

if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for cond in (REG, MIX):                                   # team size x fault type
            for p in ["certified", "entropy_u", "entropy_g", "entropy"]:
                cfgs.append(dict(BASE, seed=s, N=5, planner=p, **cond))
        for cond in (REG3, MIX3):                                 # generator x fault type
            for p in ["certified", "entropy_g"]:
                cfgs.append(dict(BASE, seed=s, N=3, planner=p, generator="rrt", **cond))
    run_all("e38", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 15)
