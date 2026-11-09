"""E43: the sample-split (provable) certificate on the remaining conditions of the heterogeneity table.

E42 measured its price on the clean and label-flip conditions. This completes the ablation over every row of
that table, so the reported cost is a range across the same conditions the paper's main result uses, measured
in the same regime and paired by world.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

B3 = dict(N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120)
B5 = dict(N=5, topology="mesh", budget=100.0, sigma_r=0.05, K=120)
CONDS = [
    (B3, dict(sigma_e_agents=(0.04, 0.04, 0.60), sigma_theta_agents=(0.003, 0.003, 0.10))),   # misregistration
    (B3, dict(drift_agents=(0.0, 0.0, 0.05))),                                                 # drift
    (B3, dict(p_flip_agents=(0.0, 0.10, 0.0), sigma_e_agents=(0.04, 0.04, 0.60),
              sigma_theta_agents=(0.003, 0.003, 0.10))),                                       # two fail differently
    (B5, dict(p_flip_agents=(0.0, 0.0, 0.0, 0.0, 0.10))),                                      # five, one unreliable
    (B5, dict(p_flip_agents=(0.0, 0.0, 0.0, 0.10, 0.10))),                                     # five, two unreliable
]
PLANNERS = ["certified", "entropy_g"]

if __name__ == "__main__":
    cfgs = [dict(base, seed=s, planner=p, split4=True, **c)
            for s in range(20) for p in PLANNERS for base, c in CONDS]
    run_all("e43", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
