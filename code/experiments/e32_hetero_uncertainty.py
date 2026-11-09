"""E32: heterogeneity beyond label reliability. Does the certified state still beat a coverage mask when a
teammate is untrustworthy for a *different* reason than flipped labels?

Conditions (N=3 mesh, 100 m each, 20 worlds):
  clean      homogeneous reference
  flip       one robot with 10% label flips (E12 replication at these settings)
  pose       one robot with poor localisation (sigma_e 0.30 m, sigma_theta 0.05 rad)
  pose_hi    the same, worse (0.60 m, 0.10 rad)
  mixed      one flip-unreliable robot and one pose-unreliable robot (different uncertainty types in one team)
Planners: certified (fused undecided set), entropy_u (same state, information gain), entropy_g (coverage mask:
the cells no neighbour has gated), entropy (no shared state), frontier_g (coordinated frontier).
Plus the obvious operator's fallback: drop the untrusted robot, i.e. N=2 clean robots at the same 300 m total.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120)
CONDS = {
    "clean":   dict(),
    "flip":    dict(p_flip_agents=(0.0, 0.0, 0.10)),
    "pose":    dict(sigma_e_agents=(0.04, 0.04, 0.30), sigma_theta_agents=(0.003, 0.003, 0.05)),
    "pose_hi": dict(sigma_e_agents=(0.04, 0.04, 0.60), sigma_theta_agents=(0.003, 0.003, 0.10)),
    "mixed":   dict(p_flip_agents=(0.0, 0.10, 0.0), sigma_e_agents=(0.04, 0.04, 0.60),
                    sigma_theta_agents=(0.003, 0.003, 0.10)),
}
PLANNERS = ["certified", "entropy_u", "entropy_g", "entropy", "frontier_g"]

if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for cond in CONDS.values():
            for p in PLANNERS:
                cfgs.append(dict(BASE, seed=s, planner=p, **cond))
        for p in ["certified", "entropy_g"]:                      # drop-the-untrusted-robot baseline
            cfgs.append(dict(BASE, seed=s, planner=p, N=2, budget=150.0))
    run_all("e32", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
