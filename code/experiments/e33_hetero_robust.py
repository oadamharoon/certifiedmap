"""E33: is the E32 heterogeneity result robust, and does it survive a *drifting* pose estimate?

E32 showed the certified state beats a coverage mask whenever a teammate is untrustworthy, with per-scan
registration error (sigma_e) as the non-label uncertainty. Three things it did not establish:
  drift     the realistic failure of a badly localised robot is accumulated, correlated error, not zero-mean
            per-scan noise. `drift_agents` random-walks the pose estimate: rays from the true pose, samples
            written at the drifted one.
  team      does the effect hold at N=5, and with two unreliable robots out of five?
  placement does it survive resampling the start poses on the same worlds (as E18 did for the static study)?
20 worlds, 100 m per robot, mesh; planners certified / entropy_u / entropy_g (the coverage mask) / entropy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

PLANNERS = ["certified", "entropy_u", "entropy_g", "entropy"]
BASE = dict(topology="mesh", budget=100.0, sigma_r=0.05, K=120)

CONDS = {
    # drift, three robots, one of them drifting
    "drift_lo":   dict(N=3, drift_agents=(0.0, 0.0, 0.02)),
    "drift_hi":   dict(N=3, drift_agents=(0.0, 0.0, 0.05)),
    # drift plus label noise on a different robot: two uncertainty types, as in E32's `mixed`
    "drift_mixed": dict(N=3, drift_agents=(0.0, 0.0, 0.05), p_flip_agents=(0.0, 0.10, 0.0)),
    # team size: five robots, one and then two unreliable
    "n5_clean":   dict(N=5),
    "n5_one":     dict(N=5, p_flip_agents=(0.0, 0.0, 0.0, 0.0, 0.10)),
    "n5_two":     dict(N=5, p_flip_agents=(0.0, 0.0, 0.0, 0.10, 0.10)),
    "n5_drift":   dict(N=5, drift_agents=(0.0, 0.0, 0.0, 0.0, 0.05)),
}

if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for cond in CONDS.values():
            for p in PLANNERS:
                cfgs.append(dict(BASE, seed=s, planner=p, **cond))
        # placement resampling on the E32 conditions that carried the result
        for st in (1, 2):
            for cond in [dict(N=3, p_flip_agents=(0.0, 0.0, 0.10)),
                         dict(N=3, sigma_e_agents=(0.04, 0.04, 0.60), sigma_theta_agents=(0.003, 0.003, 0.10))]:
                for p in ["certified", "entropy_g"]:
                    cfgs.append(dict(BASE, seed=s, planner=p, start_seed=st, **cond))
    run_all("e33", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
