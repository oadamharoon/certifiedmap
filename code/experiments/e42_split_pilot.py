"""E42: what does a provable certificate cost?

The deployed (pooled) certificate is validated empirically rather than proved, because the recalibrated level is
computed against the same score fold that carries the claim, and because the certified acquisition lets
calibration labels steer later segments. Splitting the calibration pool into four disjoint folds removes both:
fold 0 carries the claim, fold 2 is recalibrated against fold 1, fold 3 feeds the acquisition, and the e-value
denominator is the recalibrated level itself. This pilot measures the price in E32's own regime.

Decision rule, fixed in advance: coverage on the query law must stay at or above nominal; if the certified minus
coverage-map margin survives within its interval and the absolute certified fraction falls by at most 0.02, the
split is worth adopting, otherwise it is reported as a measured ablation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120)
CONDS = {"clean": dict(), "flip": dict(p_flip_agents=(0.0, 0.0, 0.10))}
PLANNERS = ["certified", "entropy_g"]

if __name__ == "__main__":
    cfgs = [dict(BASE, seed=s, planner=p, split4=True, **c)
            for s in range(20) for p in PLANNERS for c in CONDS.values()]
    run_all("e42", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
