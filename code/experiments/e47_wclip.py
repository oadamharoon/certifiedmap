"""E47: does capping the importance-weight tail buy back the finite-sample correction of Prop. 4.3?

n_tilde = (sum w)/max w is set by the largest weight, so the tail alone decides eta. Capping w at a quantile
trades a stronger tilt of the certified law (the tilted-law argument of Sec. 4.1) for a smaller correction.
Two things have to hold before it can be adopted: coverage on the operator's query law must survive the extra
tilt at every level, and the heterogeneity margins must not move.

(a) validity sweep, single robot, levels as in E2, clip at 0.90 and 0.80;
(b) the seven conditions of the heterogeneity table at clip 0.90, paired against E44's w_clip=0 arm.

Decision rule, fixed in advance: adopt only if coverage stays at or above nominal at every level and every
margin of the table stays inside its interval; otherwise report the correction as it stands.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e44_provable_denom import CONDS, PLANNERS

if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for a in [0.2, 0.1, 0.05, 0.02]:
            for wc in [0.90, 0.80]:
                cfgs.append(dict(seed=s, N=1, planner="certified", budget=100.0, alpha=a,
                                 weighted=True, sigma_r=0.05, w_clip=wc))
        for p in PLANNERS:
            for base, c in CONDS:
                cfgs.append(dict(base, seed=s, planner=p, w_clip=0.90, **c))
    run_all("e47", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
