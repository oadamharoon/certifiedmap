"""E45: the effective sample size the recalibration bound is actually entitled to.

Three corrections separate the nominal recalibration fold from the n that a rigorous bound may use.
(i) The self-normalized n_eff = (sum w)^2/sum w^2 is the wrong functional for the weighted Bernstein step,
    which needs n_w = (sum w)/max w.
(ii) One scan's pose error is shared by all of its endpoints, so the independence unit of Assumption 2 is the
     scan, not the label, costing the cluster design effect 1 + (m_bar - 1) icc.
(iii) What remains, n_ind, is what sets the correction factor eta quoted in Sec. 4.1.

This measures all three on every condition of the heterogeneity table. No behaviour changes: denom_delta = 0.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e44_provable_denom import CONDS, PLANNERS

if __name__ == "__main__":
    cfgs = [dict(base, seed=s, planner=p, **c)
            for s in range(5) for p in PLANNERS for base, c in CONDS]
    run_all("e45", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
