"""E50: a detector-based reliability baseline for the coverage-map planner.

Table 1 compares the certified state against a coverage map that trusts every teammate, and Sec. 7 compares it
against *excluding* the unreliable robot, which presupposes knowing which one that is. The baseline a reviewer
will ask for sits between them: estimate each teammate's reliability from data and drop the ones that look bad.

Each robot already computes p_hat, the weighted fraction of its well-observed recalibration labels that
contradict its own model, which is a self-estimate of theta_i requiring no oracle. The baseline runs the
coverage-map planner (entropy_g) on a shared mask built only from teammates whose p_hat clears a threshold tau.

tau is swept rather than tuned, because an operator must fix one in advance; tau = 1.0 is a control arm in which
the gate never fires and must reproduce the plain entropy_g arm of E44 run for run. Conditions, seeds, budget
and allocation match E44 exactly, so every comparison is paired by world against the certified arm of Table 1.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e44_provable_denom import CONDS

TAUS = [1.0, 0.10, 0.05, 0.03, 0.02]

if __name__ == "__main__":
    cfgs = [dict(base, seed=s, planner="entropy_g", trust_tau=t, **c)
            for s in range(20) for t in TAUS for base, c in CONDS]
    run_all("e50", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 9)
