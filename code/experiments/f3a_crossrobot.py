"""F3a: is there a cross-robot signal for drift at all?

Pre-registered screen for follow-up route R3 (a certificate for drift / consistent error). E50 showed the
self-estimated label-error rate p_hat separates the drifting robot from its clean teammates by only 1.44x,
against 1.43x on a clean team: a uniformly displaced map contradicts nothing locally, so the detector is blind
to exactly the fault the certificate is blind to. Drift should only be visible relative to another robot.

This measures the one quantity the route rests on. Each robot's own single-robot certificate commits a label
where its e-value clears the Markov threshold; on cells where two robots' gates overlap and both commit, the
rate at which their committed labels differ is the cross-robot signal. Conditions, seeds, budget, topology and
planner match the clean and drift rows of E44, so the drifting pair and the clean pair are compared within the
same world.

Decision rule, fixed in advance: R3 is feasible if the drifting pair's disagreement rate exceeds the clean
pair's by at least 3x. Below that, inter-robot label disagreement is not the signal either, and the route needs
explicit relative-pose estimation, which is a much larger project against mature incumbents (PCM,
Kimera-Multi).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

B3 = dict(N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120, planner="certified", save_maps=True)
CONDS = [("clean", dict()), ("drift", dict(drift_agents=(0.0, 0.0, 0.05)))]
SEEDS = range(6)

if __name__ == "__main__":
    cfgs = [dict(B3, seed=s, **c) for s in SEEDS for _, c in CONDS]
    run_all("f3a", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
