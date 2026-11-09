"""E44: what does an exact-alpha certificate cost?

Lemma 1 bounds the true average miscoverage at the selected level a by z/n_eff, where z is the larger root of
(z - j*)^2 = 2 z log(1/delta) and j* = eps * n_eff. Dividing the e-value by that bound rather than by the
measured miscoverage eps makes E[e] <= 1 outright, so coverage is alpha + delta with no leftover factor; the
deployed certificate divides by eps and carries the factor z/j* on the level instead.

The factor is large precisely because beta = s*alpha/d_i is split d_i ways, and at N=5 it exceeds the
single-robot commitment headroom 1/s = 1.43, so the algebra predicts this variant cannot preserve the mechanism
behind the heterogeneity margin. This measures that prediction on every condition of the heterogeneity table.

Both arms run here so the comparison is paired by world and needs no cross-experiment config matching. The
delta=0 arm reproduces the deployed certificate bit for bit and also supplies the n_eff used to evaluate the
correction factor quoted in the paper.

Decision rule, fixed in advance: adopt only if coverage stays at or above nominal and the absolute certified
fraction falls by at most 0.02 with the certified minus coverage-map margin surviving its interval; otherwise
report as a measured ablation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

B3 = dict(N=3, topology="mesh", budget=100.0, sigma_r=0.05, K=120)
B5 = dict(N=5, topology="mesh", budget=100.0, sigma_r=0.05, K=120)
CONDS = [
    (B3, dict()),                                                                              # clean
    (B3, dict(p_flip_agents=(0.0, 0.0, 0.10))),                                                # label flip
    (B3, dict(sigma_e_agents=(0.04, 0.04, 0.60), sigma_theta_agents=(0.003, 0.003, 0.10))),    # misregistration
    (B3, dict(drift_agents=(0.0, 0.0, 0.05))),                                                 # drift
    (B3, dict(p_flip_agents=(0.0, 0.10, 0.0), sigma_e_agents=(0.04, 0.04, 0.60),
              sigma_theta_agents=(0.003, 0.003, 0.10))),                                       # two fail differently
    (B5, dict(p_flip_agents=(0.0, 0.0, 0.0, 0.0, 0.10))),                                      # five, one unreliable
    (B5, dict(p_flip_agents=(0.0, 0.0, 0.0, 0.10, 0.10))),                                     # five, two unreliable
]
PLANNERS = ["certified", "entropy_g"]
DELTAS = [0.0, 0.005]

if __name__ == "__main__":
    cfgs = [dict(base, seed=s, planner=p, denom_delta=d, **c)
            for s in range(20) for p in PLANNERS for d in DELTAS for base, c in CONDS]
    run_all("e44", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
