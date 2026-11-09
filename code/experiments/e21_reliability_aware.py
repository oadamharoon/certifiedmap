"""E21: reliability-aware certified acquisition (local contradiction-rate map caps the attainable |mu| at 1 - 2 theta_hat)
on the localised-unreliability worlds of E20, plus the clean worlds as a control (it must not lose there)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner="certified_r", budget=100.0, sigma_r=0.05, p_flip_region=th) for s in range(20) for th in [0.2, 0.4]]
    cfgs += [dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, p_flip_region=0.5, region_size=(12.0, 10.0)) for s in range(20) for p in ["certified_r", "certified", "entropy", "entropy_u", "lse", "frontier"]]
    cfgs += [dict(seed=s, N=1, planner="certified_r", budget=100.0, sigma_r=0.05) for s in range(20)]
    run_all("e21", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
