"""E7 (4.4): one-factor ablations around the operating point (single robot, sigma_r=0.05, 100 m)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
SEEDS = list(range(20))
LEVELS = {"alpha": [0.02, 0.1, 0.2], "gate_r": [0.35, 1.0], "safety": [0.5, 0.9], "noise_var": [0.1, 1.0],
          "seg_len": [2.5, 10.0], "kde_bw": [0.25, 1.0], "eps_ucb": [0.5, 1.0], "ell": [0.3, 0.6],
          "trav_thr": [0.0, 0.5], "trav_inflate": [1]}
if __name__ == "__main__":
    cfgs = []
    for s in SEEDS:
        for p in ["certified", "entropy"]:
            for key, vals in LEVELS.items():
                for v in vals:
                    cfgs.append(dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, **{key: v}))
        for dm in [1.0, 5.0]:
            cfgs.append(dict(seed=s, N=1, planner="frontier", budget=100.0, sigma_r=0.05, front_dmin=dm))
        # reference (operating point) for the same seeds/conditions
        for p in ["certified", "entropy", "frontier"]:
            cfgs.append(dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05))
    run_all("e7", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
