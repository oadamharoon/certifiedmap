import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
import e7_ablation as e7
if __name__ == "__main__":
    cfgs = []
    for s in e7.SEEDS:
        for p in ["certified", "entropy"]:
            for key, vals in e7.LEVELS.items():
                for v in vals:
                    cfgs.append(dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, **{key: v}))
        for dm in [1.0, 5.0]:
            cfgs.append(dict(seed=s, N=1, planner="frontier", budget=100.0, sigma_r=0.05, front_dmin=dm))
        for p in ["certified", "entropy", "frontier"]:
            cfgs.append(dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05))
    run_all("e7", cfgs[::-1], workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
