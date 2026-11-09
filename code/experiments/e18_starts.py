"""E18: sensitivity of the team result to robot placement. Fixed worlds, start poses resampled 8 times each.
Homogeneous N=3 mesh: certified vs entropy (plans) vs entropy_g (mask). Heterogeneous (flip 0.10 on robot 3): certified vs entropy_g."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = []
    for w in range(5):
        for st in range(8):
            base = dict(seed=w, start_seed=100 * w + st, N=3, topology="mesh", budget=100.0 / 3, sigma_r=0.05)
            for p in ["certified", "entropy", "entropy_g"]:
                cfgs.append(dict(base, planner=p))
            for p in ["certified", "entropy_g"]:
                cfgs.append(dict(base, planner=p, p_flip_agents=[0.0, 0.0, 0.10]))
    run_all("e18", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
