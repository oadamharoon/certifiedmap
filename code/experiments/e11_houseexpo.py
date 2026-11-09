"""E11 (4.6): E1 on 20 real floor plans (HouseExpo)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e1_objective import PLANNERS
D = os.path.join(os.path.dirname(__file__), "..", "data", "houseexpo")
if __name__ == "__main__":
    cfgs = [dict(seed=k, N=1, planner=p, budget=150.0, sigma_r=0.05, world_file=os.path.abspath(os.path.join(D, f"he{k:02d}.npy")))
            for k in range(20) for p in PLANNERS]
    run_all("e11", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
