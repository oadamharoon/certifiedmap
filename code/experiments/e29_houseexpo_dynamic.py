"""E29 (Gate 3b): real floor plans (HouseExpo, 20 maps) with six goal-directed 1.5 m hazards (pedestrian-sized,
persistent motion between re-sampled goals) at 1 and 2 m/block; seven guards plus certified at 0.02; 150 m."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e26_gate2 import GUARDS
D = os.path.join(os.path.dirname(__file__), "..", "data", "houseexpo")
if __name__ == "__main__":
    base = dict(N=1, budget=150.0, sigma_r=0.05, n_hazards=6, hazard_size=1.5, hazard_motion="goal", hazards_move_after_plan=True,
                K=120, planner="entropy")
    cfgs = []
    for k in range(20):
        wf = os.path.abspath(os.path.join(D, f"he{k:02d}.npy"))
        for st in [1.0, 2.0]:
            for g in GUARDS:
                cfgs.append(dict(base, seed=k, world_file=wf, hazard_step=st, guard=g))
            cfgs.append(dict(base, seed=k, world_file=wf, hazard_step=st, guard="certified", guard_alpha=0.02))
    run_all("e29", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
