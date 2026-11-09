"""E24: lapsing support under fast hazards. Gate lapses after `gate_window` blocks so unrevisited cells return to
undecided; planners that read the certified state should re-certify them, variance-driven ones should not."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = []
    for s in range(10):
        for gw in [0, 15]:
            for p in ["certified", "entropy", "entropy_u", "frontier"]:
                cfgs.append(dict(seed=s, N=1, planner=p, budget=400.0, sigma_r=0.05, n_hazards=20, hazard_size=3.0, hazard_step=3.0, gate_window=gw, K=100))
    run_all("e24", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
