"""E31b: the tracker guards (map detections and oracle detections) under goal-directed hazards, where constant-velocity
prediction is at its best; 10 hazards, speeds 1 and 3, 20 worlds (certified/fresh from E27)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e26_gate2 import BASE
if __name__ == "__main__":
    base = dict(BASE, planner="entropy", n_hazards=10, hazard_motion="goal")
    cfgs = [dict(base, seed=s, hazard_step=st, guard="cp_track_oracle") for s in range(20) for st in [1.0, 3.0]]
    run_all("e27", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
