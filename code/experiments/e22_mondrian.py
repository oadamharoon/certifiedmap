"""E22: reliability-conditional (Mondrian) certificate on localised-unreliability worlds. Coverage must hold inside
and outside the unreliable region; the certified-area planner reads per-class thresholds (certified_m)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for th, rs in [(0.4, (10.0, 8.0)), (0.5, (12.0, 10.0))]:
            for p in ["certified_m", "entropy", "entropy_u", "frontier"]:
                cfgs.append(dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, p_flip_region=th, region_size=rs, mondrian=True))
        cfgs.append(dict(seed=s, N=1, planner="certified_m", budget=100.0, sigma_r=0.05, mondrian=True))
        cfgs.append(dict(seed=s, N=1, planner="entropy", budget=100.0, sigma_r=0.05, mondrian=True))
    run_all("e22", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
