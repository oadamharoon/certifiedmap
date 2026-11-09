"""E20: spatially localised unreliability. A random 10x8 m region flips sensed labels with probability theta_R;
elsewhere labels are clean. Single robot, 100 m. Does an acquisition that values certification avoid sinking budget
into the uncertifiable region, and does that show in certified fraction?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, p_flip_region=th)
            for s in range(20) for th in [0.2, 0.4] for p in ["certified", "entropy", "entropy_u", "lse", "frontier"]]
    run_all("e20", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
