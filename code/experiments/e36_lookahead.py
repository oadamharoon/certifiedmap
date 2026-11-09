"""E36 (item 4): does a non-myopic planner change the single-robot conclusion?

The project's central negative is that swapping the acquisition from entropy to certified does not change what
gets certified, with everything else held fixed. One reading of that is that a one-segment horizon is too short
for a certification-specific detour to pay. This runs the same acquisitions under a receding-horizon beam search
over 1, 2 and 3 segments (only the first is executed), so the horizon is the only thing that varies.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

if __name__ == "__main__":
    cfgs = [dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, lookahead=la, beam=3)
            for s in range(20) for p in ["certified", "entropy"] for la in [1, 2, 3]]
    run_all("e36", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
