"""E35 (item 3): the published candidate generator.

RA-L multi-robot exploration papers compare against RRT-exploration (Umari & Mukhopadhyay 2017) and relatives;
our planners all share one candidate generator (bearing-binned shortest paths plus frontier targets), so a
tree-based generator is the comparison the literature expects. Here the generator is swapped under every
acquisition, single robot and three-robot teams, clean and with one unreliable teammate, so it also checks that
the heterogeneity result does not depend on our generator.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

PLANNERS = ["certified", "entropy", "entropy_u", "entropy_g", "frontier"]
if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for gen in ["sectors", "rrt"]:
            for p in PLANNERS:
                cfgs.append(dict(seed=s, N=1, planner=p, budget=100.0, sigma_r=0.05, generator=gen))
                cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0, sigma_r=0.05, generator=gen))
                cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0, sigma_r=0.05, generator=gen,
                                 p_flip_agents=(0.0, 0.0, 0.10)))
    run_all("e35", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
