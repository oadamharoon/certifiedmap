"""E19: coordinated-frontier baselines in teams. frontier_g = nearest frontier on the shared coverage mask with
target deconfliction (Burgard-style map sharing); frontier_u = the same rule reading the fused undecided set.
Same configurations as the team (E3) and heterogeneous-team (E12) experiments."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for N, top in [(1, "mesh"), (2, "mesh"), (3, "mesh"), (5, "mesh"), (5, "ring")]:
            for p in ["frontier_g", "frontier_u"]:
                cfgs.append(dict(seed=s, N=N, topology=top, planner=p, budget=100.0 / N, sigma_r=0.05))
        for flips in [[0.0, 0.0, 0.05], [0.0, 0.0, 0.10], [0.0, 0.05, 0.10]]:
            for p in ["frontier_g", "frontier_u"]:
                cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0 / 3, sigma_r=0.05, p_flip_agents=flips))
        for p in ["frontier_g", "frontier_u"]:
            cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0 / 3, sigma_r=0.05, range_agents=[5.0, 5.0, 2.5]))
            cfgs.append(dict(seed=s, N=3, topology="mesh", planner=p, budget=100.0 / 3, sigma_r=0.05, beams_agents=[40, 40, 8]))
    run_all("e19", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
