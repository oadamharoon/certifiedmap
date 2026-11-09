"""E40: does the unequal-reliability result survive a sparser or irregular communication graph?

Every run reported so far fuses over a complete graph (topology="mesh"), so d_i = N for every robot and the
degree dependence of beta_i = s*alpha/d_i is never exercised. This sweeps the graph at the condition that carries
the result (N=5, one robot flipping 10 % of labels) over

  mesh  complete, d_i = 5 for all          (the setting of E33; cached, re-listed here so the sweep is paired)
  ring  d_i = 3 for all                    regular but sparse
  star  hub d_0 = 5, leaves d_i = 2        irregular; Prop. 4.3(ii) only guarantees the union property when
                                           the rejecting neighbour has d_j >= d_i, which fails for a leaf
                                           deciding for the hub

and places the unreliable robot both at a leaf (index 4) and at the hub (index 0), with the matching mesh control
for the hub placement so the two are paired. Two quantities matter: the certified fraction against the
coverage-map planner (does the gain survive?) and coverage on the query law (Prop. 4.3(i) claims validity for
*every* graph, which no run has tested).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all

BASE = dict(N=5, budget=100.0, sigma_r=0.05, K=120)
LEAF = (0.0, 0.0, 0.0, 0.0, 0.10)       # unreliable robot is a leaf of the star
HUB = (0.10, 0.0, 0.0, 0.0, 0.0)        # unreliable robot is the hub
PLANNERS = ["certified", "entropy_g"]

if __name__ == "__main__":
    cfgs = []
    for s in range(20):
        for p in PLANNERS:
            for topo in ["mesh", "ring", "star"]:
                cfgs.append(dict(BASE, seed=s, planner=p, topology=topo, p_flip_agents=LEAF))
            for topo in ["mesh", "star"]:
                cfgs.append(dict(BASE, seed=s, planner=p, topology=topo, p_flip_agents=HUB))
    run_all("e40", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
