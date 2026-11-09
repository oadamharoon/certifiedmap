"""E31: (a) ablation of the certificate's settings at the main setting (10 random-walk hazards, speeds 1 and 3, 300 m,
20 worlds): warm-up 5/20 (10 in E26), probe never/after 10 (3), evidence window 10/40 (20), decay rho 0.9 (1),
no region split (change_tau 1.0), observation radius 0.25/1.0 m (0.5). (b) tracker with oracle detections."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from runner import run_all
from e26_gate2 import BASE
VARIANTS = {"warmup5": dict(warmup=5), "warmup20": dict(warmup=20), "probe_never": dict(probe_after=0), "probe10": dict(probe_after=10),
            "window10": dict(horizon_window=10), "window40": dict(horizon_window=40), "rho0.9": dict(horizon_rho=0.9),
            "no_region": dict(change_tau=1.0), "obs0.25": dict(obs_r=0.25), "obs1.0": dict(obs_r=1.0)}
if __name__ == "__main__":
    base = dict(BASE, planner="entropy", n_hazards=10)
    cfgs = []
    for s in range(20):
        for st in [1.0, 3.0]:
            for name, kv in VARIANTS.items():
                cfgs.append(dict(base, seed=s, hazard_step=st, guard="certified", **kv))
            cfgs.append(dict(base, seed=s, hazard_step=st, guard="cp_track_oracle"))
    run_all("e31", cfgs, workers=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
