"""E44 analysis: the price of an exact-alpha denominator, paired by world against the deployed certificate."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from runner import load
from analysis import ci95

NAMES = {(3, (), False, False): "clean",
         (3, (0.0, 0.0, 0.1), False, False): "label flip",
         (3, (), True, False): "misregisters",
         (3, (), False, True): "drift",
         (3, (0.0, 0.1, 0.0), True, False): "two fail differently",
         (5, (0.0, 0.0, 0.0, 0.0, 0.1), False, False): "five, one bad",
         (5, (0.0, 0.0, 0.0, 0.1, 0.1), False, False): "five, two bad"}
ORDER = ["clean", "label flip", "misregisters", "drift", "two fail differently",
         "five, one bad", "five, two bad"]

df, runs = load("e44")
per = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid")[
    ["c_whole", "cov_explored", "acc_committed", "n_eff", "eps_prov", "eps_i", "a_i", "n_recal"]].mean()
D = {}
for rid, r in per.iterrows():
    c = runs[rid]["config"]
    key = (c["N"], tuple(c.get("p_flip_agents") or ()), bool(c.get("sigma_e_agents")), bool(c.get("drift_agents")))
    cond = NAMES.get(key)
    if cond is None: continue
    D[(cond, c["planner"], float(c["denom_delta"]), c["seed"])] = r

print("n_eff of the deployed certificate (delta=0 arm), and the implied correction factor eta:")
for cond in ORDER:
    rs = [v for k, v in D.items() if k[0] == cond and k[2] == 0.0 and k[1] == "certified"]
    if not rs: continue
    ne = np.array([r.n_eff for r in rs]); nr = np.array([r.n_recal for r in rs])
    js = np.array([r.eps_i * r.n_eff for r in rs]); L = np.log(1 / 0.005)
    eta = np.sqrt(2 * L / js) + 2 * L / js
    print(f"  {cond:22s} n_recal {nr.mean():6.0f}  n_eff {ne.mean():6.0f} ({ne.mean()/nr.mean():.2f})"
          f"  j* {js.mean():5.1f}  eta {eta.mean():.2f}")

print("\npaired cost of the exact-alpha denominator (delta=0.005):")
hdr = f"{'condition':22s} {'planner':10s} {'c base':>8s} {'c exact':>8s} {'drop':>16s} {'cov exact':>10s} {'worst':>7s}"
print(hdr)
for cond in ORDER:
    for p in ("certified", "entropy_g"):
        sb = sorted({k[3] for k in D if k[:3] == (cond, p, 0.0)} & {k[3] for k in D if k[:3] == (cond, p, 0.005)})
        if not sb: continue
        cb = np.array([D[(cond, p, 0.0, s)].c_whole for s in sb])
        cx = np.array([D[(cond, p, 0.005, s)].c_whole for s in sb])
        cv = np.array([D[(cond, p, 0.005, s)].cov_explored for s in sb])
        m, h = ci95(cb - cx)
        print(f"  {cond:22s} {p:10s} {cb.mean():8.3f} {cx.mean():8.3f} {m:+8.3f} +/-{h:5.3f}"
              f" {cv.mean():10.4f} {cv.min():7.4f}")

print("\ncertified minus coverage-map margin, both arms (the headline quantity):")
print(f"{'condition':22s} {'margin base':>18s} {'margin exact':>18s}")
for cond in ORDER:
    sb = sorted({k[3] for k in D if k[:3] == (cond, "certified", 0.0)}
                & {k[3] for k in D if k[:3] == (cond, "entropy_g", 0.0)}
                & {k[3] for k in D if k[:3] == (cond, "certified", 0.005)}
                & {k[3] for k in D if k[:3] == (cond, "entropy_g", 0.005)})
    if not sb: continue
    b = np.array([D[(cond, "certified", 0.0, s)].c_whole - D[(cond, "entropy_g", 0.0, s)].c_whole for s in sb])
    x = np.array([D[(cond, "certified", 0.005, s)].c_whole - D[(cond, "entropy_g", 0.005, s)].c_whole for s in sb])
    mb, hb = ci95(b); mx, hx = ci95(x)
    print(f"  {cond:22s} {mb:+10.3f} +/-{hb:5.3f} {mx:+10.3f} +/-{hx:5.3f}")
