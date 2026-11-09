"""E42 analysis: the price of a provable certificate, against E32 in the same regime."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from runner import load
from analysis import ci95

def collect(exp, want_split):
    df, runs = load(exp)
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid")[
        ["c_whole", "cov_explored", "acc_committed"]].mean()
    out = {}
    for rid, r in last.iterrows():
        c = runs[rid]["config"]
        if bool(c.get("split4", False)) != want_split or c["N"] != 3: continue
        if c.get("sigma_e_agents") or c.get("drift_agents") or c.get("range_agents") or c.get("beams_agents"): continue
        pf = tuple(c.get("p_flip_agents") or ())
        cond = "flip" if pf == (0.0, 0.0, 0.10) else ("clean" if not pf else None)
        if cond is None or c["planner"] not in ("certified", "entropy_g"): continue
        out[(cond, c["planner"], c["seed"])] = (r.c_whole, r.cov_explored, r.acc_committed)
    return out

B, S = collect("e32", False), collect("e42", True)
print(f"{'condition':8s} {'planner':10s} {'c_whole base':>13s} {'c_whole split':>14s} {'drop':>8s} {'cov split':>10s}")
for cond in ("clean", "flip"):
    for p in ("certified", "entropy_g"):
        sb = sorted({k[2] for k in B if k[:2] == (cond, p)} & {k[2] for k in S if k[:2] == (cond, p)})
        if not sb: continue
        cb = np.array([B[(cond, p, s)][0] for s in sb]); cs = np.array([S[(cond, p, s)][0] for s in sb])
        cov = np.array([S[(cond, p, s)][1] for s in sb])
        m, h = ci95(cb - cs)
        print(f"{cond:8s} {p:10s} {cb.mean():13.4f} {cs.mean():14.4f} {m:+6.4f}±{h:.4f} {cov.mean():10.4f} (min {cov.min():.4f}, n={len(sb)})")
print()
for cond in ("clean", "flip"):
    for lab, D in (("baseline", B), ("split4 ", S)):
        sa = sorted({k[2] for k in D if k[:2] == (cond, "certified")} & {k[2] for k in D if k[:2] == (cond, "entropy_g")})
        if not sa: continue
        d = np.array([D[(cond, "certified", s)][0] - D[(cond, "entropy_g", s)][0] for s in sa])
        m, h = ci95(d)
        print(f"margin {cond:6s} {lab}: certified - coverage map = {m:+.4f} +- {h:.4f}  (n={len(sa)})")
