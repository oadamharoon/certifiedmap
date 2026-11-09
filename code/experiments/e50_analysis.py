"""E50 analysis: the certified state against a coverage map with a reliability detector.

Three arms, paired by world: the certified planner and the plain coverage-map planner (both from E44's
denom_delta=0 arm, which reproduces Table 1), and the coverage-map planner gated by each teammate's
self-estimated label-error rate p_hat at threshold tau (E50).

tau = 1.0 is a control in which the gate never fires; it must reproduce the plain arm run for run, which
validates that the patch is inert when inactive. 'left uncertified' is 1 - c_whole, the unit Table 1 reports.
"""
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
ORDER = list(NAMES.values())


def key_of(c):
    return (c["N"], tuple(c.get("p_flip_agents") or ()), bool(c.get("sigma_e_agents")),
            bool(c.get("drift_agents")))


def collect(exp, want):
    df, runs = load(exp)
    per = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid")[["c_whole"]].mean()
    out = {}
    for rid, r in per.iterrows():
        c = runs[rid]["config"]
        nm = NAMES.get(key_of(c))
        if nm is None:
            continue
        k = want(c)
        if k is None:
            continue
        out[(nm, c["seed"], k)] = float(r.c_whole)
    return out


def main():
    base = collect("e44", lambda c: c["planner"] if float(c.get("denom_delta", 0.0)) == 0.0 else None)
    try:
        trust = collect("e50", lambda c: float(c["trust_tau"]))
    except Exception as e:
        print("e50 not loadable yet:", e); return
    taus = sorted({k[2] for k in trust})
    seeds_by = lambda d, nm, k: {s for (n, s, kk) in d if n == nm and kk == k}

    # ---- control: tau = 1.0 must reproduce the plain coverage-map arm ----
    if 1.0 in taus:
        diffs = []
        for nm in ORDER:
            for s in seeds_by(trust, nm, 1.0) & seeds_by(base, nm, "entropy_g"):
                diffs.append(trust[(nm, s, 1.0)] - base[(nm, s, "entropy_g")])
        if diffs:
            d = np.array(diffs)
            print(f"CONTROL tau=1.0 vs plain entropy_g: n={len(d)}  max|diff|={np.abs(d).max():.2e}  "
                  f"{'INERT (patch verified)' if np.abs(d).max() < 1e-9 else 'DIFFERS -- investigate'}")
    print()

    # ---- margins, paired by world ----
    hdr = f"{'condition':22s} {'cert-plain (Table 1)':>22s}"
    for t in [x for x in taus if x < 1.0]:
        hdr += f" {'cert-gated t=' + str(t):>20s}"
    print(hdr)
    for nm in ORDER:
        S = seeds_by(base, nm, "certified") & seeds_by(base, nm, "entropy_g")
        if not S:
            continue
        m0 = np.array([base[(nm, s, "certified")] - base[(nm, s, "entropy_g")] for s in sorted(S)])
        a, h = ci95(m0)
        row = f"{nm:22s} {a:>+12.3f} +/-{h:.3f}"
        for t in [x for x in taus if x < 1.0]:
            T = S & seeds_by(trust, nm, t)
            if not T:
                row += f" {'-':>20s}"; continue
            m1 = np.array([base[(nm, s, "certified")] - trust[(nm, s, t)] for s in sorted(T)])
            b, hh = ci95(m1)
            row += f" {b:>+10.3f} +/-{hh:.3f} ({len(T):2d})"
        print(row)

    # ---- the operator-facing unit: share of the map left uncertified ----
    # Each tau column is computed on the seeds that tau actually has, and the certified and coverage-map
    # columns are recomputed on that same seed set, so a partially finished sweep cannot compare a 20-world
    # mean against a 1-world mean.
    for t in [x for x in taus if x < 1.0]:
        print(f"\nleft uncertified (1 - certified fraction), gate at tau={t}, paired seeds only")
        print(f"{'condition':22s} {'n':>3s} {'certified':>10s} {'coverage map':>13s} {'gated':>8s}")
        for nm in ORDER:
            T = (seeds_by(base, nm, "certified") & seeds_by(base, nm, "entropy_g")
                 & seeds_by(trust, nm, t))
            if not T:
                continue
            ss = sorted(T)
            c = 1 - np.mean([base[(nm, s, "certified")] for s in ss])
            pm = 1 - np.mean([base[(nm, s, "entropy_g")] for s in ss])
            g = 1 - np.mean([trust[(nm, s, t)] for s in ss])
            print(f"{nm:22s} {len(ss):>3d} {100*c:>9.1f}% {100*pm:>12.1f}% {100*g:>7.1f}%")


if __name__ == "__main__":
    main()
