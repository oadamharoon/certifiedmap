"""E47 analysis: does clipping the importance-weight tail buy back Prop. 4.3's correction?

Decision rule, fixed in advance (see e47_wclip.py): adopt only if coverage stays at or above nominal at
every level and every heterogeneity margin stays inside its interval; otherwise report the correction as is.
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
ORDER = ["clean", "label flip", "misregisters", "drift", "two fail differently",
         "five, one bad", "five, two bad"]
DELTA = 0.01


def key_of(c):
    return (c["N"], tuple(c.get("p_flip_agents") or ()), bool(c.get("sigma_e_agents")),
            bool(c.get("drift_agents")))


def eta_exact(js, L):
    js = np.asarray(js, float); a = js + L
    return (a + np.sqrt(np.maximum(a * a - js * js, 0.0))) / js - 1


def last_per_run(df, cols):
    return df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid")[cols].mean()


def main():
    df, runs = load("e47")
    base_df, base_runs = load("e44")

    # ---- (a) validity sweep, single robot, clip 0.90 and 0.80 ----
    per = last_per_run(df, ["cov_explored", "n_w", "eps_i"])
    val = {}
    for rid, r in per.iterrows():
        c = runs[rid]["config"]
        if c["N"] != 1:
            continue
        val.setdefault((float(c["w_clip"]), float(c["alpha"])), []).append(r.cov_explored)
    print("(a) coverage on the query law, one robot, against nominal")
    print(f"{'clip':>6} {'nominal':>9} {'coverage':>18} {'runs':>5}  verdict")
    ok_cov = True
    for clip in sorted({k[0] for k in val}):
        for a in sorted({k[1] for k in val}, reverse=True):
            v = np.array(val.get((clip, a), []))
            if not len(v):
                continue
            m, h = ci95(v)
            nom = 1 - a
            good = m >= nom
            ok_cov &= bool(good)
            print(f"{clip:>6.2f} {nom:>9.2f} {m:>11.4f} +/-{h:.4f} {len(v):>5}  "
                  f"{'ok' if good else 'BELOW NOMINAL'}")

    # ---- (b) heterogeneity margins at clip 0.90 vs the w_clip=0 arm of e44 ----
    def margins(d, rs, want_clip):
        per = last_per_run(d, ["c_whole"])
        out = {}
        for rid, r in per.iterrows():
            c = rs[rid]["config"]
            if float(c.get("denom_delta", 0.0)) != 0.0:
                continue
            if want_clip is not None and float(c.get("w_clip", 0.0)) != want_clip:
                continue
            cond = NAMES.get(key_of(c))
            if cond is None:
                continue
            out[(cond, c["planner"], c["seed"])] = r.c_whole
        return out

    new = margins(df, runs, 0.90)
    old = margins(base_df, base_runs, 0.0)
    print("\n(b) certified minus coverage-map margin, clip 0.90 against the deployed arm")
    print(f"{'condition':22s} {'deployed':>18} {'clipped':>18}  verdict")
    ok_marg = True
    for cond in ORDER:
        seeds = sorted({k[2] for k in old if k[0] == cond} & {k[2] for k in new if k[0] == cond})
        if not seeds:
            continue
        mo = np.array([old[(cond, "certified", s)] - old[(cond, "entropy_g", s)] for s in seeds
                       if (cond, "certified", s) in old and (cond, "entropy_g", s) in old])
        mn = np.array([new[(cond, "certified", s)] - new[(cond, "entropy_g", s)] for s in seeds
                       if (cond, "certified", s) in new and (cond, "entropy_g", s) in new])
        if not len(mo) or not len(mn):
            continue
        a_m, a_h = ci95(mo); b_m, b_h = ci95(mn)
        inside = (b_m >= a_m - a_h) and (b_m <= a_m + a_h)
        ok_marg &= bool(inside)
        print(f"{cond:22s} {a_m:>+9.3f} +/-{a_h:.3f} {b_m:>+9.3f} +/-{b_h:.3f}  "
              f"{'inside' if inside else 'OUTSIDE'}  (n={len(mn)})")

    # ---- (c) what clipping buys on the correction ----
    print("\n(c) the correction, clipped against deployed")
    for tag, d, rs, clip in (("deployed (e45-style, e44 arm)", base_df, base_runs, 0.0),
                             ("clipped 0.90", df, runs, 0.90)):
        cols = ["n_w", "eps_i"] if "n_w" in d.columns else None
        if cols is None:
            print(f"  {tag:30s} n_w not logged in this cache")
            continue
        per = last_per_run(d, cols)
        by = {}
        for rid, r in per.iterrows():
            c = rs[rid]["config"]
            if float(c.get("denom_delta", 0.0)) != 0.0 or float(c.get("w_clip", 0.0)) != clip:
                continue
            if NAMES.get(key_of(c)) is None:
                continue
            by.setdefault(c["N"], []).append(r)
        for N in sorted(by):
            R = by[N]; nw = np.array([r.n_w for r in R]); js = np.array([r.eps_i * r.n_w for r in R])
            e = eta_exact(js, np.log(N / DELTA))
            print(f"  {tag:30s} N={N}  n~={nw.mean():7.1f}  j*={js.mean():6.2f}  "
                  f"1+eta={1 + e.mean():.2f}  bound={(1 + e.mean()) * 0.05 + DELTA:.3f}")

    print(f"\nPRE-REGISTERED RULE: coverage ok = {ok_cov}; margins inside = {ok_marg}")
    print("ADOPT" if (ok_cov and ok_marg) else "DO NOT ADOPT (report the correction as it stands)")


if __name__ == "__main__":
    main()
