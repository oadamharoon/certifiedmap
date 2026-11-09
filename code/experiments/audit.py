"""Re-audit harness: recompute paper quantities from cached runs without loading whole result sets.

`runner.load` keeps every block of every run plus trajectories and maps, which exhausts memory on the larger
experiments. `endpoints` streams the pickles and keeps only the final block of each agent together with the
config, which is what every endpoint claim in the paper is computed from.

Usage:  PYTHONPATH=experiments python experiments/audit.py <exp> [<exp> ...]
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

from analysis import ci95
from compare import holm, paired_t
from runner import RESULTS

METRICS = ("c_whole", "cov_explored", "acc_committed", "ambiguous", "empty", "nodata", "undecided_whole",
           "path_len")


def endpoints(exp: str, metrics=METRICS, cfg_keys=None) -> pd.DataFrame:
    """One row per run: the mean over agents of each metric at that run's last block, plus config."""
    rows = []
    d = os.path.join(RESULTS, exp)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".pkl"):
            continue
        with open(os.path.join(d, fn), "rb") as fh:
            out = pickle.load(fh)
        log, cfg = out["log"], out["config"]
        del out                                   # drop trajectories and maps before the next file
        if not log:
            continue
        # each AGENT's own last block. A global max(block) would drop every agent that finished earlier,
        # biasing the endpoint toward whichever robot ran longest.
        last_of = {}
        for r in log:
            a = r.get("agent", 0)
            if a not in last_of or r["block"] > last_of[a]["block"]:
                last_of[a] = r
        tail = list(last_of.values())
        row = {"rid": fn[:-4]}
        for m in metrics:
            vals = [r[m] for r in tail if m in r and r[m] is not None]
            row[m] = float(np.mean(vals)) if vals else np.nan
        keys = cfg.keys() if cfg_keys is None else cfg_keys
        row.update({f"cfg_{k}": (str(cfg[k]) if isinstance(cfg[k], (list, tuple)) else cfg[k])
                    for k in keys if k in cfg})
        rows.append(row)
    return pd.DataFrame(rows)


def margins(df: pd.DataFrame, cond_col: str, bases: list[str], ref="certified", metric="c_whole",
            planner_col="cfg_planner", seed_col="cfg_seed"):
    """Paired margin of `ref` over each baseline, per condition, with a Holm correction within a condition."""
    out = []
    for cond, d in df.groupby(cond_col, sort=False):
        base = d[d[planner_col] == ref].groupby(seed_col)[metric].mean()
        stats, ps = [], []
        for b in bases:
            o = d[d[planner_col] == b].groupby(seed_col)[metric].mean()
            j = base.index.intersection(o.index)
            if len(j) < 3:
                stats.append((np.nan, np.nan, 0)); ps.append(1.0); continue
            diff = (base.loc[j] - o.loc[j]).values
            m, h = ci95(diff)
            stats.append((m, h, len(j))); ps.append(paired_t(diff))
        for b, (m, h, n), pa in zip(bases, stats, holm(ps)):
            out.append(dict(cond=cond, baseline=b, margin=m, ci=h, p_holm=pa, n=n))
    return pd.DataFrame(out)


def show(name: str, m: pd.DataFrame, bases: list[str], labels=None):
    labels = labels or {b: b for b in bases}
    print(f"\n{name}: paired margin of certified over each baseline (mean +/- 95% CI, Holm p within row)")
    print(f"  {'condition':22s}" + "".join(f"{labels.get(b, b):>30s}" for b in bases))
    for cond in m.cond.unique():
        d = m[m.cond == cond].set_index("baseline")
        cells = []
        for b in bases:
            if b not in d.index or not d.loc[b, "n"]:
                cells.append("--"); continue
            r = d.loc[b]
            cells.append(f"{r.margin:+.3f}+/-{r.ci:.3f} p={r.p_holm:5.3f} n={int(r.n):<3d}")
        print(f"  {str(cond):22s}" + "".join(f"{c:>30s}" for c in cells))


def _main():
    for exp in sys.argv[1:]:
        df = endpoints(exp)
        print(f"\n===== {exp}: {len(df)} runs, planners {sorted(df.cfg_planner.dropna().unique())}")
        print(df.groupby("cfg_planner")[["c_whole", "undecided_whole"]].mean().round(3).to_string())


# ---------------------------------------------------------------------------------------------------
# Paper claim re-audit. Each entry names a quantity asserted in the paper, where it is asserted, the
# asserted value, and how to recompute it from cache. Run `python experiments/audit.py --claims`.
# ---------------------------------------------------------------------------------------------------

def _cov(df, col="cov_whole"):
    return df[col]


def claim_table1():
    """Table 1: both margin columns, recomputed from e32 (N=3 rows) and e33 (drift and N=5 rows)."""
    def c32(f, e):
        f, e = str(f), str(e)
        if "0.1" in f and "0.6" in e: return "two fail differently"
        if "0.6" in e: return "one misregisters (0.6 m)"
        if "0.3" in e: return None
        if "0.1" in f: return "one flips 10% of labels"
        return "all robots reliable"

    d32 = endpoints("e32"); d32 = d32[d32.cfg_N == 3].copy()
    d32["cond"] = [c32(a, b) for a, b in zip(d32.cfg_p_flip_agents, d32.cfg_sigma_e_agents)]

    d33 = endpoints("e33")
    def c33(r):
        d, f, N = str(r.cfg_drift_agents), str(r.cfg_p_flip_agents), r.cfg_N
        st = r.get("cfg_start_seed", -1)
        if st == st and st is not None and st > 0: return None
        nd = sum(1 for x in d.strip("()").replace(" ", "").split(",") if x and float(x) > 0)
        nf = sum(1 for x in f.strip("()").replace(" ", "").split(",") if x and float(x) > 0)
        if N == 3:
            return "one drifts (0.05 m/block)" if (nd and not nf and "0.05" in d) else None
        if nd: return None
        return {1: "five robots, one unreliable", 2: "five robots, two unreliable"}.get(nf)
    d33["cond"] = d33.apply(c33, axis=1)

    keep = ["cond", "cfg_planner", "cfg_seed", "c_whole", "undecided_whole"]
    both = pd.concat([d32[keep], d33[keep]])
    both = both[both["cond"].notna()]
    ORDER = ["all robots reliable", "one flips 10% of labels", "one misregisters (0.6 m)",
             "one drifts (0.05 m/block)", "two fail differently",
             "five robots, one unreliable", "five robots, two unreliable"]
    both["cond"] = pd.Categorical(both["cond"], ORDER, ordered=True)
    return both.sort_values("cond")


# Asserted values, transcribed once from main.tex with the source column named, so a mismatch is visible
# rather than silently absorbed. (condition, margin vs coverage map, margin vs uncoordinated, uncertified pair)
TABLE1_ASSERTED = {
    "all robots reliable":         (+0.014, +0.012, (3.6, 2.2)),
    "one flips 10% of labels":     (+0.075, +0.032, (11.3, 3.8)),
    "one misregisters (0.6 m)":    (+0.054, +0.020, (9.8, 4.4)),
    "one drifts (0.05 m/block)":   (+0.015, +0.023, (5.4, 3.9)),
    "two fail differently":        (+0.130, +0.022, (25.3, 12.3)),
    "five robots, one unreliable": (+0.051, +0.002, (7.9, 2.7)),
    "five robots, two unreliable": (+0.091, +0.009, (12.7, 3.6)),
}


def audit_table1(tol=0.004):
    """The paper's "uncertified" column is the share of observable cells that are NOT committed, which is
    undecided + empty, equivalently 1 - c_whole. Comparing it against undecided_whole alone agrees to within
    0.2 points wherever "empty" is small and silently disagrees where it is not, which is how the drift row
    (empty 1.2-1.5%) was once "corrected" to a wrong value. The cross-check below is the one that cannot drift:
    a row's margin over the coverage-map planner must equal its uncertified difference, since both are 1 - c.
    """
    both = claim_table1()
    m = margins(both, "cond", ["entropy_g", "entropy"])
    print("\n" + "=" * 104)
    print("TABLE 1 re-audit. 'asserted' is what main.tex prints; 'recomputed' comes from the cached runs.")
    print(f"  {'condition':28s} {'col2 asrt':>9s} {'recomp':>8s} {'col3 asrt':>10s} {'recomp':>8s} "
          f"{'uncert asrt':>12s} {'recomp':>12s}  verdict")
    bad = 0
    for cond, (a2, a3, (u_cov, u_cert)) in TABLE1_ASSERTED.items():
        r = m[m.cond == cond].set_index("baseline")
        r2, r3 = r.loc["entropy_g", "margin"], r.loc["entropy", "margin"]
        d = both[both["cond"] == cond]
        rc = (1 - d[d.cfg_planner == "entropy_g"].c_whole.mean()) * 100
        rt = (1 - d[d.cfg_planner == "certified"].c_whole.mean()) * 100
        ok = abs(r2 - a2) <= tol and abs(r3 - a3) <= tol
        okun = abs(rc - u_cov) <= 0.1 and abs(rt - u_cert) <= 0.1
        # The printed column must reproduce the printed margin. Budget for display rounding only: each
        # percentage carries +/-0.05 and the margin +/-0.0005, so 0.0015 is the most rounding can explain.
        # The drift row's old value missed by 0.003, twice that, which is what makes this check worth running.
        selfc = abs((u_cov - u_cert) / 100 - a2) <= 0.0015
        v = "ok" if ok and okun and selfc else ("MISMATCH" if not ok else
            ("UNCERT off" if not okun else "COLUMN CONTRADICTS ITS OWN MARGIN"))
        bad += 0 if (ok and okun and selfc) else 1
        print(f"  {cond:28s} {a2:+9.3f} {r2:+8.3f} {a3:+10.3f} {r3:+8.3f} "
              f"{u_cov:6.1f}->{u_cert:<5.1f} {rc:6.1f}->{rt:<5.1f}  {v}")
    print(f"  rows needing attention: {bad} of {len(TABLE1_ASSERTED)}")
    return m


def audit_validity():
    """Sec. 6: the weighted certificate covers at nominal and the unweighted one does not (e2, N=1).

    Two selections matter and neither is the default. The query law is uniform on the GATED region, so the
    metric is cov_explored, not cov_whole. And Fig. 1 averages the two adaptive acquisitions only: e2 also
    runs a non-adaptive lawnmower, whose inclusion changes the counts the paper reports.
    """
    df = endpoints("e2", metrics=("cov_explored", "c_whole"), cfg_keys=("weighted", "alpha", "planner", "seed"))
    df = df[df.cfg_planner.isin(["certified", "entropy"])]
    print("\n" + "=" * 104)
    print("VALIDITY re-audit (e2): realized coverage on the query law against nominal 1-alpha")
    print(f"  {'weights':9s} {'1-alpha':>8s} {'mean cov':>9s} {'worst run':>10s} {'runs below nominal':>19s} {'n':>5s}")
    for wt in (True, False):
        for a in sorted(df.cfg_alpha.unique()):
            d = df[(df.cfg_weighted == wt) & (df.cfg_alpha == a)]
            if not len(d): continue
            nom = 1 - a
            below = int((d.cov_explored < nom).sum())
            print(f"  {str(wt):9s} {nom:8.2f} {d.cov_explored.mean():9.4f} {d.cov_explored.min():10.4f} "
                  f"{below:12d}/{len(d):<6d} {len(d):5d}")
    return df


def audit_split4():
    """Sec. 4: the four-fold variant holds the level and costs certified fraction (e43)."""
    df = endpoints("e43", metrics=METRICS + ("cov_whole",), cfg_keys=("split4", "planner", "seed", "N"))
    print("\n" + "=" * 104)
    print("FOUR-FOLD re-audit (e43): certified fraction with and without the sample split")
    for p in sorted(df.cfg_planner.dropna().unique()):
        d = df[df.cfg_planner == p]
        for sp in (False, True):
            dd = d[d.cfg_split4 == sp]
            if not len(dd): continue
            print(f"  planner {p:10s} split4={str(sp):5s} c_whole {dd.c_whole.mean():.3f} "
                  f"cov {dd.cov_whole.mean():.4f}  n={len(dd)}")
        a = d[d.cfg_split4 == False].groupby("cfg_seed").c_whole.mean()
        b = d[d.cfg_split4 == True].groupby("cfg_seed").c_whole.mean()
        j = a.index.intersection(b.index)
        if len(j) >= 3:
            m, h = ci95((a.loc[j] - b.loc[j]).values)
            print(f"    cost of the split for {p}: {m:.3f} +/- {h:.3f} of certified fraction (paper: 0.017 to 0.092)")
    return df


if __name__ == "__main__":
    if "--claims" in sys.argv:
        audit_table1(); audit_validity(); audit_split4()
    else:
        _main()
