"""Analysis for the dynamic line (E26 guards/density/acquisitions, E27 frontier + goal motion, E28 teams, E29 HouseExpo).
Validity: miss_cell = occupied cells among the cells the robot entered on certified blocks (Prop. A, the collision rate),
summed over robots, also by staleness bucket; strict = every admitted cell (reached or not); miss_all = all traversals
including warm-up and probe blocks (unc_blocks per robot). Progress: c_time = certified static fraction averaged over blocks 10..H on a common horizon
of H = 80 blocks (a run ending earlier keeps its final state; idling spends time, not budget). idle = blocks with no
admissible segment (mean over robots)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from runner import load
from analysis import ci95
import numpy as np, pandas as pd
H = 80
BK = ["1", "2-3", "4-7", "8-15", "16+", "never"]   # execute() accounting buckets (the certificate uses 1, 2-3, 4-7, 8-15, 16-31, >=32/never)

def per_run(df):
    lb = df.groupby("rid").block.transform("max")
    agg = {"collisions": ("collisions", "sum"), "steps": ("steps", "sum"), "path_len": ("path_len", "sum"), "idle": ("idle", "mean"),
           "c_whole": ("c_whole", "mean"), "unc_cells": ("unc_cells", "sum"), "unc_occ": ("unc_occ", "sum"), "unc_blocks": ("unc_blocks", "mean"),
           **{f"pc{b}": (f"pc{b}", "sum") for b in range(6)}, **{f"po{b}": (f"po{b}", "sum") for b in range(6)},
           **{f"tc{b}": (f"tc{b}", "sum") for b in range(6)}, **{f"to{b}": (f"to{b}", "sum") for b in range(6)},
           **{k: (k, "first") for k in df.columns if k.startswith("cfg_")}}
    last = df[df.block == lb].groupby("rid").agg(**agg).reset_index()
    last["pc"] = sum(last[f"pc{b}"] for b in range(6)); last["po"] = sum(last[f"po{b}"] for b in range(6))
    last["tc"] = sum(last[f"tc{b}"] for b in range(6)); last["to"] = sum(last[f"to{b}"] for b in range(6))
    last["miss_cell"] = last.to / last.tc.clip(lower=1)                     # certified traversals (all traversals for other guards)
    last["strict"] = last.po / last.pc.clip(lower=1)                        # every admitted cell, reached or not
    last["miss_all"] = (last.to + last.unc_occ) / (last.tc + last.unc_cells).clip(lower=1)   # incl. warm-up and probes
    last["stop_pose"] = last.collisions / last.steps.clip(lower=1)
    def series(d):
        c = d.groupby("block").c_whole.mean().reindex(range(H + 1)).ffill().values
        return pd.Series({"c_time": c[10:].mean(), "c_H": c[H]})
    return last.join(df.groupby("rid").apply(series), on="rid")

def table(last, keys):
    rows = []
    for key, d in last.groupby(keys):
        row = dict(zip(keys, key)); row["n"] = len(d)
        for c in ["miss_cell", "strict", "miss_all", "unc_blocks", "idle", "path_len", "c_time", "c_H"]:
            m, h = ci95(d[c].values); row[c] = m; row[c + "_ci"] = h
        row["miss_max"] = d.miss_cell.max()
        # pooled-by-staleness miss (cells summed over runs) and share of admitted cells per bucket
        pc = np.array([d[f"tc{b}"].sum() for b in range(6)]); po = np.array([d[f"to{b}"].sum() for b in range(6)])
        row["stale_miss"] = " ".join(f"{po[b]/pc[b]:.3f}" if pc[b] > 0 else "-" for b in range(6))
        row["stale_share"] = " ".join(f"{x:.2f}" for x in pc / max(pc.sum(), 1))
        rows.append(row)
    return pd.DataFrame(rows)

def paired(last, keys, col, ref, of):
    """Paired differences vs `ref` over seeds; Holm-corrected paired t-test within each setting (family = all
    comparisons vs ref at that setting); `sig` marks p_holm < 0.05."""
    from scipy.stats import ttest_rel
    out = []
    for key, d in last.groupby(keys):
        piv = d.pivot_table(index="cfg_seed", columns=of, values=col)
        if ref not in piv: continue
        rows = []
        for g in piv.columns:
            if g == ref: continue
            dd = (piv[g] - piv[ref]).dropna().values
            m, h = ci95(dd); pv = ttest_rel(piv[g].dropna(), piv[ref].loc[piv[g].dropna().index]).pvalue if len(dd) > 1 else 1.0
            rows.append([g, m, h, pv])
        ps = np.array([r[3] for r in rows]); order = np.argsort(ps); holm = np.empty(len(ps))
        for rank, i in enumerate(order):
            holm[i] = min(1.0, ps[i] * (len(ps) - rank))
        holm = np.maximum.accumulate(holm[order])[np.argsort(order)]
        for (g, m, h, pv), ph in zip(rows, holm):
            out.append(dict(zip(keys, key), **{"vs": f"{g} - {ref}", col: round(m, 3), "ci": round(h, 3), "p_holm": round(ph, 3), "sig": "*" if ph < 0.05 else ""}))
    return pd.DataFrame(out)

COLS = ["n", "miss_cell", "miss_cell_ci", "miss_max", "strict", "miss_all", "unc_blocks", "idle", "path_len", "c_time", "c_time_ci", "c_H"]

if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "e26"
    df, runs = load(exp)
    if exp == "e29":
        H = 40   # 150 m runs end near block 40
    if exp == "e31":
        # ablation labels from the fields that differ from the defaults
        from e31_ablation import VARIANTS
        def label(r):
            if r.cfg_guard != "certified": return r.cfg_guard
            for name, kv in VARIANTS.items():
                if all(abs(r[f"cfg_{k}"] - v) < 1e-9 for k, v in kv.items()): return name
            return "default"
        df["cfg_guard"] = df.apply(label, axis=1)
    df["cfg_hazard_motion"] = df.get("cfg_hazard_motion", "walk"); df["cfg_hazard_motion"] = df.cfg_hazard_motion.fillna("walk")
    if "cfg_share_pairs" in df:
        df["cfg_guard"] = np.where(df.cfg_share_pairs.fillna(True).astype(bool), df.cfg_guard, df.cfg_guard + "_own")
    pd.set_option("display.width", 250)
    last = per_run(df)
    std = (last.cfg_guard_alpha < 0) & (last.cfg_prob_decay == 0.9) & (last.cfg_alpha == 0.05) & (last.cfg_gate_window == 0)
    a = last[(last.cfg_planner == "entropy") & std]
    for motion, dm in a.groupby("cfg_hazard_motion"):
        keys = ["cfg_n_hazards", "cfg_hazard_step", "cfg_guard"]
        print(f"=== (a) guards, entropy planner, motion={motion} ({len(dm)} runs) ===")
        t = table(dm, keys)
        print(t[keys + COLS].round(3).to_string(index=False))
        print(t[keys + ["stale_share", "stale_miss"]].to_string(index=False))
        print(paired(dm, keys[:2], "c_time", "certified", "cfg_guard").to_string(index=False))
    lv = last[(last.cfg_guard == "certified") & (last.cfg_planner == "entropy") & (last.cfg_gate_window == 0)].copy()
    if (lv.cfg_guard_alpha > 0).any():
        lv["level"] = np.where(lv.cfg_guard_alpha > 0, lv.cfg_guard_alpha, lv.cfg_alpha)
        keys = ["cfg_hazard_motion", "cfg_n_hazards", "cfg_hazard_step", "level"]
        print(f"=== (e) certified guard by level ({len(lv)} runs); fresh from (a) for comparison ===")
        t = table(lv, keys)
        print(t[keys + COLS].round(3).to_string(index=False))
    b = last[(last.cfg_guard == "certified") & std & (last.cfg_hazard_motion == "walk")]
    if b.cfg_planner.nunique() > 1:
        keys = ["cfg_n_hazards", "cfg_hazard_step", "cfg_planner"]
        print(f"=== (b) acquisitions under the certified guard ({len(b)} runs) ===")
        print(table(b, keys)[keys + COLS].round(3).to_string(index=False))
        print(paired(b, keys[:2], "c_time", "certified", "cfg_planner").to_string(index=False))
    c = last[(last.cfg_planner == "entropy") & (last.cfg_hazard_motion == "walk") & ~std & (last.cfg_gate_window == 0) & last.cfg_guard.isin(["certified", "prob"])]
    if len(c):
        c = c.copy()
        c["setting"] = np.where(c.cfg_guard == "certified", "cert@" + c.cfg_guard_alpha.astype(str),
                                "prob@" + c.cfg_alpha.astype(str) + "/d" + c.cfg_prob_decay.astype(str))
        keys = ["cfg_hazard_step", "setting"]
        print(f"=== (c) safety-progress frontier ({len(c)} runs; add cert@0.05 and prob@0.05/d0.9 from (a)) ===")
        print(table(c, keys)[keys + COLS].round(3).to_string(index=False))
