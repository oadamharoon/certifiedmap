"""Shared analysis utilities: curves vs distance, endpoint stats, trajectory divergence."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

GRID = np.arange(0, 150.1, 2.5)


def interp_curve(df_run: pd.DataFrame, col: str, grid=GRID, xcol="path_len"):
    """Interpolate a per-block metric onto a common distance grid (agent-averaged)."""
    g = df_run.groupby("block").agg(x=(xcol, "mean"), y=(col, "mean")).sort_values("x")
    x, y = g.x.values, g.y.values
    out = np.interp(grid, x, y, left=y[0], right=y[-1])
    out[grid > x[-1] + 2.5] = np.nan     # do not extrapolate beyond the run's budget
    return out


def ci95(a, axis=0):
    a = np.asarray(a, float)
    n = np.sum(~np.isnan(a), axis=axis)
    m = np.nanmean(a, axis=axis)
    s = np.nanstd(a, axis=axis, ddof=1)
    h = tdist.ppf(0.975, np.maximum(n - 1, 1)) * s / np.sqrt(np.maximum(n, 1))
    return m, h


def curves(df: pd.DataFrame, col: str, keys=("cfg_planner",), grid=GRID):
    """dict key -> (mean, ci, n) curves over runs."""
    res = {}
    for key, d in df.groupby(list(keys)):
        arr = np.stack([interp_curve(dr, col, grid) for _, dr in d.groupby("rid")], 0)
        m, h = ci95(arr, 0)
        res[key if len(keys) > 1 else key[0]] = (m, h, len(arr))
    return res


def final_table(df: pd.DataFrame, cols, keys=("cfg_planner",)):
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1)
    last = last.groupby(["rid"] + list(keys))[cols].mean().reset_index()
    rows = []
    for key, d in last.groupby(list(keys)):
        row = {k: v for k, v in zip(keys, key)}
        row["n"] = len(d)
        for c in cols:
            m, h = ci95(d[c].values)
            row[c] = m; row[c + "_ci"] = h
        rows.append(row)
    return pd.DataFrame(rows)


def distance_to(df: pd.DataFrame, col: str, level: float, keys=("cfg_planner",)):
    """Distance at which the (agent-averaged) metric first reaches ``level`` (nan if never)."""
    rows = []
    for (rid,), d in df.groupby(["rid"]):
        g = d.groupby("block").agg(x=("path_len", "mean"), y=(col, "mean")).sort_values("x")
        hit = g[g.y >= level]
        rows.append({"rid": rid, **{k: d[k].iloc[0] for k in keys}, "dist": hit.x.iloc[0] if len(hit) else np.nan,
                     "reached": len(hit) > 0})
    return pd.DataFrame(rows)


def visitation_hist(traj: np.ndarray, W=30.0, H=20.0, cell=1.0):
    h, _, _ = np.histogram2d(traj[:, 0], traj[:, 1], bins=[int(W / cell), int(H / cell)], range=[[0, W], [0, H]])
    return h.ravel() / max(h.sum(), 1)


def jsd(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
