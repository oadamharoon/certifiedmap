"""E4: predicted vs realised certified gain, and a gain-rate stopping rule (post hoc on E1 certified runs)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from runner import load, RESULTS
from analysis import ci95

df, runs = load("e1")
out_dir = os.path.join(RESULTS, "e4_analysis"); os.makedirs(out_dir, exist_ok=True)
d = df[(df.cfg_planner == "certified")].sort_values(["rid", "block"]).copy()
M_obs = {rid: (runs[rid]["occ"].size if "occ" in runs[rid] else None) for rid in d.rid.unique()}
# realised gain of block k's plan = n_committed(k+1) - n_committed(k)
d["real_gain"] = d.groupby("rid").n_committed.shift(-1) - d.n_committed
d["moved"] = d.groupby("rid").path_len.shift(-1) - d.path_len
# warm-up: the certificate exists only once q is finite (block >= 1); the predicted gain is over
# the full candidate path while only the first segment is executed, so compare *rates* (cells / m)
d = d[(d.moved > 0.25) & np.isfinite(d.q_i) & (d.block >= 1)]
d["rate_pred"] = d.gain_pred_obs / d.plan_len.clip(lower=0.5)
d["rate_real"] = d.real_gain / d.moved
pred, real = d.rate_pred.values, d.rate_real.values
slope = np.sum(pred * real) / np.sum(pred ** 2)
r = np.corrcoef(pred, real)[0, 1]
print(f"predicted vs realised gain RATE: n={len(d)} corr={r:.3f} slope(through origin)={slope:.3f} mean pred={pred.mean():.1f} mean real={real.mean():.1f} cells/m")
bins = np.quantile(pred, np.linspace(0, 1, 11))
d["bin"] = np.clip(np.searchsorted(bins, pred, side="right") - 1, 0, 9)
cal = d.groupby("bin").agg(pred=("rate_pred", "mean"), real=("rate_real", "mean"), n=("rate_real", "size"), sd=("rate_real", "std"))
print(cal.round(1).to_string())
# how does realised rate decay with distance? (what the rule is tracking)
d["dbin"] = (d.path_len // 25) * 25
print(d.groupby("dbin").agg(rate_pred=("rate_pred", "mean"), rate_real=("rate_real", "mean"), n=("rate_real", "size")).round(1).to_string())

# stopping rule: first block where predicted rate < tau (cells / m); report c at stop, final c, budget used
rows = []
PERSIST = 3      # stop when the predicted rate has been below tau for PERSIST consecutive blocks
GUARD = "relaxed" in d.columns   # 4.7: a block in which the robot was enclosed (traversability relaxed) cannot count
for tau in [5, 10, 15, 20, 30, 40]:
    for rid, dr in d.groupby("rid"):
        below = (dr.rate_pred < tau).values
        if GUARD:
            below = below & (dr.relaxed.values == 0)
        run = np.array([below[max(0, i - PERSIST + 1):i + 1].all() and i >= PERSIST - 1 for i in range(len(below))])
        hit = dr[run]
        final = df[(df.rid == rid)].sort_values("block").iloc[-1]
        if len(hit):
            k = hit.iloc[0]
            rows.append(dict(tau=tau, rid=rid, stopped=True, c_stop=k.c_whole, c_final=final.c_whole, dist_stop=k.path_len, dist_final=final.path_len,
                             cond=f"sr={k.cfg_sigma_r},wt={k.cfg_wall_thick}"))
        else:
            rows.append(dict(tau=tau, rid=rid, stopped=False, c_stop=final.c_whole, c_final=final.c_whole, dist_stop=final.path_len, dist_final=final.path_len,
                             cond=f"sr={final.cfg_sigma_r},wt={final.cfg_wall_thick}"))
s = pd.DataFrame(rows)
s["loss"] = s.c_final - s.c_stop; s["saved"] = 1 - s.dist_stop / s.dist_final
print(f"\nstopping rule (rate_pred < tau for {PERSIST} consecutive blocks; enclosure guard={GUARD}):")
g = s.groupby("tau").agg(stopped=("stopped", "mean"), c_stop=("c_stop", "mean"), c_final=("c_final", "mean"), loss=("loss", "mean"),
                         loss_max=("loss", "max"), dist_stop=("dist_stop", "mean"), saved=("saved", "mean"))
print(g.round(3).to_string())
s.to_csv(os.path.join(out_dir, "stopping.csv"), index=False); cal.to_csv(os.path.join(out_dir, "calibration.csv"))

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
axes[0].scatter(pred, real, s=4, alpha=0.3); axes[0].plot(cal.pred, cal.real, "o-", color="C3", label="binned mean")
lim = max(pred.max(), real.max()); axes[0].plot([0, lim], [0, lim], "k--", lw=0.8, label="ideal"); axes[0].set_xlabel("predicted gain rate (cells/m)"); axes[0].set_ylabel("realised gain rate (cells/m)"); axes[0].legend(fontsize=7)
for tau in [10, 20, 40]:
    ss = s[s.tau == tau]
    axes[1].scatter(ss.saved, ss.loss, s=8, label=f"tau={tau}")
axes[1].set_xlabel("fraction of budget saved"); axes[1].set_ylabel("certified fraction forgone"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(out_dir, "stopping.png"), dpi=130)
print("saved", out_dir)

# ---- feasibility rule (E4b): halt when the believed distance to the nearest undecided observable cell
# exceeds the remaining budget, or when no such cell is reachable ----------------------------------
if os.path.isdir(os.path.join(RESULTS, "e4b")):
    dfb, _ = load("e4b")
    rows = []
    for rid, dr in dfb.sort_values("block").groupby("rid"):
        final = dr.iloc[-1]; B = dr.cfg_budget.iloc[0]
        fire = dr[(dr.block >= 1) & ((dr.dist_undecided > (B - dr.path_len)) | ~np.isfinite(dr.dist_undecided))]
        if len(fire):
            k = fire.iloc[0]; rows.append(dict(rid=rid, fired=True, c_stop=k.c_whole, c_final=final.c_whole, saved=1 - k.path_len / final.path_len, dist_at_fire=k.dist_undecided))
        else:
            rows.append(dict(rid=rid, fired=False, c_stop=final.c_whole, c_final=final.c_whole, saved=0.0, dist_at_fire=np.nan))
    fb = pd.DataFrame(rows); fb["loss"] = fb.c_final - fb.c_stop
    print(f"\nfeasibility rule (E4b, n={len(fb)}): fired in {fb.fired.mean():.2f} of runs; mean saved {fb.saved.mean():.3f}; mean loss {fb.loss.mean():.4f}; max loss {fb.loss.max():.3f}")
    print("median believed distance to nearest undecided cell per block (m):", dfb[dfb.block >= 1].groupby("block").dist_undecided.median().round(1).tolist()[:12], "...")
    fb.to_csv(os.path.join(out_dir, "feasibility.csv"), index=False)
