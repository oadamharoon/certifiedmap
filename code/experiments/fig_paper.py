"""Paper figures (PDF) from cached results -> paper/figs/."""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from runner import load, RESULTS
from analysis import curves, ci95, GRID

# "acm" as the first argument retargets the ACM draft and its narrower column (3.34 in against IEEEtran's 3.5)
ACM = "acm" in sys.argv
FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps" if ACM else "paper", "figs")
os.makedirs(FIGS, exist_ok=True)
COL, FULL = (3.34, 7.0) if ACM else (3.5, 7.0)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "axes.titlesize": 8})
PL = ["certified", "entropy", "lse", "frontier", "lawnmower", "random"]
LAB = {"certified": "certified", "entropy": "entropy", "lse": "level-set", "frontier": "frontier", "lawnmower": "lawnmower", "random": "random"}

what = sys.argv[1:] or ["e1", "e2", "e3", "e4", "maps"]

if "e1" in what:
    df, _ = load("e1")
    df["cond"] = list(zip(df.cfg_sigma_r, df.cfg_wall_thick))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.55), sharey=True)
    for ax, cond, title in zip(axes, [(0.02, 0.25), (0.10, 0.75)], [r"$\sigma_r$=0.02 m, walls 0.25 m", r"$\sigma_r$=0.10 m, walls 0.75 m"]):
        cv = curves(df[df.cond == cond], "c_whole", keys=("cfg_planner",))
        for p in PL:
            m, h, n = cv[p]; ax.plot(GRID, m, label=LAB[p], lw=1); ax.fill_between(GRID, m - h, m + h, alpha=0.15)
        ax.set_title(title); ax.set_xlabel("path length (m)"); ax.grid(alpha=0.3); ax.set_xlim(0, 150)
    axes[0].set_ylabel("certified fraction $c$")
    h, l = axes[0].get_legend_handles_labels(); leg = fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    for ln in leg.get_lines(): ln.set_linewidth(2.0)
    plt.tight_layout(pad=0.3, rect=(0, 0.16, 1, 1)); plt.savefig(os.path.join(FIGS, "e1_curves.pdf")); print("e1_curves.pdf")

if "e2" in what:
    t = pd.read_csv(os.path.join(RESULTS, "e2_analysis", "validity.csv"))
    t = t[t.planner.isin(["certified", "entropy"])]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.85))
    for wt, mk, lab in [(True, "o", "query law (weighted)"), (False, "^", "sampled law (unweighted)")]:
        d = t[t.weighted == wt].groupby("nominal").agg(cov=("cov_final", "mean"), lo=("cov_block_min", "mean"),
                                                       frac=("frac_runs_below", "mean")).reset_index()
        axes[0].plot(d["nominal"], d["cov"], marker=mk, ms=3.5, lw=1, label=lab)
        axes[1].plot(d["nominal"], 100 * d["frac"], marker=mk, ms=3.5, lw=1, label=lab)
    axes[0].plot([0.78, 1.0], [0.78, 1.0], "k--", lw=0.7, label="nominal")
    axes[0].set_xlabel(r"nominal $1-\alpha$"); axes[0].set_ylabel("empirical coverage"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel(r"nominal $1-\alpha$"); axes[1].set_ylabel("runs below\nnominal (%)"); axes[1].grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.0), fontsize=6.5, columnspacing=1.0, handletextpad=0.4)
    plt.tight_layout(pad=0.3, w_pad=1.4, rect=(0, 0.12, 1, 1)); plt.savefig(os.path.join(FIGS, "validity.pdf")); print("validity.pdf")

if "e3" in what:
    df, _ = load("e3")
    df["team"] = df.cfg_N.astype(str) + "-" + df.cfg_topology; df["team_path"] = df.path_len * df.cfg_N
    fig, ax = plt.subplots(figsize=(COL * 0.86, 2.1)); grid = np.arange(0, 100.1, 2.5)
    for team, lab in [("1-mesh", "N=1"), ("2-mesh", "N=2"), ("3-mesh", "N=3"), ("5-mesh", "N=5 mesh"), ("5-ring", "N=5 ring")]:
        d = df[(df.team == team) & (df.cfg_planner == "certified")]; arr = []
        for rid, dr in d.groupby("rid"):
            g = dr.groupby("block").agg(x=("team_path", "mean"), y=("c_whole", "mean")).sort_values("x")
            arr.append(np.interp(grid, g.x.values, g.y.values))
        m, h = ci95(np.array(arr), 0); ax.plot(grid, m, label=lab, lw=1); ax.fill_between(grid, m - h, m + h, alpha=0.15)
    ax.set_xlabel("total team path length (m)"); ax.set_ylabel("certified fraction $c$"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(pad=0.3); plt.savefig(os.path.join(FIGS, "team.pdf")); print("team.pdf")

if "e4" in what:
    cal = pd.read_csv(os.path.join(RESULTS, "e4_analysis", "calibration.csv")); s = pd.read_csv(os.path.join(RESULTS, "e4_analysis", "stopping.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.1))
    axes[0].errorbar(cal.pred, cal.real, yerr=cal.sd / np.sqrt(cal.n) * 1.96, fmt="o-", ms=3, lw=1, color="C3", capsize=2, label="binned mean")
    lim = max(cal.pred.max(), cal.real.max()); axes[0].plot([0, lim], [0, lim], "k--", lw=0.7, label="ideal")
    axes[0].set_xlabel("predicted gain rate (cells/m)"); axes[0].set_ylabel("realised gain rate (cells/m)"); axes[0].grid(alpha=0.3)
    g = s.groupby("tau").agg(saved=("saved", "mean"), loss=("loss", "mean"), s_ci=("saved", lambda v: ci95(v)[1]), l_ci=("loss", lambda v: ci95(v)[1])).reset_index()
    axes[1].errorbar(g.saved, g.loss, xerr=g.s_ci, yerr=g.l_ci, fmt="o-", ms=3, lw=1, capsize=2)
    for _, r in g.iterrows(): axes[1].annotate(rf"$\tau$={int(r.tau)}", (r.saved, r.loss), fontsize=6, xytext=(3, 2), textcoords="offset points")
    axes[1].set_xlabel("budget saved"); axes[1].set_ylabel("certified fraction forgone"); axes[1].grid(alpha=0.3)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1)); plt.savefig(os.path.join(FIGS, "stopping.pdf")); print("stopping.pdf")

if "maps" in what:
    shutil.copy(os.path.join(RESULTS, "figs", "maps.pdf"), os.path.join(FIGS, "maps.pdf")); print("maps.pdf")

if "e6" in what:
    t = pd.read_csv(os.path.join(RESULTS, "e6_endpoint.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.4), sharex=True)
    beta = 0.7 * 0.05
    for p, mk in [("certified", "o"), ("entropy", "^"), ("lse", "s")]:
        d = t[(t.score == "residual") & (t.planner == p)].sort_values("p_flip")
        axes[0].errorbar(d.p_flip, d.c_whole, yerr=d["c_whole±"], marker=mk, ms=3, lw=1, capsize=2, label=LAB[p])
    d = t[(t.score == "margin") & (t.planner == "certified")].sort_values("p_flip")
    axes[0].errorbar(d.p_flip, d.c_whole, yerr=d["c_whole±"], marker="o", mfc="none", ms=3, lw=1, ls="--", capsize=2, color="C0", label="certified, margin score")
    d = t[(t.score == "residual") & (t.planner == "certified")].sort_values("p_flip")
    axes[1].plot(d.p_flip, d.band_mid, "o-", ms=3, lw=1, color="C3", label=r"cells with $1/q<\sigma<2/q$")
    axes[1].plot(d.p_flip, d.ambiguous, "s-", ms=3, lw=1, color="C4", label="undecided with data")
    for ax in axes:
        ax.axvline(beta, color="k", ls=":", lw=0.8); ax.set_xscale("log"); ax.grid(alpha=0.3); ax.set_xlabel("label-flip probability $p$")
    axes[0].text(beta * 1.08, 0.35, r"$\beta$", fontsize=7); axes[0].set_ylabel("certified fraction $c$"); axes[1].set_ylabel("fraction of calibrated cells")
    h0, l0 = axes[0].get_legend_handles_labels(); h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(pad=0.3, rect=(0, 0.24, 1, 1)); plt.savefig(os.path.join(FIGS, "labelnoise.pdf")); print("labelnoise.pdf")

if "e9" in what:
    df, _ = load("e9"); df = df[df.cfg_planner == "certified"]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.1))
    for N in [1, 2, 3, 5, 8]:
        d = df[df.cfg_N == N]; arr = []
        for rid, dr in d.groupby("rid"):
            g = dr.groupby("block").c_whole.mean(); arr.append(np.interp(np.arange(0, 9), g.index.values, g.values, right=g.values[-1]))
        m, h = ci95(np.array(arr), 0); axes[0].plot(np.arange(0, 9), m, label=f"N={N}", lw=1); axes[0].fill_between(np.arange(0, 9), m - h, m + h, alpha=0.15)
    axes[0].set_xlabel("elapsed blocks (5 m each)"); axes[0].set_ylabel("certified fraction $c$"); axes[0].grid(alpha=0.3)
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(N=("cfg_N", "first"), sep=("team_sep", "mean"), ov=("team_overlap", "mean"))
    g = last.groupby("N").agg(sep=("sep", "mean"), sep_ci=("sep", lambda v: ci95(v)[1]), ov=("ov", "mean"), ov_ci=("ov", lambda v: ci95(v)[1])).reset_index()
    g = g[g.N > 1]
    axes[1].errorbar(g.N, g.ov, yerr=g.ov_ci, marker="o", ms=3, lw=1, capsize=2, color="C3"); axes[1].set_xlabel("team size $N$"); axes[1].set_ylabel("doubly-gated fraction", color="C3"); axes[1].grid(alpha=0.3)
    ax2 = axes[1].twinx(); ax2.errorbar(g.N, g.sep, yerr=g.sep_ci, marker="s", ms=3, lw=1, capsize=2, color="C0"); ax2.set_ylabel("mean separation (m)", color="C0")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1)); plt.savefig(os.path.join(FIGS, "team.pdf")); print("team.pdf")

if "e12" in what:
    df, _ = load("e12"); df["flips"] = df.cfg_p_flip_agents.astype(str)
    last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(flips=("flips", "first"), planner=("cfg_planner", "first"), c=("c_whole", "mean"), amb=("ambiguous", "mean"), nod=("nodata", "mean"))
    order = [("[0.0, 0.0, 0.05]", "0 / 0 / .05"), ("[0.0, 0.0, 0.1]", "0 / 0 / .10"), ("[0.0, 0.05, 0.1]", "0 / .05 / .10")]
    pls = [("certified", "certified (fused $\\mathcal{U}$)"), ("entropy_u", "entropy on fused $\\mathcal{U}$"), ("entropy_g", "entropy on coverage mask"), ("entropy", "entropy")]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.4), sharex=True)
    x = np.arange(len(order)); wd = 0.2
    for i, (p, lab) in enumerate(pls):
        m = [ci95(last[(last.flips == f) & (last.planner == p)].c.values) for f, _ in order]
        axes[0].bar(x + (i - 1.5) * wd, [a for a, _ in m], wd, yerr=[b for _, b in m], capsize=1.5, label=lab)
        m = [ci95(last[(last.flips == f) & (last.planner == p)].amb.values) for f, _ in order]
        axes[1].bar(x + (i - 1.5) * wd, [a for a, _ in m], wd, yerr=[b for _, b in m], capsize=1.5)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in order], fontsize=6.5, rotation=20); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("certified fraction $c$"); axes[1].set_ylabel("explored but undecided"); axes[0].set_ylim(0.35, 0.78)
    fig.text(0.5, 0.2, "per-robot label-flip probabilities", ha="center", fontsize=8)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(pad=0.3, rect=(0, 0.25, 1, 1)); plt.savefig(os.path.join(FIGS, "hetero_team.pdf")); print("hetero_team.pdf")

if "frontier" in what:
    # certification-resource frontier A_alpha(E) from E2 (weighted calibration)
    df, _ = load("e2"); df = df[df.cfg_weighted == True]
    grid = np.arange(0, 100.1, 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.4), sharey=True)
    for ax, p in zip(axes, ["certified", "lawnmower"]):
        for a_, col in zip([0.02, 0.05, 0.1, 0.2], ["C0", "C1", "C2", "C3"]):
            d = df[(df.cfg_planner == p) & (df.cfg_alpha == a_)]; arr = []
            for rid, dr in d.groupby("rid"):
                g = dr.groupby("block").agg(x=("path_len", "mean"), y=("c_whole", "mean")).sort_values("x"); arr.append(np.interp(grid, g.x.values, g.y.values, right=g.y.values[-1]))
            m, h = ci95(np.array(arr), 0); ax.plot(grid, m, color=col, lw=1, label=rf"$\alpha$={a_}"); ax.fill_between(grid, m - h, m + h, color=col, alpha=0.15)
        ax.set_title(LAB[p]); ax.set_xlabel("path length (m)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("certified fraction $c$")
    h, l = axes[0].get_legend_handles_labels(); leg = fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    for ln in leg.get_lines(): ln.set_linewidth(2.0)
    plt.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1)); plt.savefig(os.path.join(FIGS, "frontier.pdf")); print("frontier.pdf")

if "dichotomy" in what:
    # Left: direct test of Cor. dichotomy -- own-certificate commit share among well-observed cells (E17),
    # for a single robot (beta = s*alpha) and the unreliable robot of a three-robot team (beta_i = s*alpha/3).
    # Right: the forecast theta-hat from the calibration fold (E6).
    TOL = 1.2                                       # well observed := sigma(x) <= TOL * sigma_min
    d17 = pd.read_csv(os.path.join(RESULTS, "e17_analysis", "runs.csv")); d17 = d17[d17.tol == TOL]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.70))
    beta = 0.7 * 0.05
    pp = np.linspace(0.015, 0.11, 200)
    for kind, bi, col, mk, lab in [("single", beta, "C0", "o", "one robot"), ("team", beta / 3, "C1", "s", "team member")]:
        g = d17[d17.kind == kind].groupby("p").share.agg(["mean", lambda v: ci95(v)[1]]).reset_index()
        axes[0].errorbar(g.p, g["mean"], yerr=g["<lambda_0>"], marker=mk, ms=3, lw=0, elinewidth=0.8, capsize=2.5, color=col, label=lab)
        axes[0].plot(pp, np.minimum(1.0, bi / pp), ls="--", lw=0.9, color=col, zorder=0)
    axes[0].plot([], [], "k--", lw=0.9, label=r"$\min(1,\beta_i/\theta)$")
    axes[0].set_xlabel(r"label-error rate $\theta$"); axes[0].set_ylabel("committed share")
    axes[0].set_xlim(0.01, 0.112); axes[0].set_ylim(0, 1.05); axes[0].set_xticks([0.02, 0.05, 0.08, 0.11]); axes[0].grid(alpha=0.3)
    df, _ = load("e6"); df = df[df.cfg_score == "residual"]
    last = df.sort_values("block").groupby("rid").tail(1)
    d = last[last.cfg_planner == "certified"]; g = d.groupby("cfg_p_flip").p_hat.agg(["mean", lambda v: ci95(v)[1]]).reset_index()
    lim = [0, 0.112]; axes[1].plot(lim, lim, "k--", lw=0.7, label=r"$\hat\theta=\theta$"); axes[1].set_xlim(-0.004, 0.115)
    axes[1].errorbar(g.cfg_p_flip, g["mean"], yerr=g["<lambda_0>"], marker="o", ms=2.5, lw=0, elinewidth=0.7, capsize=2.5, capthick=0.7, color="0.2", label=r"mean $\pm$ 95% CI")
    axes[1].axhline(beta, color="C3", ls=":", lw=0.8, label=r"budget $\beta$")
    axes[1].set_xlabel(r"flip probability $\theta$"); axes[1].set_ylabel(r"forecast $\hat\theta$"); axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 0.195); axes[1].set_yticks([0.0, 0.05, 0.10])
    axes[1].legend(loc="upper left", fontsize=6, frameon=False, handlelength=1.3, borderpad=0.1, labelspacing=0.2)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=6.5, columnspacing=0.8, handletextpad=0.4)
    plt.tight_layout(pad=0.5, w_pad=2.2, rect=(0, 0.13, 1, 0.99)); plt.savefig(os.path.join(FIGS, "dichotomy.pdf")); print("dichotomy.pdf")
    for kind in ["single", "team"]:
        g = d17[d17.kind == kind].groupby("p").agg(share=("share", "mean"), pred=("pred", "first")).reset_index()
        print(kind, [(float(r.p), round(r.share, 3), round(r.pred, 3), round(abs(r.share - r.pred), 3)) for r in g.itertuples()])

if "coord" in what:
    # left: fixed total budget, c vs N for four planners (E3); right: heterogeneity bars (E3 N=3, E12, E13)
    df3, _ = load("e3"); l3 = df3.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(N=("cfg_N", "first"), t=("cfg_topology", "first"), p=("cfg_planner", "first"), c=("c_whole", "mean"))
    df12, _ = load("e12"); l12 = df12.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(f=("cfg_p_flip_agents", lambda v: str(v.iloc[0])), p=("cfg_planner", "first"), c=("c_whole", "mean"))
    df13, _ = load("e13"); df13["het"] = np.where(df13.cfg_beams_agents.astype(str).str.contains("8"), "beams", "range")
    l13 = df13.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(h=("het", "first"), p=("cfg_planner", "first"), c=("c_whole", "mean"))
    pls = [("certified", "certified (fused $\\mathcal{U}$)", "C0"), ("entropy_u", "entropy ($\\mathcal{U}$)", "C1"), ("entropy_g", "entropy (mask)", "C2"), ("entropy", "entropy (plans)", "C3"),
           ("frontier_g", "frontier (mask)", "C4"), ("frontier_u", "frontier ($\\mathcal{U}$)", "C5")]
    d19, _ = load("e19"); l19 = d19.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(N=("cfg_N", "first"), t=("cfg_topology", "first"), p=("cfg_planner", "first"), c=("c_whole", "mean"), f=("cfg_p_flip_agents", lambda v: str(list(v.iloc[0]))))
    l19["het"] = np.where(d19.groupby("rid").cfg_beams_agents.first().reindex(l19.index).astype(str).str.contains("8"), "beams", np.where(d19.groupby("rid").cfg_range_agents.first().reindex(l19.index).astype(str).str.contains("2.5"), "range", "none")) if "cfg_beams_agents" in d19 else "none"
    l3 = pd.concat([l3, l19[(l19.f == "[]") & (l19.het == "none")][["N", "t", "p", "c"]]]); l12 = pd.concat([l12, l19[l19.f != "[]"][["f", "p", "c"]]])
    l13 = pd.concat([l13, l19[l19.het != "none"].rename(columns={"het": "h"})[["h", "p", "c"]]])
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.35), gridspec_kw=dict(width_ratios=[1, 0.8, 1.5]))
    for p, lab, col in pls:
        d = l3[(l3.p == p) & (l3.t == "mesh")]; g = d.groupby("N").c.agg(["mean", lambda v: ci95(v)[1]]).reset_index()
        axes[0].errorbar(g.N, g["mean"], yerr=g["<lambda_0>"], marker="o", ms=3, lw=1, capsize=2, color=col, label=lab)
    d = l3[(l3.t == "ring")]
    for i, (p, lab, col) in enumerate(pls):
        g = d[d.p == p].c; axes[0].errorbar([5 + 0.06 * (i - 2.5)], [g.mean()], yerr=[ci95(g)[1]], marker="s", ms=3, lw=1, capsize=2, color=col, mfc="none")
    axes[0].annotate("ring, $N=5$", xy=(5, 0.64), xytext=(3.4, 0.66), fontsize=7, arrowprops=dict(arrowstyle="-", lw=0.6))
    axes[0].set_xticks([1, 2, 3, 5]); axes[0].set_xlabel("team size $N$, total budget 100 m"); axes[0].set_ylabel("certified fraction $c$"); axes[0].grid(alpha=0.3)
    # middle: fixed per-robot budget (E9, 40 m each, 10 worlds)
    df9, _ = load("e9"); l9 = df9.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(N=("cfg_N", "first"), p=("cfg_planner", "first"), c=("c_whole", "mean"))
    for p, lab, col in pls:
        d = l9[l9.p == p]
        if len(d) == 0 or d.groupby("N").size().min() < 10: continue     # complete series only (10 worlds per N)
        g = d.groupby("N").c.agg(["mean", lambda v: ci95(v)[1]]).reset_index()
        axes[1].errorbar(g.N, g["mean"], yerr=g["<lambda_0>"], marker="o", ms=3, lw=1, capsize=2, color=col)
    axes[1].set_xticks(sorted(l9.N.unique())); axes[1].set_xlabel("team size $N$, 40 m per robot"); axes[1].grid(alpha=0.3); axes[1].set_ylabel("certified fraction $c$")
    groups = [("homog.", lambda p: l3[(l3.N == 3) & (l3.t == "mesh") & (l3.p == p)].c), ("range\n5/5/2.5 m", lambda p: l13[(l13.h == "range") & (l13.p == p)].c), ("beams\n40/40/8", lambda p: l13[(l13.h == "beams") & (l13.p == p)].c),
              ("labels\n0/0/.05", lambda p: l12[(l12.f == "[0.0, 0.0, 0.05]") & (l12.p == p)].c), ("labels\n0/0/.10", lambda p: l12[(l12.f == "[0.0, 0.0, 0.1]") & (l12.p == p)].c), ("labels\n0/.05/.10", lambda p: l12[(l12.f == "[0.0, 0.05, 0.1]") & (l12.p == p)].c)]
    x = np.arange(len(groups)); wd = 0.14
    for i, (p, lab, col) in enumerate(pls):
        ms_ = [ci95(getf(p).values) if len(getf(p)) else (np.nan, 0) for _, getf in groups]
        axes[2].bar(x + (i - 2.5) * wd, [m for m, _ in ms_], wd, yerr=[h for _, h in ms_], capsize=1.2, color=col)
    axes[2].set_xticks(x); axes[2].set_xticklabels([g for g, _ in groups], fontsize=6); axes[2].set_ylim(0.2, 0.9); axes[2].set_ylabel("certified fraction $c$"); axes[2].grid(alpha=0.3, axis="y")
    axes[2].set_xlabel("three robots, 100 m total; odd robot differs")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7, columnspacing=0.9, handletextpad=0.4)
    plt.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1)); plt.savefig(os.path.join(FIGS, "coordination.pdf")); print("coordination.pdf")
