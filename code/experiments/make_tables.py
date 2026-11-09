"""LaTeX table bodies from analysis CSVs -> paper/tables/*.tex"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd
from runner import RESULTS
T = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "tables"); os.makedirs(T, exist_ok=True)
LAB = {"certified": "certified", "entropy": "entropy", "lse": "level-set", "frontier": "frontier", "lawnmower": "lawnmower", "random": "random", "entropy_u": "entropy-U", "entropy_g": "entropy-G"}
def f3(x): return f"{x:.3f}"
HEAD = {
 "e1.tex": (r"\begin{tabular}{llccccc}", r"cond. & planner & $c$ & cov & acc & und. & n.d.\\"),
 "e2.tex": (r"\begin{tabular}{lccccccc}", r"weights & $1{-}\alpha$ & cov & min & below & $c$ & $\emptyset$ & und.\\"),
  "e3.tex": (r"\begin{tabular}{lccccccc}", r"team & cert. & entr. & entr.-U & entr.-G & cov & $-$entr. & $-$entr.-U\\"),
 "e7.tex": (r"\begin{tabular}{llccccc}", r"factor & level & cert. & entr. & cert.$-$entr. & cov & frontier\\"),
}
def w(name, s):
    b, h = HEAD[name]
    full = b + "\n\\toprule\n" + h + "\n\\midrule\n" + s + "\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(T, name), "w").write(full); print("wrote", name)

# ---- E1 -----------------------------------------------------------------------------
e = pd.read_csv(os.path.join(RESULTS, "e1_analysis", "endpoint.csv"))
rows = []
for cond, lab in [("sr=0.02,wt=0.25", r"\multirow{6}{*}{\shortstack[l]{$\sigma_r{=}0.02$\\thin}}"), ("sr=0.1,wt=0.75", r"\multirow{6}{*}{\shortstack[l]{$\sigma_r{=}0.10$\\thick}}")]:
    d = e[e.grp == cond].set_index("planner")
    for i, p in enumerate(["certified", "entropy", "lse", "frontier", "lawnmower", "random"]):
        r = d.loc[p]
        rows.append(f"{lab if i == 0 else ''} & {LAB[p]} & {f3(r.c_whole)} $\\pm$ {r['c_whole±']:.3f} & {f3(r.cov_explored)} & {f3(r.acc_committed)} & {f3(r.ambiguous)} & {f3(r.nodata)}\\\\")
    rows.append(r"\midrule")
w("e1.tex", "\n".join(rows[:-1]))

# ---- E2 -----------------------------------------------------------------------------
v = pd.read_csv(os.path.join(RESULTS, "e2_analysis", "validity.csv")); v = v[v.planner == "certified"]
rows = []
for wt, lab in [(False, r"\multirow{4}{*}{no}"), (True, r"\multirow{4}{*}{yes}")]:
    d = v[v.weighted == wt].sort_values("alpha", ascending=True)
    for i, (_, r) in enumerate(d.iterrows()):
        rows.append(f"{lab if i == 0 else ''} & {1 - r.alpha:.2f} & {f3(r.cov_final)} & {f3(r.cov_block_min)} & {r.frac_runs_below:.2f} & {f3(r.c_expl)} & {f3(r['empty'])} & {f3(r.ambiguous)}\\\\")
    rows.append(r"\midrule")
w("e2.tex", "\n".join(rows[:-1]))

# ---- E3 -----------------------------------------------------------------------------
t = pd.read_csv(os.path.join(RESULTS, "e3_analysis", "team.csv"))
import pickle
pr = {}
try:
    from runner import load
    from analysis import ci95
    df, _ = load("e3"); last = df.sort_values("block").groupby(["rid", "agent"]).tail(1).groupby("rid").agg(team=("cfg_N", "first"), topo=("cfg_topology", "first"), planner=("cfg_planner", "first"), seed=("cfg_seed", "first"), c=("c_whole", "mean"))
    last["team"] = last.team.astype(str) + "-" + last.topo
    for team, d in last.groupby("team"):
        a = d[d.planner == "certified"].set_index("seed").c
        for o in ["entropy", "entropy_u"]:
            b = d[d.planner == o].set_index("seed").c; j = a.index.intersection(b.index)
            if len(j): m, h = ci95((a[j] - b[j]).values); pr[(team, o)] = f"${m:+.2f}{{\\scriptstyle\\pm}}{h:.2f}$"
except Exception as ex:
    print("e3 paired failed", ex)
rows = []
for team, lab in [("1-mesh", "$N{=}1$"), ("2-mesh", "$N{=}2$ mesh"), ("3-mesh", "$N{=}3$ mesh"), ("5-mesh", "$N{=}5$ mesh"), ("5-ring", "$N{=}5$ ring")]:
    d = t[t.team == team].set_index("planner")
    def g(p, c="c_whole"): return f3(d.loc[p, c]) if p in d.index else "--"
    rows.append(f"{lab} & {g('certified')} & {g('entropy')} & {g('entropy_u')} & {g('entropy_g')} & {g('certified','cov_min')} & {pr.get((team,'entropy'),'--')} & {pr.get((team,'entropy_u'),'--')}\\\\")
w("e3.tex", "\n".join(rows))

# ---- E7 -----------------------------------------------------------------------------
e7p = os.path.join(RESULTS, "e7_analysis", "ablation.csv")
a = pd.read_csv(e7p) if os.path.exists(e7p) else pd.DataFrame(columns=["factor","level","cert_c_whole","entr_c_whole","cert-entr","cert_cov_explored"])
names = {"reference": "operating point", "alpha": r"$\alpha$", "gate_r": "$r$ (m)", "safety": "$s$", "noise_var": r"$\sigma_n^2$", "seg_len": "segment (m)", "kde_bw": "density box (m)", "eps_ucb": r"UCB $\kappa$", "ell": r"$\ell$ (m)", "trav_thr": "trav. thr.", "trav_inflate": "trav. inflate", "front_dmin": "frontier $d_{\min}$ (m)"}
rows = []
for _, r in a.iterrows():
    lvl = "" if r.factor == "reference" else (f"{r.level}" if isinstance(r.level, str) else f"{float(r.level):g}")
    fr = f"{r.fron_c_whole:.3f}" if "fron_c_whole" in a.columns and not pd.isna(r.get("fron_c_whole", np.nan)) else "--"
    ce = f"{r.cert_c_whole:.3f}" if not pd.isna(r.cert_c_whole) else "--"; en = f"{r.entr_c_whole:.3f}" if not pd.isna(r.entr_c_whole) else "--"
    cov = f"{r.cert_cov_explored:.3f}" if not pd.isna(r.cert_cov_explored) else "--"
    diff = r["cert-entr"] if isinstance(r["cert-entr"], str) else None
    if diff is not None:
        diff = diff.replace("\u00b1", "{\\scriptstyle\\pm}")
        diff = diff.replace("+0.", "+.").replace("-0.", "-.").replace("}0.", "}.")
    dcol = f"${diff}$" if diff is not None else "--"
    rows.append(f"{names.get(r.factor, r.factor)} & {lvl} & {ce} & {en} & {dcol} & {cov} & {fr}" + "\\\\")
w("e7.tex", "\n".join(rows))

# ---- E15 real LiDAR -----------------------------------------------------------------
e15p = os.path.join(RESULTS, "e15_real_lidar.csv")
if os.path.exists(e15p):
    d = pd.read_csv(e15p); rows = []
    for log, lab in [("intel.clf", r"\multirow{3}{*}{\shortstack[l]{Intel\\559 m}}"), ("fr079.clf", r"\multirow{3}{*}{\shortstack[l]{FR079\\402 m}}")]:
        for i, a in enumerate([0.05, 0.1, 0.2]):
            def g(N, w, c): return d[(d.log == log) & (d.N == N) & (d.alpha == a) & (d.weighted == w)][c].iloc[0]
            rows.append(f"{lab if i == 0 else ''} & {1 - a:.2f} & {g(1, True, 'cov_explored'):.3f} & {g(3, True, 'cov_explored'):.3f} & {g(1, False, 'cov_explored'):.3f} & {g(3, False, 'cov_explored'):.3f} & {g(1, True, 'c_explored'):.2f} / {g(3, True, 'c_explored'):.2f} & {g(1, True, 'acc'):.2f} / {g(3, True, 'acc'):.2f}\\\\")
        rows.append(r"\midrule")
    HEAD["e15.tex"] = (r"\begin{tabular}{lccccccc}", r"log & $1{-}\alpha$ & \multicolumn{2}{c}{cov, weighted} & \multicolumn{2}{c}{cov, unweighted} & $c$ (w.) & acc (w.)\\ & & $N{=}1$ & $N{=}3$ & $N{=}1$ & $N{=}3$ & $1$ / $3$ & $1$ / $3$\\")
    w("e15.tex", "\n".join(rows[:-1]))
