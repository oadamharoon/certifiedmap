"""Mechanism figure: calibrated gated cells in the (sigma, |mu|) plane with the commitment regions."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from certmap.sim import Simulation, Config
FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps" if "acm" in sys.argv else "paper", "figs")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7})
sim = Simulation(Config(seed=3, N=1, planner="certified", budget=100, save_maps=True)); out = sim.run()
w = sim.world; a = sim.agents[0]; c = a.cert; F = a.fused
fin = c.gate & w.obs & np.isfinite(c.q)
sig, mu = c.sigma[fin], np.abs(c.mu[fin]); q = np.median(c.q[fin])
cat = np.where(F.committed[fin], 0, np.where(F.empty[fin], 2, 1))
fig, ax = plt.subplots(figsize=(3.34, 1.72))
ss = np.linspace(0.5, 1.05, 200)
ax.fill_between(ss, np.abs(q * ss - 1), 1.05, color="#c6dbef", alpha=0.6, label="committed")
ax.fill_between(ss[ss < 1 / q], 0, (1 - q * ss)[ss < 1 / q], color="#a1d99b", alpha=0.7, label="empty")
ax.fill_between(ss[ss > 1 / q], 0, np.minimum(q * ss - 1, 1.05)[ss > 1 / q], color="#fdd835", alpha=0.6, label="undecided")
for k, col, lab in [(0, "C0", None), (1, "C1", None), (2, "C2", None)]:
    m = cat == k
    ax.scatter(sig[m], mu[m], s=2, color=col, alpha=0.4, rasterized=True)
ax.axvline(1 / q, color="k", lw=0.6, ls="--"); ax.axvline(np.sqrt(sim.cfg.noise_var), color="k", lw=0.6, ls=":")
ax.text(1 / q + 0.005, 0.5, "$1/q$", fontsize=7); ax.text(np.sqrt(sim.cfg.noise_var) - 0.045, 0.5, "$\\sigma_{\\min}$", fontsize=7)
ax.set_xlim(0.5, 1.0); ax.set_ylim(0, 1.05); ax.set_xlabel("predictive std $\\sigma$"); ax.set_ylabel("$|\\mu|$")
fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), frameon=False)
plt.tight_layout(rect=(0, 0.1, 1, 1)); os.makedirs(FIGS, exist_ok=True); plt.savefig(os.path.join(FIGS, "mechanism.pdf"))
print("cells", fin.sum(), "q", round(q, 2), "frac sigma<1/q", np.mean(sig < 1 / q).round(3))
