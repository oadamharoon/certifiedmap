"""Block-protocol diagram: why the certificate stays valid although the planner chooses where to look.

One block is a 5 m segment. Scans taken over the first 60 % of it train the model; the rest calibrate it. The
model, the gate and the recalibrated level of a block are therefore functions of data collected strictly before
that block's calibration fold, which is the measurability condition Proposition 1 needs.
"""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

FIGS = os.path.join(os.path.dirname(__file__), "..", "..", "paper_iccps", "figs"); os.makedirs(FIGS, exist_ok=True)
plt.rcParams.update({"font.size": 8})
TRAIN, CAL = "#9ecae1", "#fdae6b"

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7.1, 1.45))
    y, W_TR, W_CA, GAP = 0.0, 2.1, 1.4, 0.55
    xs = [0.0, W_TR + W_CA + GAP, 2 * (W_TR + W_CA + GAP)]
    names = ["block $k-1$", "block $k$", "block $k+1$"]
    for x0, nm in zip(xs, names):
        ax.add_patch(Rectangle((x0, y), W_TR, 0.5, facecolor=TRAIN, edgecolor="k", lw=0.6))
        ax.add_patch(Rectangle((x0 + W_TR, y), W_CA, 0.5, facecolor=CAL, edgecolor="k", lw=0.6))
        ax.text(x0 + W_TR / 2, y + 0.25, "train", ha="center", va="center", fontsize=7)
        ax.text(x0 + W_TR + W_CA / 2, y + 0.25, "calibrate", ha="center", va="center", fontsize=7)
        ax.text(x0 + (W_TR + W_CA) / 2, y + 0.60, nm, ha="center", fontsize=8)
        ax.plot([x0, x0 + W_TR + W_CA], [y - 0.16, y - 0.16], lw=0.6, color="0.45")
        ax.text(x0 + (W_TR + W_CA) / 2, y - 0.34, "one 5 m segment", ha="center", fontsize=6.5, color="0.35")
    # the conditioning set: everything up to and including block k's training fold
    gx0, gx1 = xs[0] - 0.08, xs[1] + W_TR + 0.08
    ax.plot([gx0, gx0, gx1, gx1], [y + 0.86, y + 0.98, y + 0.98, y + 0.86], lw=0.9, color="#2171b5")
    ax.text((gx0 + gx1) / 2, y + 1.06, r"$\mathcal{G}_k$ fixes the model, the gate and the level",
            ha="center", fontsize=7.5, color="#2171b5")
    ax.annotate("", xy=(xs[1] + W_TR + W_CA / 2, y + 0.54), xytext=((gx0 + gx1) / 2, y + 0.92),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#2171b5",
                                connectionstyle="arc3,rad=-0.3"))
    # planning happens between blocks, from data of earlier blocks only
    ax.annotate("", xy=(xs[2] - 0.05, y + 0.25), xytext=(xs[1] + W_TR + W_CA + 0.05, y + 0.25),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
    ax.text((xs[1] + W_TR + W_CA + xs[2]) / 2, y + 0.33, "plan", ha="center", fontsize=6.8)
    ax.set_xlim(-0.4, xs[2] + W_TR + W_CA + 0.4); ax.set_ylim(-0.55, 1.3); ax.axis("off")
    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(FIGS, "protocol.pdf")); print("protocol.pdf")
