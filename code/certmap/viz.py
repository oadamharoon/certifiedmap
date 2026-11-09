"""Plot helpers: fused partition maps and trajectories."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# categories: 0 nodata, 1 committed free, 2 committed occupied, 3 undecided, 4 empty
CMAP = ListedColormap(["#d9d9d9", "#c6dbef", "#cb181d", "#fdd835", "#31a354"])
NAMES = ["no data", "free", "occupied", "undecided", "empty"]


def partition_image(m: dict, obs: np.ndarray, ny: int, nx: int) -> np.ndarray:
    cat = np.zeros(ny * nx, int)
    cat[m["committed"] & (m["label"] < 0)] = 1
    cat[m["committed"] & (m["label"] > 0)] = 2
    cat[m["undecided"] & ~m["nodata"]] = 3
    cat[m["empty"]] = 4
    cat[~obs] = 0
    return cat.reshape(ny, nx)


def plot_run(out: dict, ax=None, title: str = "", agent: int = 0, res: float = 0.25):
    occ = out["occ"]; ny, nx = occ.shape
    obs = out.get("obs")
    if obs is None:
        from scipy import ndimage
        surf = occ & ndimage.binary_dilation(~occ, structure=np.ones((3, 3)))
        obs = (~occ | surf).ravel()
    img = partition_image(out["maps"][agent], obs, ny, nx)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.imshow(img, cmap=CMAP, vmin=0, vmax=4, origin="lower", extent=[0, nx * res, 0, ny * res], interpolation="nearest")
    ax.contour(np.arange(nx) * res + res / 2, np.arange(ny) * res + res / 2, occ.astype(float), levels=[0.5], colors="k", linewidths=0.5)
    for k, tr in enumerate(out["traj"]):
        ax.plot(tr[:, 0], tr[:, 1], "-", lw=1.2, color=f"C{k}")
        ax.plot(tr[0, 0], tr[0, 1], "o", color=f"C{k}", ms=5, mec="k")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return ax
