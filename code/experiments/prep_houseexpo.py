"""Rasterise HouseExpo floor plans (free-space polygons, metres) onto the 30x20 m / 0.25 m grid.

Usage: python -I prep_houseexpo.py <dir of HouseExpo json files> [n plans] [output dir]
The first 20 plans that fit the workspace are the set used by the papers (data/houseexpo); larger sets are for
training the occupancy predictor and go in their own directory so the paper's 20 stay fixed.
"""
import json, os, sys, glob
import numpy as np
from matplotlib.path import Path
from scipy import ndimage
SRC = sys.argv[1]
N_WANT = int(sys.argv[2]) if len(sys.argv) > 2 else 20          # how many plans to rasterise
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), "..", "data", "houseexpo")
os.makedirs(OUT, exist_ok=True)
W, H, res = 30.0, 20.0, 0.25
nx, ny = int(W / res), int(H / res)
files = sorted(glob.glob(os.path.join(SRC, "*.json")))
picked = []
for f in files:
    d = json.load(open(f)); bb = d["bbox"]; w = bb["max"][0] - bb["min"][0]; h = bb["max"][1] - bb["min"][1]
    if w < h: w, h = h, w
    if 18 <= w <= 28 and 10 <= h <= 18 and d["room_num"] >= 5:
        picked.append((f, d))
    if len(picked) >= N_WANT: break
print(len(picked), "plans")
ix, iy = np.meshgrid(np.arange(nx), np.arange(ny)); cx = (ix + 0.5) * res; cy = (iy + 0.5) * res
for k, (f, d) in enumerate(picked):
    v = np.array(d["verts"], float); bb = d["bbox"]
    v = v - np.array(bb["min"]); ext = np.array(bb["max"]) - np.array(bb["min"])
    if ext[0] < ext[1]: v = v[:, ::-1]; ext = ext[::-1]           # long side along x
    v = v + (np.array([W, H]) - ext) / 2                            # centre in the workspace
    inside = Path(v).contains_points(np.stack([cx.ravel(), cy.ravel()], 1)).reshape(ny, nx)
    # polygon edges (exterior and interior walls, encoded as slivers of the outline) as 1-cell walls
    edge = np.zeros((ny, nx), bool)
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        n_s = max(2, int(np.ceil(np.linalg.norm(b - a) / (res / 3))))
        pts = a[None, :] + np.linspace(0, 1, n_s)[:, None] * (b - a)[None, :]
        jx = np.clip((pts[:, 0] / res).astype(int), 0, nx - 1); jy = np.clip((pts[:, 1] / res).astype(int), 0, ny - 1)
        edge[jy, jx] = True
    occ = ~inside | edge
    lab, n = ndimage.label(~occ)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)); keep = 1 + int(np.argmax(sizes)); occ[(lab != keep)] = True
    np.save(os.path.join(OUT, f"he{k:02d}.npy"), occ)
    print(k, d["id"][:8], f"{ext[0]:.1f}x{ext[1]:.1f} m rooms={d['room_num']} free={(~occ).mean():.2f}")
