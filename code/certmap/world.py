"""Procedural 2D occupancy worlds on a regular grid.

Conventions
-----------
* Grid array ``occ[iy, ix]`` is True where the cell is occupied.
* Metric coordinates x in [0, W), y in [0, H); cell (ix, iy) covers
  [ix*res, (ix+1)*res) x [iy*res, (iy+1)*res).
* Labels: +1 occupied, -1 free (as in Mahajan et al. 2026).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


class World:
    def __init__(self, occ: np.ndarray, res: float):
        self.res = float(res)
        self.ny, self.nx = occ.shape
        self.W = self.nx * self.res
        self.H = self.ny * self.res
        self.M = self.nx * self.ny
        # query grid = cell centres, row-major (iy, ix) -> flat index iy*nx+ix
        ix, iy = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        self.Q = np.stack([(ix.ravel() + 0.5) * self.res, (iy.ravel() + 0.5) * self.res], 1)
        self.set_occ(occ)

    def set_occ(self, occ: np.ndarray):
        """(Re)set the occupancy grid; labels and the observable domain follow (used for moving hazards)."""
        self.occ = occ.astype(bool)
        self.labels = np.where(self.occ.ravel(), 1.0, -1.0)
        # observable cells: free cells and occupied cells with a free 8-neighbour.
        # Obstacle interiors cannot be sampled by any range sensor and are excluded
        # from the map's query domain.
        surf = self.occ & ndimage.binary_dilation(~self.occ, structure=np.ones((3, 3)))
        self.obs = (~self.occ | surf).ravel()

    # ---- coordinate helpers -------------------------------------------------
    def cell_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xy = np.atleast_2d(xy)
        ix = np.clip(np.floor(xy[:, 0] / self.res).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor(xy[:, 1] / self.res).astype(int), 0, self.ny - 1)
        return ix, iy

    def flat_of(self, xy: np.ndarray) -> np.ndarray:
        ix, iy = self.cell_of(xy)
        return iy * self.nx + ix

    def is_occ(self, xy: np.ndarray) -> np.ndarray:
        ix, iy = self.cell_of(xy)
        return self.occ[iy, ix]

    def inside(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(xy)
        return (xy[:, 0] >= 0) & (xy[:, 0] < self.W) & (xy[:, 1] >= 0) & (xy[:, 1] < self.H)


def make_world(seed: int, W: float = 30.0, H: float = 20.0, res: float = 0.25,
               wall_thick: float = 0.25, n_walls: int = 8, n_boxes: int = 6,
               box_size=(1.0, 3.0), door: float = 1.5) -> World:
    """Random indoor-like world: boundary + axis-aligned walls with doors + boxes.

    ``wall_thick`` controls the fraction of boundary (mixed) cells, i.e. the
    aleatoric ambiguity of the map at the query resolution.
    """
    rng = np.random.default_rng(seed)
    nx, ny = int(round(W / res)), int(round(H / res))
    occ = np.zeros((ny, nx), bool)
    t = max(1, int(round(wall_thick / res)))
    occ[:t, :] = occ[-t:, :] = True
    occ[:, :t] = occ[:, -t:] = True

    def fill(x0, y0, x1, y1):
        ix0, ix1 = int(np.floor(x0 / res)), int(np.ceil(x1 / res))
        iy0, iy1 = int(np.floor(y0 / res)), int(np.ceil(y1 / res))
        occ[max(iy0, 0):min(iy1, ny), max(ix0, 0):min(ix1, nx)] = True

    for _ in range(n_walls):
        vertical = rng.random() < 0.5
        L = rng.uniform(4.0, 12.0)
        if vertical:
            x = rng.uniform(2.0, W - 2.0)
            y0 = rng.uniform(0.0, H - L)
            d0 = y0 + rng.uniform(0.5, L - door - 0.5)
            fill(x, y0, x + wall_thick, d0)
            fill(x, d0 + door, x + wall_thick, y0 + L)
        else:
            y = rng.uniform(2.0, H - 2.0)
            x0 = rng.uniform(0.0, W - L)
            d0 = x0 + rng.uniform(0.5, L - door - 0.5)
            fill(x0, y, d0, y + wall_thick)
            fill(d0 + door, y, x0 + L, y + wall_thick)
    for _ in range(n_boxes):
        bw, bh = rng.uniform(*box_size, size=2)
        x0, y0 = rng.uniform(1.0, W - bw - 1.0), rng.uniform(1.0, H - bh - 1.0)
        fill(x0, y0, x0 + bw, y0 + bh)

    # keep only the largest free component so every free cell is reachable
    lab, n = ndimage.label(~occ)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        keep = 1 + int(np.argmax(sizes))
        occ[(lab != keep) & (~occ)] = True
    return World(occ, res)


def sample_starts(world: World, n: int, rng: np.random.Generator, clearance: float = 0.6,
                  min_sep: float = 2.0) -> np.ndarray:
    """n start poses (x, y, theta) in free space with clearance from obstacles."""
    dist = ndimage.distance_transform_edt(~world.occ) * world.res
    cand = np.argwhere(dist > clearance)
    starts = []
    for _ in range(10000):
        iy, ix = cand[rng.integers(len(cand))]
        p = np.array([(ix + 0.5) * world.res, (iy + 0.5) * world.res])
        if all(np.linalg.norm(p - s[:2]) >= min_sep for s in starts):
            starts.append(np.array([p[0], p[1], rng.uniform(-np.pi, np.pi)]))
            if len(starts) == n:
                break
    return np.array(starts)
