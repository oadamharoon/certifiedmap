"""Local Gaussian features on a regular lattice (sparse rows, banded Gram matrix)."""
from __future__ import annotations

import numpy as np
from scipy import sparse


class LocalFeatures:
    def __init__(self, W: float, H: float, spacing: float = 0.5, ell: float = 0.4, support: float = 3.0):
        self.spacing, self.ell = spacing, ell
        self.nx = int(np.floor(W / spacing)) + 1
        self.ny = int(np.floor(H / spacing)) + 1
        self.n = self.nx * self.ny
        self.R = int(np.ceil(support * ell / spacing))
        # index offsets of candidate centres around the nearest lattice node
        o = np.arange(-self.R, self.R + 1)
        self.dox, self.doy = np.meshgrid(o, o)
        self.dox, self.doy = self.dox.ravel(), self.doy.ravel()
        self.cut = np.exp(-0.5 * (support) ** 2)
        # normalisation so that sum_m phi_m(x)^2 ~= 1 in the interior
        self.norm = 1.0
        p = self.phi(np.array([[W / 2, H / 2]]))
        self.norm = 1.0 / np.sqrt(p.multiply(p).sum())
        self.bandwidth = 2 * self.R * self.nx + 2 * self.R

    def phi(self, X: np.ndarray) -> sparse.csr_matrix:
        X = np.atleast_2d(X)
        n = len(X)
        cx = np.rint(X[:, 0] / self.spacing).astype(int)
        cy = np.rint(X[:, 1] / self.spacing).astype(int)
        ix = cx[:, None] + self.dox[None, :]
        iy = cy[:, None] + self.doy[None, :]
        valid = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        zx, zy = ix * self.spacing, iy * self.spacing
        d2 = (X[:, 0, None] - zx) ** 2 + (X[:, 1, None] - zy) ** 2
        v = np.exp(-0.5 * d2 / self.ell ** 2) * self.norm
        valid &= v > self.cut * self.norm
        rows = np.repeat(np.arange(n), valid.sum(1))
        cols = (iy * self.nx + ix)[valid]
        return sparse.csr_matrix((v[valid], (rows, cols)), shape=(n, self.n))
