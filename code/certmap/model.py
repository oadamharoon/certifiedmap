"""Bayesian linear occupancy map on local features (a Bayesian Hilbert map).

    f(x) = w^T phi(x),  w ~ N(0, tau^2 I),  y = f(x) + eps,  eps ~ N(0, sn^2)

Posterior precision Lam = I/tau^2 + Phi^T Phi / sn^2 is banded (features are
local), so the posterior covariance is obtained with a banded Cholesky solve.
Predictive variance depends only on sample *locations*, which is what makes
the preposterior acquisition of the planner closed-form.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import solveh_banded

from .features import LocalFeatures


class BayesLinearMap:
    def __init__(self, feats: LocalFeatures, noise_var: float = 0.3, prior_var: float = 1.0):
        self.f = feats
        self.sn2 = float(noise_var)
        self.tau2 = float(prior_var)
        n = feats.n
        self.Lam = np.eye(n) / self.tau2
        self.b = np.zeros(n)
        self.n_data = 0
        self.Sigma = None
        self.w = None
        self._dirty = True

    # ---- data -------------------------------------------------------------
    def add(self, X: np.ndarray, y: np.ndarray):
        if len(X) == 0:
            return
        Phi = self.f.phi(X)
        G = (Phi.T @ Phi).tocoo()
        np.add.at(self.Lam, (G.row, G.col), G.data / self.sn2)
        self.b += Phi.T @ y / self.sn2
        self.n_data += len(X)
        self._dirty = True

    def copy(self) -> "BayesLinearMap":
        m = BayesLinearMap.__new__(BayesLinearMap)
        m.f, m.sn2, m.tau2 = self.f, self.sn2, self.tau2
        m.Lam, m.b, m.n_data = self.Lam.copy(), self.b.copy(), self.n_data
        m.Sigma, m.w, m._dirty = None, None, True
        return m

    def solve(self):
        if not self._dirty:
            return
        n, bw = self.f.n, self.f.bandwidth
        ab = np.zeros((bw + 1, n))
        for d in range(bw + 1):
            ab[bw - d, d:] = np.diagonal(self.Lam, d)
        self.w = solveh_banded(ab, self.b, lower=False, check_finite=False)
        # float32 banded solve for the covariance: halves memory traffic; ~1e-4 relative
        # accuracy is ample for predictive variances
        self.Sigma = solveh_banded(ab.astype(np.float32), np.eye(n, dtype=np.float32),
                                   lower=False, check_finite=False).astype(np.float64)
        self._dirty = False

    # ---- prediction -------------------------------------------------------
    @staticmethod
    def _quad(Phi: sparse.csr_matrix, S: np.ndarray, chunk: int = 4096) -> np.ndarray:
        """phi_i^T S phi_i for every row i of a sparse Phi, via local gathers of S."""
        n = Phi.shape[0]
        out = np.empty(n)
        if n == 0:
            return out
        nnz = np.diff(Phi.indptr)
        kmax = int(nnz.max())
        for s0 in range(0, n, chunk):
            s1 = min(n, s0 + chunk)
            m = s1 - s0
            idx = np.zeros((m, kmax), dtype=np.int64)
            val = np.zeros((m, kmax))
            rows = np.repeat(np.arange(m), nnz[s0:s1])
            cols = np.concatenate([np.arange(k) for k in nnz[s0:s1]])
            p0, p1 = Phi.indptr[s0], Phi.indptr[s1]
            idx[rows, cols] = Phi.indices[p0:p1]
            val[rows, cols] = Phi.data[p0:p1]
            G = S[idx[:, :, None], idx[:, None, :]]              # (m, k, k)
            out[s0:s1] = np.einsum("ik,ikl,il->i", val, G, val)
        return out

    def predict(self, Phi: sparse.csr_matrix):
        """Predictive mean and std (including observation noise) at rows of Phi."""
        self.solve()
        mu = np.clip(Phi @ self.w, -1.0, 1.0)   # labels are +-1; clipping is a fixed post-hoc map
        var_f = self._quad(Phi, self.Sigma)
        return mu, np.sqrt(np.maximum(var_f, 0.0) + self.sn2)

    def latent_var(self, Phi: sparse.csr_matrix, Sigma: np.ndarray | None = None):
        return self._quad(Phi, self.Sigma if Sigma is None else Sigma)

    def var_after(self, PhiS: sparse.csr_matrix, PhiQ: sparse.csr_matrix):
        """Latent variance at rows of PhiQ after observing rows of PhiS (values unknown).

        Local Woodbury: with A = PhiS_L^T PhiS_L / sn^2 on the touched feature set L,
            Sigma_S = Sigma - Sigma[:,L] A (I + Sigma[L,L] A)^{-1} Sigma[L,:].
        """
        self.solve()
        L = np.unique(PhiS.indices)
        if len(L) == 0:
            return self.latent_var(PhiQ)
        Lp = np.unique(PhiQ.indices)
        A = (PhiS[:, L].T @ PhiS[:, L]).toarray() / self.sn2
        SLL = self.Sigma[np.ix_(L, L)]
        SpL = self.Sigma[np.ix_(Lp, L)]
        K = np.linalg.solve((np.eye(len(L)) + SLL @ A).T, (SpL @ A).T).T   # SpL A (I+SLL A)^{-1}
        Spp = self.Sigma[np.ix_(Lp, Lp)] - K @ SpL.T
        return self._quad(PhiQ[:, Lp].tocsr(), Spp)
