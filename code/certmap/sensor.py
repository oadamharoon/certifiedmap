"""2D LiDAR with Gaussian range/bearing/pose noise and labelled-sample generation.

Follows the sensing model of Mahajan et al. (2026), Sec. II-A: the noisy beam
endpoint is labelled occupied (+1) and points along the beam are labelled
free (-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LidarSpec:
    n_beams: int = 40
    fov: float = np.pi          # 180 degrees, forward facing
    max_range: float = 5.0
    sigma_r: float = 0.02       # range noise (m)
    sigma_phi: float = 0.003    # bearing noise (rad)
    sigma_e: float = 0.04       # position noise (m)
    sigma_theta: float = 0.003  # heading noise (rad)
    free_spacing: float = 0.5   # spacing of free-space samples along a beam (m)
    step: float = 0.125         # ray-march step (m)


def _raycast(occ: np.ndarray, res: float, origin: np.ndarray, angles: np.ndarray,
             max_range: float, step: float):
    """First-hit ranges for rays from ``origin`` at ``angles`` (vectorised march).

    Returns (ranges, hit) where hit=False means the ray reached max_range (or
    left the grid, which is treated as no hit and truncated at the boundary).
    """
    ny, nx = occ.shape
    ts = np.arange(step, max_range + 1e-9, step)
    d = np.stack([np.cos(angles), np.sin(angles)], 1)              # (B,2)
    pts = origin[None, None, :] + ts[None, :, None] * d[:, None, :]  # (B,S,2)
    ix = np.floor(pts[..., 0] / res).astype(int)
    iy = np.floor(pts[..., 1] / res).astype(int)
    out = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)
    ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
    blocked = occ[iyc, ixc] | out
    first = np.argmax(blocked, axis=1)                 # index of first True, 0 if none
    any_hit = blocked[np.arange(len(angles)), first]
    r = np.where(any_hit, ts[first], max_range)
    hit = any_hit & ~out[np.arange(len(angles)), first]
    return r, hit


class Lidar:
    def __init__(self, spec: LidarSpec):
        self.s = spec
        self.rel_angles = np.linspace(-spec.fov / 2, spec.fov / 2, spec.n_beams)

    def scan(self, occ: np.ndarray, res: float, pose: np.ndarray, rng: np.random.Generator,
             place_pose: np.ndarray | None = None):
        """One noisy scan -> labelled samples (X (n,2), y (n,)) placed with the noisy pose.

        Rays are cast from the true pose; samples are placed relative to `place_pose` when given, which is how a
        drifting pose estimate enters the map: the geometry the robot sees is right, where it writes it is wrong.
        """
        s = self.s
        x, y, th = pose
        px, py, pth = (x, y, th) if place_pose is None else tuple(place_pose)
        r, hit = _raycast(occ, res, np.array([x, y]), th + self.rel_angles, s.max_range, s.step)
        r_t = np.clip(r + rng.normal(0, s.sigma_r, r.shape), 0.05, None)
        phi_t = self.rel_angles + rng.normal(0, s.sigma_phi, r.shape)
        pose_t = np.array([px, py]) + rng.normal(0, s.sigma_e, 2)
        th_t = pth + rng.normal(0, s.sigma_theta)
        d = np.stack([np.cos(th_t + phi_t), np.sin(th_t + phi_t)], 1)
        Xs, ys = [], []
        end = pose_t[None, :] + r_t[:, None] * d
        Xs.append(end[hit]); ys.append(np.ones(hit.sum()))
        # free samples at free_spacing, ..., up to r - free_spacing/2 (or max_range if no hit)
        for b in range(s.n_beams):
            rr = r_t[b] if hit[b] else s.max_range
            ts = np.arange(s.free_spacing, rr - s.free_spacing / 2 + 1e-9, s.free_spacing)
            if len(ts):
                Xs.append(pose_t[None, :] + ts[:, None] * d[b][None, :]); ys.append(-np.ones(len(ts)))
        X = np.concatenate(Xs, 0) if Xs else np.zeros((0, 2))
        yv = np.concatenate(ys, 0) if ys else np.zeros(0)
        return X, yv

    def predict_samples(self, est_occ: np.ndarray, res: float, pose: np.ndarray,
                        n_beams: int | None = None, free_spacing: float | None = None):
        """Noise-free predicted sample locations against an *estimated* map (for planning)."""
        s = self.s
        nb = n_beams or s.n_beams
        fs = free_spacing or s.free_spacing
        angles = pose[2] + np.linspace(-s.fov / 2, s.fov / 2, nb)
        r, hit = _raycast(est_occ, res, pose[:2], angles, s.max_range, s.step)
        d = np.stack([np.cos(angles), np.sin(angles)], 1)
        pts = [pose[None, :2] + r[hit][:, None] * d[hit]]
        for b in range(nb):
            rr = r[b] if hit[b] else s.max_range
            ts = np.arange(fs, rr - fs / 2 + 1e-9, fs)
            if len(ts):
                pts.append(pose[None, :2] + ts[:, None] * d[b][None, :])
        return np.concatenate(pts, 0)
