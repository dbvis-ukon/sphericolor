"""Full 3D Nelder-Mead gamut fitting."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.transform import Rotation

from ..gamut import gamut_centroid, load_srgb_gamut, srgb_hull

W_STRESS = 0.1
W_OUTSIDE = 1.0
W_SCALE = 0.1


def apply_transform(
    points: NDArray[np.float64], params: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply translation, full rotation, and uniform exponential scale."""
    center = points.mean(axis=0)
    rotated = Rotation.from_rotvec(params[3:6]).apply(points - center)
    return np.asarray(np.exp(params[6]) * rotated + center + params[:3], dtype=np.float64)


def _normalize_embedding(
    embedding: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    gamut = load_srgb_gamut()
    centered = embedding - embedding.mean(axis=0)
    embedding_extent = float(np.linalg.norm(np.ptp(embedding, axis=0)))
    if embedding_extent <= 1e-12:
        raise ValueError("MDS embedding is degenerate")
    scale = float(np.mean(np.ptp(gamut, axis=0)) / embedding_extent)
    return centered * scale + gamut_centroid(), scale


def fit_embedding_to_gamut(
    embedding: NDArray[np.float64], distances: NDArray[np.float64]
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Fit an embedding to the convex gamut using a seven-parameter transform."""
    base, normalization_scale = _normalize_embedding(embedding)
    target = squareform(distances * normalization_scale, checks=False)
    equations = np.asarray(srgb_hull().equations, dtype=np.float64)

    def objective(params: NDArray[np.float64]) -> float:
        points = apply_transform(base, params)
        stress = float(np.mean((pdist(points) - target) ** 2))
        signed = points @ equations[:, :3].T + equations[:, 3]
        outside = np.maximum(np.max(signed, axis=1), 0.0)
        penalty = float(np.sum(outside**2))
        scale = float(np.exp(params[6]))
        value = W_STRESS * stress + W_OUTSIDE * penalty - W_SCALE * scale
        return value if np.isfinite(value) else float("inf")

    x0 = np.zeros(7, dtype=np.float64)
    mean_extent = float(np.mean(np.ptp(load_srgb_gamut(), axis=0)))
    steps = np.asarray(
        [mean_extent * 0.25] * 3 + [np.pi / 4.0] * 3 + [0.25], dtype=np.float64
    )
    initial_simplex = np.vstack([x0, x0 + np.diag(steps)])
    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": initial_simplex,
            "maxiter": 5000,
            "xatol": 1e-7,
            "fatol": 1e-7,
            "adaptive": False,
        },
    )
    params = np.asarray(result.x, dtype=np.float64)
    output = apply_transform(base, params)
    if not np.all(np.isfinite(output)) or not np.isfinite(result.fun):
        raise RuntimeError("Simplex optimization produced non-finite output")

    return output, {
        "converged": bool(result.success),
        "iterations": int(result.nit),
        "objective": float(result.fun),
        "message": str(result.message),
        "normalization_scale": normalization_scale,
        "transform": params.tolist(),
    }
