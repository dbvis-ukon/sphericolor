"""Centroid-ray gamut projection."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from ..gamut import gamut_centroid, srgb_hull


def _scale_to_cam16ucs_bounds(
    embedding: NDArray[np.float64], scalar: float = 1.1
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Scale each axis around the gamut centroid."""
    source_min = embedding.min(axis=0)
    source_extent = np.ptp(embedding, axis=0)
    if np.any(source_extent <= 1e-12):
        raise ValueError("MDS embedding is degenerate")
    target_extent = np.asarray([100.0, 100.0, 100.0])
    axis_scale = target_extent / source_extent
    normalized = (embedding - source_min) / source_extent - 0.5
    center = gamut_centroid()
    return center + scalar * normalized * target_extent, axis_scale


def project_directions_to_gamut(
    embedding: NDArray[np.float64], *, chunk_size: int = 4096, pre_scale: bool = True
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Project embedding directions onto the convex gamut boundary."""
    hull = srgb_hull()
    projection_center = gamut_centroid()
    if pre_scale:
        source, axis_scale = _scale_to_cam16ucs_bounds(embedding)
        centered = source - projection_center
    else:
        axis_scale = np.ones(3)
        centered = embedding - embedding.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1)
    central = norms <= 1e-12

    directions = np.zeros_like(centered)
    directions[~central] = centered[~central] / norms[~central, None]
    projected = np.repeat(projection_center[None, :], embedding.shape[0], axis=0)

    normals = np.asarray(hull.equations[:, :3], dtype=np.float64)
    offsets = np.asarray(hull.equations[:, 3], dtype=np.float64)
    numerators = -(normals @ projection_center + offsets)

    active = np.flatnonzero(~central)
    for start in range(0, active.size, chunk_size):
        indices = active[start : start + chunk_size]
        denominators = directions[indices] @ normals.T
        candidates = np.full_like(denominators, np.inf)
        np.divide(
            numerators[None, :],
            denominators,
            out=candidates,
            where=denominators > 1e-12,
        )
        ray_lengths = np.min(candidates, axis=1)
        if not np.all(np.isfinite(ray_lengths)):
            raise RuntimeError("could not intersect all embedding rays with the sRGB gamut")
        projected[indices] = (
            projection_center + directions[indices] * ray_lengths[:, None]
        )

    mean_distance = float(np.mean(pdist(projected))) if embedding.shape[0] > 1 else 0.0
    return projected, {
        "central_point_count": int(np.count_nonzero(central)),
        "chunk_size": chunk_size,
        "pre_scaled": pre_scale,
        "axis_scale": axis_scale.tolist(),
        "mean_pairwise_distance": mean_distance,
    }
