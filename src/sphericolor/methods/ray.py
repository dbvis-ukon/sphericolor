"""Centroid-ray gamut projection."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..gamut import gamut_centroid, srgb_hull


def project_directions_to_gamut(
    embedding: NDArray[np.float64], *, chunk_size: int = 4096
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Project centered embedding directions onto the convex gamut boundary."""
    hull = srgb_hull()
    centroid = gamut_centroid()
    centered = embedding - embedding.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1)
    central = norms <= 1e-12

    directions = np.zeros_like(centered)
    directions[~central] = centered[~central] / norms[~central, None]
    projected = np.repeat(centroid[None, :], embedding.shape[0], axis=0)

    normals = np.asarray(hull.equations[:, :3], dtype=np.float64)
    offsets = np.asarray(hull.equations[:, 3], dtype=np.float64)
    numerators = -(normals @ centroid + offsets)

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
        projected[indices] = centroid + directions[indices] * ray_lengths[:, None]

    return projected, {
        "central_point_count": int(np.count_nonzero(central)),
        "chunk_size": chunk_size,
    }

