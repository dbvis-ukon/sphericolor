"""Bundled sRGB gamut access."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull


@lru_cache(maxsize=1)
def load_srgb_gamut() -> NDArray[np.float64]:
    """Load the bundled ASCII PLY vertices."""
    resource = files("sphericolor.data").joinpath("srgb_cam16ucs.ply")
    points: list[list[float]] = []
    in_header = True
    with resource.open("r", encoding="utf-8") as handle:
        for line in handle:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] < 4:
        raise RuntimeError("bundled sRGB gamut is invalid")
    result.setflags(write=False)
    return result


@lru_cache(maxsize=1)
def srgb_hull() -> ConvexHull:
    return ConvexHull(load_srgb_gamut())


def gamut_centroid() -> NDArray[np.float64]:
    return np.asarray(load_srgb_gamut().mean(axis=0), dtype=np.float64)

