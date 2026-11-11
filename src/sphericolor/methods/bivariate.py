"""Planar bivariate colormap baselines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pycolormap_2d import (
    ColorMap2DBremm,
    ColorMap2DCubeDiagonal,
    ColorMap2DSchumann,
    ColorMap2DSteiger,
    ColorMap2DTeuling2,
    ColorMap2DZiegler,
)

MAPS = {
    "bremm": ColorMap2DBremm,
    "cube_diagonal": ColorMap2DCubeDiagonal,
    "schumann": ColorMap2DSchumann,
    "steiger": ColorMap2DSteiger,
    "teuling2": ColorMap2DTeuling2,
    "ziegler": ColorMap2DZiegler,
}


def bivariate_colors(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64], name: str
) -> NDArray[np.float64]:
    color_map = MAPS[name]()
    output = np.empty((latitudes.size, 3), dtype=np.float64)
    for index, (latitude, longitude) in enumerate(zip(latitudes, longitudes)):  # noqa: B905
        x = (longitude + 180.0) / 360.0
        y = (90.0 - latitude) / 180.0
        color = np.asarray(color_map(x, y), dtype=np.float64)
        if np.max(color) > 1.0:
            color = color / 255.0
        output[index] = color[:3]
    return np.clip(output, 0.0, 1.0)
