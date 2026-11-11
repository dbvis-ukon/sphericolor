"""HSLuv double-cone baseline."""

from __future__ import annotations

import hsluv
import numpy as np
from numpy.typing import NDArray


def hsluv_colors(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64]
) -> NDArray[np.float64]:
    output = np.empty((latitudes.size, 3), dtype=np.float64)
    for index, (latitude, longitude) in enumerate(zip(latitudes, longitudes)):  # noqa: B905
        hue = longitude + 180.0
        lightness = (latitude + 90.0) / 180.0 * 100.0
        output[index] = hsluv.hsluv_to_rgb([hue, 100.0, lightness])
    return np.clip(output, 0.0, 1.0)
