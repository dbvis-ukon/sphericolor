"""Geographic distance utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EARTH_RADIUS_KM = 6371.0088


def great_circle_matrix(
    latitudes: NDArray[np.float64], longitudes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return pairwise great-circle distances in kilometers."""
    lat = np.radians(latitudes)
    lon = np.radians(longitudes)
    delta_lat = lat[:, None] - lat[None, :]
    delta_lon = lon[:, None] - lon[None, :]
    hav = np.sin(delta_lat / 2.0) ** 2
    hav += np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(delta_lon / 2.0) ** 2
    angular = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))
    distances = EARTH_RADIUS_KM * angular
    np.fill_diagonal(distances, 0.0)
    return np.asarray(distances, dtype=np.float64)
