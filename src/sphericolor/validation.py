"""Input and output validation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

METHODS = frozenset({"ray", "simplex", "hsluv", "bivariate"})
DISTANCE_METHODS = frozenset({"ray", "simplex"})
BIVARIATE_MAPS = frozenset(
    {"bremm", "cube_diagonal", "schumann", "steiger", "teuling2", "ziegler"}
)


def validate_method(method: str) -> str:
    if not isinstance(method, str) or method not in METHODS:
        allowed = ", ".join(sorted(METHODS))
        raise ValueError(f"method must be one of: {allowed}")
    return method


def validate_bivariate_map(name: str) -> str:
    if not isinstance(name, str) or name not in BIVARIATE_MAPS:
        allowed = ", ".join(sorted(BIVARIATE_MAPS))
        raise ValueError(f"bivariate_map must be one of: {allowed}")
    return name


def _numeric_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def validate_coordinates(
    latitudes: ArrayLike, longitudes: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lat = _numeric_vector(latitudes, "latitudes")
    lon = _numeric_vector(longitudes, "longitudes")
    if lat.shape != lon.shape:
        raise ValueError("latitudes and longitudes must have equal lengths")
    if np.any((lat < -90.0) | (lat > 90.0)):
        raise ValueError("latitudes must be within [-90, 90]")
    if np.any((lon < -180.0) | (lon > 180.0)):
        raise ValueError("longitudes must be within [-180, 180]")
    return lat, lon


def validate_labels(labels: Sequence[Any] | None, size: int) -> tuple[Any, ...]:
    if labels is None:
        return tuple(range(size))
    result = tuple(labels)
    if len(result) != size:
        raise ValueError("labels must have the same length as the input")
    try:
        unique = set(result)
    except TypeError as exc:
        raise ValueError("labels must be hashable") from exc
    if len(unique) != size:
        raise ValueError("labels must be unique")
    return result


def validate_distance_matrix(
    distances: ArrayLike,
) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(distances, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("distances must contain numeric values") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distances must be a square matrix")
    if matrix.shape[0] < 2:
        raise ValueError("distances must contain at least two entities")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distances must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError("distances must be nonnegative")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-10):
        raise ValueError("distances must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("distances must have an approximately zero diagonal")
    if np.allclose(matrix, 0.0):
        raise ValueError("distances must not be entirely zero")
    return np.ascontiguousarray(matrix)


def validate_colors(rgb: ArrayLike, size: int) -> NDArray[np.float64]:
    colors = np.asarray(rgb, dtype=np.float64)
    if colors.shape != (size, 3):
        raise RuntimeError(f"generated RGB colors must have shape ({size}, 3)")
    if not np.all(np.isfinite(colors)):
        raise RuntimeError("generated RGB colors contain non-finite values")
    if np.any((colors < 0.0) | (colors > 1.0)):
        raise RuntimeError("generated RGB colors fall outside [0, 1]")
    return colors

