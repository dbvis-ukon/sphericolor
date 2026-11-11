"""Public generation API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .colors import cam16ucs_to_rgb, rgb_to_cam16ucs, rgb_to_hex
from .diagnostics import build_diagnostics
from .distances import great_circle_matrix
from .embedding import metric_mds
from .methods.bivariate import bivariate_colors
from .methods.hsluv import hsluv_colors
from .methods.ray import project_directions_to_gamut
from .methods.simplex import fit_embedding_to_gamut
from .result import ColorMapping, Diagnostics, immutable_metadata, readonly
from .validation import (
    DISTANCE_METHODS,
    validate_bivariate_map,
    validate_colors,
    validate_coordinates,
    validate_distance_matrix,
    validate_labels,
    validate_method,
)


def _mapping(
    *,
    labels: tuple[Any, ...],
    rgb: NDArray[np.float64],
    method: str,
    metadata: dict[str, Any],
    diagnostics: Diagnostics | None,
) -> ColorMapping:
    checked_rgb = validate_colors(rgb, len(labels))
    cam16ucs = rgb_to_cam16ucs(checked_rgb)
    return ColorMapping(
        labels=labels,
        rgb=readonly(checked_rgb),
        hex=readonly(rgb_to_hex(checked_rgb)),
        cam16ucs=readonly(cam16ucs),
        method=method,
        metadata=immutable_metadata(metadata),
        diagnostics=diagnostics,
    )


def _distance_mapping(
    distances: NDArray[np.float64],
    labels: tuple[Any, ...],
    method: str,
    random_state: int | None,
    diagnostics: bool,
    metadata: dict[str, Any],
) -> ColorMapping:
    embedding = metric_mds(distances, random_state)
    if method == "ray":
        mapped, method_metadata = project_directions_to_gamut(embedding)
    else:
        mapped, method_metadata = fit_embedding_to_gamut(embedding, distances)
    rgb, clipped = cam16ucs_to_rgb(mapped)
    displayed_cam16ucs = rgb_to_cam16ucs(rgb)
    metadata.update(method_metadata)
    metadata.update({"clipped_color_count": clipped, "random_state": random_state})
    diagnostic_result = (
        build_diagnostics(distances, displayed_cam16ucs, embedding, mapped)
        if diagnostics
        else None
    )
    return _mapping(
        labels=labels,
        rgb=rgb,
        method=method,
        metadata=metadata,
        diagnostics=diagnostic_result,
    )


def from_latlon(
    latitudes: ArrayLike,
    longitudes: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
    method: str = "ray",
    random_state: int | None = 69,
    diagnostics: bool = False,
    bivariate_map: str = "cube_diagonal",
) -> ColorMapping:
    """Generate colors for an ordered set of latitude-longitude points."""
    selected_method = validate_method(method)
    lat, lon = validate_coordinates(latitudes, longitudes)
    checked_labels = validate_labels(labels, lat.size)
    distances = great_circle_matrix(lat, lon)
    metadata: dict[str, Any] = {
        "input": "latitude_longitude",
        "distance": "great_circle_km",
    }

    if selected_method in DISTANCE_METHODS:
        return _distance_mapping(
            distances,
            checked_labels,
            selected_method,
            random_state,
            diagnostics,
            metadata,
        )

    if selected_method == "hsluv":
        rgb = hsluv_colors(lat, lon)
    else:
        selected_bivariate = validate_bivariate_map(bivariate_map)
        rgb = bivariate_colors(lat, lon, selected_bivariate)
        metadata["bivariate_map"] = selected_bivariate
    cam16ucs = rgb_to_cam16ucs(rgb)
    diagnostic_result = (
        build_diagnostics(distances, cam16ucs, None, None) if diagnostics else None
    )
    metadata["clipped_color_count"] = 0
    return _mapping(
        labels=checked_labels,
        rgb=rgb,
        method=selected_method,
        metadata=metadata,
        diagnostics=diagnostic_result,
    )


def from_distance_matrix(
    distances: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
    method: str = "ray",
    random_state: int | None = 69,
    diagnostics: bool = False,
) -> ColorMapping:
    """Generate colors from an ordered pairwise-distance matrix."""
    selected_method = validate_method(method)
    if selected_method not in DISTANCE_METHODS:
        raise ValueError("distance-matrix input supports only ray and simplex")
    matrix = validate_distance_matrix(distances)
    checked_labels = validate_labels(labels, matrix.shape[0])
    return _distance_mapping(
        matrix,
        checked_labels,
        selected_method,
        random_state,
        diagnostics,
        {"input": "distance_matrix", "distance": "user_supplied"},
    )
