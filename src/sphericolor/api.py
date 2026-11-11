"""Public generation API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist, squareform

from .colors import cam16ucs_to_rgb, rgb_to_cam16ucs, rgb_to_hex
from .diagnostics import build_diagnostics
from .distances import great_circle_matrix
from .embedding import metric_mds
from .gamut import srgb_hull
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
    mds_seed_trials: int,
    diagnostics: bool,
    metadata: dict[str, Any],
) -> ColorMapping:
    seeds = _mds_seed_candidates(random_state, mds_seed_trials)
    input_distances = squareform(distances, checks=False)
    gamut_volume = float(srgb_hull().volume)
    seed_scores: list[dict[str, float | int | None]] = []
    selected: tuple[
        float,
        int | None,
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        int,
        dict[str, Any],
    ] | None = None

    for seed in seeds:
        candidate_embedding = metric_mds(distances, seed)
        if method == "ray":
            candidate_mapped, candidate_metadata = project_directions_to_gamut(
                candidate_embedding
            )
        else:
            candidate_mapped, candidate_metadata = fit_embedding_to_gamut(
                candidate_embedding, distances
            )
        candidate_rgb, candidate_clipped = cam16ucs_to_rgb(candidate_mapped)
        candidate_displayed = rgb_to_cam16ucs(candidate_rgb)
        preservation, exploitation, score = _mds_seed_score(
            input_distances, candidate_displayed, gamut_volume
        )
        seed_scores.append(
            {
                "seed": seed,
                "distance_preservation": preservation,
                "colorspace_exploitation": exploitation,
                "score": score,
            }
        )
        if selected is None or score > selected[0]:
            selected = (
                score,
                seed,
                candidate_embedding,
                candidate_mapped,
                candidate_rgb,
                candidate_clipped,
                candidate_metadata,
            )

    if selected is None:
        raise RuntimeError("MDS seed selection produced no candidates")
    _, selected_seed, embedding, mapped, rgb, clipped, method_metadata = selected
    displayed_cam16ucs = rgb_to_cam16ucs(rgb)
    metadata.update(method_metadata)
    metadata.update(
        {
            "clipped_color_count": clipped,
            "random_state": random_state,
            "mds_seed_trials": mds_seed_trials,
            "selected_mds_seed": selected_seed,
            "mds_seed_scores": tuple(seed_scores),
        }
    )
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


def _mds_seed_candidates(
    random_state: int | None, trials: int
) -> tuple[int | None, ...]:
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("mds_seed_trials must be a positive integer")
    if trials == 1:
        return (random_state,)

    seeds = [69]
    if random_state is not None and random_state != 69:
        seeds.append(random_state)
    generator = np.random.default_rng(random_state)
    while len(seeds) < trials:
        seed = int(generator.integers(0, 2**32 - 1))
        if seed not in seeds:
            seeds.append(seed)
    return tuple(seeds[:trials])


def _mds_seed_score(
    input_distances: NDArray[np.float64],
    displayed_cam16ucs: NDArray[np.float64],
    gamut_volume: float,
) -> tuple[float, float, float]:
    color_distances = 1.41 * np.power(pdist(displayed_cam16ucs), 0.63)
    if np.std(input_distances) <= 1e-12 or np.std(color_distances) <= 1e-12:
        preservation = 0.0
    else:
        correlation = float(np.corrcoef(input_distances, color_distances)[0, 1])
        preservation = float(np.clip(correlation, 0.0, 1.0))

    if displayed_cam16ucs.shape[0] < 4:
        return preservation, 0.0, preservation
    try:
        exploitation = float(
            np.clip(ConvexHull(displayed_cam16ucs).volume / gamut_volume, 0.0, 1.0)
        )
    except QhullError:
        return preservation, 0.0, preservation
    return preservation, exploitation, 0.5 * (preservation + exploitation)


def from_latlon(
    latitudes: ArrayLike,
    longitudes: ArrayLike,
    *,
    labels: Sequence[Any] | None = None,
    method: str = "ray",
    random_state: int | None = 69,
    mds_seed_trials: int = 1,
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
            mds_seed_trials,
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
    mds_seed_trials: int = 1,
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
        mds_seed_trials,
        diagnostics,
        {"input": "distance_matrix", "distance": "user_supplied"},
    )
