"""Metric multidimensional scaling."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.manifold import MDS


def metric_mds(
    distances: NDArray[np.float64], random_state: int | None
) -> NDArray[np.float64]:
    options: dict[str, Any] = {
        "n_components": 3,
        "normalized_stress": "auto",
        "random_state": random_state,
        "n_init": 1,
        "n_jobs": 1,
    }
    if "metric_mds" in inspect.signature(MDS).parameters:
        options.update(metric_mds=True, metric="precomputed", init="random")
    else:
        options.update(metric=True, dissimilarity="precomputed")
    model = MDS(**options)
    result = np.asarray(model.fit_transform(distances), dtype=np.float64)
    if result.shape != (distances.shape[0], 3) or not np.all(np.isfinite(result)):
        raise RuntimeError("MDS produced an invalid embedding")
    return result
