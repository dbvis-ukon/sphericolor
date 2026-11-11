"""Optional diagnostic construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from .result import Diagnostics, readonly


def build_diagnostics(
    distances: NDArray[np.float64],
    cam16ucs: NDArray[np.float64],
    embedding_pre: NDArray[np.float64] | None,
    embedding_post: NDArray[np.float64] | None,
) -> Diagnostics:
    return Diagnostics(
        input_distances=readonly(squareform(distances, checks=False)),
        cam16ucs_distances=readonly(pdist(cam16ucs)),
        embedding_pre=None if embedding_pre is None else readonly(embedding_pre),
        embedding_post=None if embedding_post is None else readonly(embedding_post),
    )

