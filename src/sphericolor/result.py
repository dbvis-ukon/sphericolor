"""Public result types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]
Scalar = TypeVar("Scalar", bound=np.generic)


@dataclass(frozen=True)
class Diagnostics:
    """Optional condensed distances and intermediate embeddings."""

    input_distances: FloatArray
    cam16ucs_distances: FloatArray
    embedding_pre: FloatArray | None = None
    embedding_post: FloatArray | None = None


@dataclass(frozen=True)
class ColorMapping:
    """Colors generated for an ordered, fixed set of entities."""

    labels: tuple[Any, ...]
    rgb: FloatArray
    hex: StringArray
    cam16ucs: FloatArray
    method: str
    metadata: Mapping[str, Any]
    diagnostics: Diagnostics | None = None


def readonly(array: NDArray[Scalar]) -> NDArray[Scalar]:
    """Return a contiguous, read-only copy."""
    result = cast(NDArray[Scalar], np.ascontiguousarray(array).copy())
    result.setflags(write=False)
    return result


def immutable_metadata(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Protect top-level metadata from mutation."""
    return MappingProxyType(dict(values))
