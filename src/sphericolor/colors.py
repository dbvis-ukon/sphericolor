"""Color-space conversions and serialization."""

from __future__ import annotations

import numpy as np
from coloraide import Color as BaseColor
from coloraide.distance.delta_e_cam16 import DECAM16
from coloraide.spaces.cam16_ucs import CAM16LCD, CAM16SCD, CAM16UCS, CAM16JMh  # type: ignore[attr-defined]
from numpy.typing import ArrayLike, NDArray


class Color(BaseColor):
    """Package-local ColorAide registry."""


Color.register([CAM16UCS(), CAM16SCD(), CAM16LCD(), DECAM16(), CAM16JMh()])


def rgb_to_cam16ucs(rgb: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(rgb, dtype=np.float64)
    output = np.empty_like(values)
    for index, color in enumerate(values):
        converted = Color("srgb", color.tolist()).convert("cam16-ucs")
        output[index] = converted.coords()
    return output


def cam16ucs_to_rgb(
    cam16ucs: ArrayLike,
) -> tuple[NDArray[np.float64], int]:
    values = np.asarray(cam16ucs, dtype=np.float64)
    output = np.empty_like(values)
    clipped = 0
    for index, color in enumerate(values):
        converted = np.asarray(
            Color("cam16-ucs", color.tolist()).convert("srgb").coords(), dtype=np.float64
        )
        clipped += int(np.any((converted < 0.0) | (converted > 1.0)))
        output[index] = np.clip(converted, 0.0, 1.0)
    return output, clipped


def rgb_to_hex(rgb: ArrayLike) -> NDArray[np.str_]:
    values = np.rint(np.asarray(rgb, dtype=np.float64) * 255.0).astype(np.uint8)
    return np.asarray(
        [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in values], dtype=np.str_
    )
