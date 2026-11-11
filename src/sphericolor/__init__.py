"""Spherical geographic color mappings."""

from .api import from_distance_matrix, from_latlon
from .result import ColorMapping, Diagnostics

__all__ = ["ColorMapping", "Diagnostics", "from_distance_matrix", "from_latlon"]
__version__ = "0.1.0"

