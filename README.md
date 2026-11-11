# SpheriColor

[![PyPI](https://img.shields.io/pypi/v/sphericolor.svg)](https://pypi.org/project/sphericolor/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SpheriColor generates dataset-specific colors for spherical geographic references. It maps
geographic or custom pairwise distances into CAM16-UCS and the displayable sRGB gamut.

![Comparison of SpheriColor and baseline colormap methods](https://raw.githubusercontent.com/dbvis-ukon/sphericolor/main/docs/teaser.png)

## Installation

```bash
pip install sphericolor
```

SpheriColor supports Python 3.10 through 3.13. It needs no Rust compiler or Numba installation.

## Geographic coordinates

```python
from sphericolor import from_latlon

mapping = from_latlon(
    latitudes=[52.52, 48.86, 41.90, 40.42],
    longitudes=[13.40, 2.35, 12.50, -3.70],
    labels=["Berlin", "Paris", "Rome", "Madrid"],
    method="ray",
)

print(dict(zip(mapping.labels, mapping.hex)))
```

Coordinate input uses great-circle distances in kilometers for Ray and Simplex.

## Precomputed distances

```python
import numpy as np
from sphericolor import from_distance_matrix

distances = np.array([
    [0.0, 1.0, 2.0, 1.0],
    [1.0, 0.0, 1.0, 2.0],
    [2.0, 1.0, 0.0, 1.0],
    [1.0, 2.0, 1.0, 0.0],
])
mapping = from_distance_matrix(distances, method="simplex")
```

Distance-matrix input supports Ray and Simplex. HSLuv and bivariate mappings require geographic
coordinates because they use latitude and longitude orientation directly.

## Methods

```python
ray = from_latlon(latitudes, longitudes, method="ray")
simplex = from_latlon(latitudes, longitudes, method="simplex")
hsluv = from_latlon(latitudes, longitudes, method="hsluv")
bivariate = from_latlon(
    latitudes,
    longitudes,
    method="bivariate",
    bivariate_map="steiger",
)
```

Available bivariate maps are `bremm`, `cube_diagonal`, `schumann`, `steiger`, `teuling2`, and
`ziegler`.

Set `diagnostics=True` to include condensed input and output distances plus intermediate MDS
embeddings. Diagnostics require quadratic memory and are disabled by default.

### Stability

Ray and Simplex derive their colors from an MDS embedding of the complete input set. Because MDS
is sensitive to the point configuration, changing the entity set may produce a different color
mapping. HSLuv and the bivariate methods provide dataset-independent alternatives when consistent
colors across datasets are needed.

## License and citation

SpheriColor and its bundled gamut asset are MIT licensed. If you use SpheriColor in academic
work, please cite the associated paper:

> Julius Rauscher, Johannes Fuchs, Daniel A. Keim, and Frederik L. Dennig. “SpheriColor:
> Colormaps for Spherical Geospatial Input Topographies.” EuroVis 2026 Short Papers, 2026.

```bibtex
@inproceedings{rauscher2026sphericolor,
  title     = {{SpheriColor}: Colormaps for Spherical Geospatial Input Topographies},
  author    = {Rauscher, Julius and Fuchs, Johannes and Keim, Daniel A. and Dennig, Frederik L.},
  booktitle = {EuroVis 2026 Short Papers},
  year      = {2026}
}
```

See `CITATION.cff` for machine-readable software and paper citation metadata.
