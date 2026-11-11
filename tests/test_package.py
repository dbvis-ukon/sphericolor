from importlib.resources import files

import numpy as np

import sphericolor


def test_package_data_available():
    resource = files("sphericolor.data").joinpath("srgb_cam16ucs.ply")
    assert resource.is_file()


def test_default_diagnostics_disabled():
    result = sphericolor.from_latlon([0.0, 10.0], [0.0, 10.0], method="hsluv")
    assert result.diagnostics is None
    assert result.cam16ucs.shape == (2, 3)
    assert np.all(np.isfinite(result.cam16ucs))

