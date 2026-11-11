import numpy as np
import pytest

from sphericolor import from_latlon


def test_hsluv_wraps_at_antimeridian():
    result = from_latlon([0.0, 0.0], [-180.0, 180.0], method="hsluv")
    np.testing.assert_allclose(result.rgb[0], result.rgb[1], atol=1e-12)


@pytest.mark.parametrize(
    "name", ["bremm", "cube_diagonal", "schumann", "steiger", "teuling2", "ziegler"]
)
def test_all_bivariate_maps(name):
    result = from_latlon(
        [-45.0, 0.0, 45.0],
        [-120.0, 0.0, 120.0],
        method="bivariate",
        bivariate_map=name,
    )
    assert result.rgb.shape == (3, 3)
    assert result.metadata["bivariate_map"] == name
    assert np.all((result.rgb >= 0.0) & (result.rgb <= 1.0))


def test_invalid_bivariate_name_rejected():
    with pytest.raises(ValueError, match="bivariate_map"):
        from_latlon([0.0], [0.0], method="bivariate", bivariate_map="unknown")


def test_baseline_diagnostics_and_hex_consistency():
    result = from_latlon(
        [-20.0, 20.0], [-30.0, 30.0], method="hsluv", diagnostics=True
    )
    assert result.diagnostics is not None
    assert result.diagnostics.embedding_pre is None
    expected = [
        f"#{red:02x}{green:02x}{blue:02x}"
        for red, green, blue in np.rint(result.rgb * 255.0).astype(np.uint8)
    ]
    assert result.hex.tolist() == expected

