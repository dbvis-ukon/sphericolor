import numpy as np

from sphericolor import from_distance_matrix
from sphericolor.methods.simplex import apply_transform


def test_transform_rotates_around_all_axes():
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    around_x = apply_transform(points, np.array([0, 0, 0, 0.5, 0, 0, 0], dtype=float))
    around_y = apply_transform(points, np.array([0, 0, 0, 0, 0.5, 0, 0], dtype=float))
    around_z = apply_transform(points, np.array([0, 0, 0, 0, 0, 0.5, 0], dtype=float))
    np.testing.assert_allclose(around_x, points)
    assert not np.allclose(around_y, points)
    assert not np.allclose(around_z, points)


def test_simplex_returns_valid_colors_and_metadata():
    distances = np.array(
        [
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    result = from_distance_matrix(distances, method="simplex")
    assert result.rgb.shape == (4, 3)
    assert isinstance(result.metadata["converged"], bool)
    assert result.metadata["iterations"] > 0
    assert len(result.metadata["transform"]) == 7
    assert np.all((result.rgb >= 0.0) & (result.rgb <= 1.0))

