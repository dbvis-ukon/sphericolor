import numpy as np

from sphericolor import from_distance_matrix
from sphericolor.methods.simplex import apply_transform


def test_transform_rotates_around_z_axis():
    points = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    transformed = apply_transform(points, np.array([0, 0, 0, 0.5, 0], dtype=float))
    assert not np.allclose(transformed, points)
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), [1.0, 1.0])


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
    assert len(result.metadata["transform"]) == 5
    assert np.all((result.rgb >= 0.0) & (result.rgb <= 1.0))
