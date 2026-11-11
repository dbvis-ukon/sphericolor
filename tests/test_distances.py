import numpy as np

from sphericolor.distances import EARTH_RADIUS_KM, great_circle_matrix


def test_quarter_circumference():
    distances = great_circle_matrix(np.array([0.0, 0.0]), np.array([0.0, 90.0]))
    np.testing.assert_allclose(distances[0, 1], np.pi * EARTH_RADIUS_KM / 2.0, rtol=1e-12)


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    result = great_circle_matrix(
        np.array([-45.0, 0.0, 45.0]), np.array([-120.0, 0.0, 120.0])
    )
    np.testing.assert_allclose(result, result.T)
    np.testing.assert_array_equal(np.diag(result), np.zeros(3))
