import numpy as np
import pytest

from sphericolor import from_distance_matrix, from_latlon


@pytest.mark.parametrize(
    ("latitudes", "longitudes", "message"),
    [
        ([[0.0]], [0.0], "one-dimensional"),
        ([0.0], [0.0, 1.0], "equal lengths"),
        ([91.0], [0.0], "latitudes"),
        ([0.0], [181.0], "longitudes"),
        ([np.nan], [0.0], "finite"),
    ],
)
def test_coordinate_validation(latitudes, longitudes, message):
    with pytest.raises(ValueError, match=message):
        from_latlon(latitudes, longitudes)


def test_labels_must_match_and_be_unique():
    with pytest.raises(ValueError, match="same length"):
        from_latlon([0.0, 1.0], [0.0, 1.0], labels=["one"])
    with pytest.raises(ValueError, match="unique"):
        from_latlon([0.0, 1.0], [0.0, 1.0], labels=["same", "same"])


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[0.0, 1.0]], "square"),
        ([[0.0]], "at least two"),
        ([[0.0, -1.0], [-1.0, 0.0]], "nonnegative"),
        ([[0.0, 1.0], [2.0, 0.0]], "symmetric"),
        ([[1.0, 2.0], [2.0, 1.0]], "zero diagonal"),
        ([[0.0, 0.0], [0.0, 0.0]], "entirely zero"),
    ],
)
def test_distance_matrix_validation(matrix, message):
    with pytest.raises(ValueError, match=message):
        from_distance_matrix(matrix)


def test_distance_matrix_rejects_coordinate_only_methods():
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="only ray and simplex"):
        from_distance_matrix(distances, method="hsluv")


def test_unknown_method_rejected():
    with pytest.raises(ValueError, match="method must be"):
        from_latlon([0.0, 1.0], [0.0, 1.0], method="unknown")

