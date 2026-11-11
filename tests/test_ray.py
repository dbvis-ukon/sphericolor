import numpy as np

from sphericolor import from_distance_matrix
from sphericolor.gamut import gamut_centroid, srgb_hull
from sphericolor.methods.ray import project_directions_to_gamut


def test_projection_preserves_centered_directions_and_enters_hull():
    embedding = np.array(
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-1.0, -1.0, 2.0], [-1.0, -2.0, -2.0]]
    )
    projected, metadata = project_directions_to_gamut(embedding, chunk_size=2)
    centroid = gamut_centroid()
    source = embedding - embedding.mean(axis=0)
    target = projected - centroid
    source /= np.linalg.norm(source, axis=1, keepdims=True)
    target /= np.linalg.norm(target, axis=1, keepdims=True)
    np.testing.assert_allclose(source, target, atol=1e-10)
    signed = projected @ srgb_hull().equations[:, :3].T + srgb_hull().equations[:, 3]
    assert np.max(signed) < 1e-8
    assert metadata["central_point_count"] == 0


def test_centroid_point_maps_to_gamut_centroid():
    embedding = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    projected, metadata = project_directions_to_gamut(embedding)
    np.testing.assert_allclose(projected[0], gamut_centroid())
    assert metadata["central_point_count"] == 1


def test_ray_api_is_deterministic_and_read_only():
    distances = np.array(
        [
            [0.0, 1.0, 2.0, 1.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
    )
    first = from_distance_matrix(distances, labels=list("abcd"), diagnostics=True)
    second = from_distance_matrix(distances, labels=list("abcd"), diagnostics=True)
    np.testing.assert_allclose(first.rgb, second.rgb)
    assert first.labels == tuple("abcd")
    assert first.diagnostics is not None
    assert first.diagnostics.input_distances.shape == (6,)
    assert not first.rgb.flags.writeable
    assert np.all((first.rgb >= 0.0) & (first.rgb <= 1.0))

