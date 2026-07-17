import numpy as np
import pytest
from cartoboost.deep import available_deep_backends
from cartoboost.neural import NeuralEmbeddingFeatures


def test_train_export_load_roundtrip(tmp_path):
    ids = np.array([10, 10, 20, 30, 30], dtype=np.uint64)
    residual = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

    transformer = NeuralEmbeddingFeatures(dim=4, random_state=7).fit(ids, residual)
    artifact_path = tmp_path / "embeddings.json"
    transformer.export(artifact_path)

    loaded = NeuralEmbeddingFeatures.from_artifact(artifact_path)

    assert loaded.dim == transformer.dim
    assert np.allclose(loaded.transform([10, 20, 30, 99]), transformer.transform([10, 20, 30, 99]))


def test_transformed_features_can_concat_to_dense_input(tmp_path):
    ids = np.array([1, 2, 3], dtype=np.uint64)
    residual = np.array([10.0, 12.0, 14.0], dtype=np.float64)

    transformer = NeuralEmbeddingFeatures(dim=3, fallback="zero_vector").fit(ids, residual)
    neural_features = transformer.transform(ids)

    x = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.5],
            [2.0, 2.5],
        ],
        dtype=np.float64,
    )
    x_aug = np.hstack([x, neural_features])

    assert x_aug.shape == (3, 5)
    assert np.allclose(x_aug[:, 0:2], x)


def test_transform_with_hierarchical_and_neighbor_fallback():
    ids = np.array([1, 2, 10, 20], dtype=np.uint64)
    residual = np.array([1.0, 1.2, 5.0, 7.0], dtype=np.float64)

    transformer = NeuralEmbeddingFeatures(dim=2, random_state=3).fit(ids, residual)
    direct = transformer.transform([1, 10])
    fallback = transformer.transform_with_fallback(
        [99, 98, 97],
        fallback_ids=[[1, 2], [42, 10], [42, 43]],
        neighbor_ids=[[10, 20], [], [2]],
    )

    assert np.allclose(fallback[0], np.mean(transformer.transform([10, 20]), axis=0))
    assert np.allclose(fallback[1], direct[1])
    assert np.allclose(fallback[2], transformer.transform([2])[0])


def test_support_prior_strength_controls_rare_id_shrinkage():
    ids = np.array([1, 2, 2, 2, 2], dtype=np.uint64)
    residual = np.array([20.0, -1.0, -1.2, -0.8, -1.1], dtype=np.float64)

    weak_prior = NeuralEmbeddingFeatures(
        dim=3,
        random_state=4,
        support_prior_strength=0.25,
    ).fit(ids, residual)
    strong_prior = NeuralEmbeddingFeatures(
        dim=3,
        random_state=4,
        support_prior_strength=12.0,
    ).fit(ids, residual)

    weak_norm = float(np.linalg.norm(weak_prior.transform([1])[0]))
    strong_norm = float(np.linalg.norm(strong_prior.transform([1])[0]))

    assert strong_norm < weak_norm


@pytest.mark.parametrize("backend", available_deep_backends())
def test_embedding_features_support_every_available_backend(backend):
    ids = np.array([1, 1, 2, 2, 3, 3], dtype=np.uint64)
    target = np.array([1.0, 1.2, 2.0, 2.2, 3.0, 3.2], dtype=np.float64)
    expected = NeuralEmbeddingFeatures(dim=4, random_state=9).fit_transform(ids, target)
    actual = NeuralEmbeddingFeatures(dim=4, random_state=9, backend=backend).fit_transform(
        ids, target
    )
    assert np.allclose(actual, expected, rtol=2e-3, atol=2e-3)
