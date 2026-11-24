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


def test_embedding_ids_preserve_full_u64_range_and_fallbacks():
    ids = [1, 2**63 + 1, 2**64 - 1]
    model = NeuralEmbeddingFeatures(dim=2).fit(ids, [0.0, 1.0, 2.0])

    assert model.known_ids() == set(ids)
    np.testing.assert_array_equal(
        model.transform_with_fallback([7, 8], fallback_ids=[ids[1], ids[2]]),
        model.transform(ids[1:]),
    )
    np.testing.assert_array_equal(
        model.transform_with_fallback([7], neighbor_ids=[[ids[2]]]),
        model.transform([ids[2]]),
    )


@pytest.mark.parametrize("invalid", [-1, 1.5, 2**64, float("nan"), float("inf"), True])
def test_embedding_ids_reject_invalid_integer_values(invalid):
    with pytest.raises(ValueError, match="ids must"):
        NeuralEmbeddingFeatures(dim=2).fit([1, invalid], [0.0, 1.0])


@pytest.mark.parametrize("kind", ["pandas", "polars"])
def test_embedding_integer_tables_preserve_ids_and_pipeline_fallbacks(kind):
    from cartoboost.neural.features import _as_u64_array
    from cartoboost.neural.pipeline import _split_fallback_by_key, _to_u64_id_matrix

    large = [2**64 - 1, 2**63 + 1]
    if kind == "pandas":
        pd = pytest.importorskip("pandas")
        ids = pd.Series(large, dtype="uint64")
        table = pd.DataFrame({"pickup": ids, "dropoff": pd.Series([1, 2], dtype="int64")})
    else:
        pl = pytest.importorskip("polars")
        ids = pl.Series("pickup", large, dtype=pl.UInt64)
        table = pl.DataFrame([ids, pl.Series("dropoff", [1, 2], dtype=pl.Int64)])
    expected = np.array([[large[0], 1], [large[1], 2]], dtype=np.uint64)
    np.testing.assert_array_equal(_as_u64_array(table), expected)
    np.testing.assert_array_equal(_to_u64_id_matrix(table), expected)
    fallbacks = _split_fallback_by_key(table, n_keys=2, row_count=2)
    for index, values in enumerate(fallbacks):
        np.testing.assert_array_equal(_as_u64_array(values), expected[:, index])
    model = NeuralEmbeddingFeatures(dim=2).fit(ids, [0.0, 1.0])
    assert model.known_ids() == set(large)
    np.testing.assert_array_equal(model.transform(ids), model.transform(large))


def test_embedding_nullable_integer_tables_reject_missing_and_invalid_ids():
    pd = pytest.importorskip("pandas")
    from cartoboost.neural.features import _as_u64_array

    np.testing.assert_array_equal(
        _as_u64_array(pd.Series([2**64 - 1, 1], dtype="UInt64")),
        np.array([2**64 - 1, 1], dtype=np.uint64),
    )
    for values in [
        pd.Series([2**64 - 1, None], dtype="UInt64"),
        pd.Series([1, -1], dtype="Int64"),
        pd.Series([1.0, 1.5], dtype="Float64"),
        pd.Series([True, False], dtype="boolean"),
    ]:
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            _as_u64_array(values)


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "reversed", "strided"])
def test_embedding_native_arrays_preserve_flat_outputs_and_artifacts(layout, tmp_path):
    from cartoboost import _native

    ids = np.array([2**64 - 1, 2**63 + 1, 2**64 - 1, 1], dtype=np.uint64)
    targets = np.array([1.0, 2.0, 1.5, 3.0])
    if layout == "reversed":
        ids, targets = ids[::-1].copy()[::-1], targets[::-1].copy()[::-1]
    elif layout == "strided":
        ids = np.repeat(ids, 2)[::2]
        targets = np.repeat(targets, 2)[::2]
    elif layout == "fortran":
        ids, targets = np.asfortranarray(ids), np.asfortranarray(targets)
    legacy = _native.NeuralEmbeddingFeatures(3, random_state=11)
    candidate = _native.NeuralEmbeddingFeatures(3, random_state=11)
    expected = legacy.fit_transform(np.ascontiguousarray(ids), np.ascontiguousarray(targets))
    actual = candidate.fit_transform_arrays(ids, targets)
    assert actual.dtype == np.float32 and actual.shape == (4, 3) and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(candidate.transform_arrays(ids), actual)
    np.testing.assert_array_equal(
        candidate.transform_arrays(np.array([7], dtype=np.uint64)),
        legacy.transform(np.array([7], dtype=np.uint64)),
    )
    assert candidate.transform_arrays(np.empty(0, dtype=np.uint64)).shape == (0, 3)
    old_path, new_path = tmp_path / "legacy.json", tmp_path / "arrays.json"
    legacy.export(str(old_path))
    candidate.export(str(new_path))
    assert old_path.read_bytes() == new_path.read_bytes()
    loaded = _native.NeuralEmbeddingFeatures.from_artifact(str(new_path))
    np.testing.assert_array_equal(loaded.transform_arrays(ids), actual)
