from __future__ import annotations

import numpy as np
from cartoboost.representation import (
    EntityEmbedding,
    FuturePatchReconstruction,
    GraphEdgeDenoising,
    HistoricalAnalogRetriever,
    KNNContextMemory,
    MaskedEntityTimeModeling,
    MaskedPairTimeModeling,
    MultiViewSpatialAttention,
    PairEmbedding,
    RegimeRouter,
    RetrievalAugmentedForecaster,
    RetrievalAugmentedPairModel,
    SelfSupervisedPretrainer,
    SpatialNeighborContrastiveLoss,
    SpatioTemporalAdaptiveEmbedding,
    TemporalOrderContrastiveLoss,
)


def test_entity_embedding_supports_unknown_hash_fallback_and_roundtrip(tmp_path):
    model = EntityEmbedding(embedding_dim=4, hash_bucket_count=8, random_seed=3).fit(
        ["region_a", "region_b"],
        training_cutoff="2024-01-01",
        training_metrics={"loss": 0.0},
    )

    known = model.transform(["region_a"])
    unknown_first = model.transform(["never_seen"])
    unknown_second = model.transform(["never_seen"])
    path = tmp_path / "entity.json"
    model.save(path)
    loaded = EntityEmbedding.load(path)

    np.testing.assert_array_equal(unknown_first, unknown_second)
    np.testing.assert_array_equal(loaded.transform(["region_a"]), known)
    np.testing.assert_array_equal(loaded.transform(["never_seen"]), unknown_first)
    artifact = loaded.artifact_metadata()
    assert artifact["model_class"] == "EntityEmbedding"
    assert artifact["artifact_version"] == 1
    assert artifact["id_maps"]["entity"]["__unknown__"] == 0
    assert artifact["hash_bucket_config"]["entity"] == 8
    assert artifact["embedding_dim"] == 4
    assert artifact["random_seed"] == 3
    assert artifact["training_cutoff"] == "2024-01-01"
    assert artifact["training_metrics"]["loss"] == 0.0
    assert artifact["save_load_parity_checked"] is True
    assert artifact["backend"]["selected"] == "cpu"
    assert artifact["backend"]["accelerator_ready"] == {
        "cuda": True,
        "rocm": True,
        "mlx": True,
    }


def test_pair_embedding_preserves_direction_and_unseen_pair_fallback(tmp_path):
    model = PairEmbedding(embedding_dim=3, pair_hash_bucket_count=12, random_seed=5).fit(
        ["A", "B"],
        ["B", "A"],
        training_cutoff="2024-02-01",
    )

    forward = model.transform(["A"], ["B"])
    reverse = model.transform(["B"], ["A"])
    unseen = model.transform(["A"], ["Z"])
    path = tmp_path / "pair.json"
    model.save(path)
    loaded = PairEmbedding.load(path)

    assert not np.array_equal(forward, reverse)
    np.testing.assert_array_equal(loaded.transform(["A"], ["B"]), forward)
    np.testing.assert_array_equal(loaded.transform(["A"], ["Z"]), unseen)
    artifact = loaded.artifact_metadata()
    assert artifact["model_class"] == "PairEmbedding"
    assert artifact["id_maps"]["pair"]["A\u0000B"] != artifact["id_maps"]["pair"]["B\u0000A"]
    assert artifact["hash_bucket_config"]["pair"] == 12
    assert artifact["save_load_parity_checked"] is True


def test_spatiotemporal_adaptive_embedding_changes_with_time_and_context(tmp_path):
    model = SpatioTemporalAdaptiveEmbedding(embedding_dim=4, random_seed=9, backend="auto").fit(
        ["node_1"], training_cutoff="2024-04-01"
    )

    early = model.transform(
        ["node_1"],
        time_features=[[0.0, 0.0]],
        context_features=[[1.0, 0.0]],
    )
    late = model.transform(
        ["node_1"],
        time_features=[[1.0, 1.0]],
        context_features=[[0.0, 1.0]],
    )

    assert set(early) == {
        "adaptive_embedding",
        "static_embedding",
        "temporal_embedding",
        "interaction_embedding",
    }
    np.testing.assert_array_equal(early["static_embedding"], late["static_embedding"])
    assert not np.allclose(early["adaptive_embedding"], late["adaptive_embedding"])
    path = tmp_path / "adaptive.json"
    model.save(path)
    loaded = SpatioTemporalAdaptiveEmbedding.load(path)
    loaded_early = loaded.transform(
        ["node_1"],
        time_features=[[0.0, 0.0]],
        context_features=[[1.0, 0.0]],
    )
    np.testing.assert_array_equal(loaded_early["adaptive_embedding"], early["adaptive_embedding"])
    artifact = loaded.artifact_metadata()
    assert artifact["model_class"] == "SpatioTemporalAdaptiveEmbedding"
    assert artifact["architecture"] == "spatiotemporal_adaptive_embedding"
    assert artifact["training_cutoff"] == "2024-04-01"
    assert artifact["save_load_parity_checked"] is True


def test_spatiotemporal_time_ablation_worsens_rolling_origin_validation():
    model = SpatioTemporalAdaptiveEmbedding(embedding_dim=6, random_seed=21).fit(["entity_a"])
    time_values = np.linspace(0.0, 1.0, 24)
    seasonal = np.column_stack([time_values, time_values**2, np.sin(time_values * np.pi)])
    y = 2.0 + 3.0 * time_values + 0.8 * np.sin(time_values * np.pi)

    report = model.rolling_origin_time_ablation_report(
        ["entity_a"] * len(time_values),
        y,
        time_features=seasonal,
        min_train_size=16,
    )

    assert report["validation_strategy"] == "rolling_origin_holdout"
    assert report["time_features_help"] is True
    assert report["adaptive_time_rmse"] < report["no_time_rmse"]


def test_multi_view_spatial_attention_reports_ablation_and_missing_views(tmp_path):
    views = {
        "physical_distance": [[1.0, 0.2], [0.8, 0.1], [0.1, 1.0]],
        "observed_flow": [[0.0, 2.0], [0.2, 1.8], [1.5, 0.3]],
        "hub_centrality": [[3.0], [1.0], [0.5]],
    }
    model = MultiViewSpatialAttention(embedding_dim=4, random_seed=23).fit(
        ["pickup", "dropoff", "hub"],
        views,
    )

    output = model.transform(views)
    missing = model.transform(
        {
            "physical_distance": views["physical_distance"],
            "observed_flow": views["observed_flow"],
        }
    )
    report = model.view_ablation_report(views)
    path = tmp_path / "multi-view.json"
    model.save(path)
    loaded = MultiViewSpatialAttention.load(path)

    assert output["embedding"].shape == (3, 4)
    assert output["view_weights"].shape == (3, 3)
    np.testing.assert_allclose(output["view_weights"].sum(axis=1), [1.0, 1.0, 1.0])
    assert missing["missing_views"] == ["hub_centrality"]
    missing_idx = model.view_names_.index("hub_centrality")
    assert np.all(missing["view_weights"][:, missing_idx] == 0.0)
    assert report["full_beats_best_single_view"] is True
    assert set(report["single_view_proxy_scores"]) == set(views)
    np.testing.assert_array_equal(loaded.transform(views)["embedding"], output["embedding"])
    artifact = loaded.artifact_metadata()
    assert artifact["model_class"] == "MultiViewSpatialAttention"
    assert artifact["architecture"] == "multi_view_spatial_attention"
    assert artifact["hash_bucket_config"]["view_count"] == 3
    assert set(artifact["learned_view_weights"]) == set(views)
    assert artifact["save_load_parity_checked"] is True


def test_multi_view_spatial_attention_maintained_graph_benchmark_beats_single_view():
    views = {
        "physical_distance": [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        "observed_flow": [
            [0.0, 2.0],
            [0.3, 1.7],
            [1.0, 1.0],
            [1.7, 0.3],
            [2.0, 0.0],
            [1.8, 0.2],
        ],
        "hub_centrality": [[2.5], [1.8], [1.0], [0.8], [0.3], [0.5]],
    }
    model = MultiViewSpatialAttention(embedding_dim=4, random_seed=29).fit(
        [f"node_{idx}" for idx in range(6)],
        views,
    )
    embedding = model.transform(views)["embedding"]
    target = embedding[:, 0] + 0.5 * embedding[:, 1] - 0.25 * embedding[:, 2]

    benchmark = model.maintained_graph_benchmark(views, target, train_size=4)

    assert benchmark["benchmark"] == "maintained_multi_view_graph_holdout"
    assert benchmark["multi_view_beats_best_single_view"] is True
    assert benchmark["multi_view_rmse"] < benchmark["best_single_view_rmse"]
    assert benchmark["improvement"] > 0.0


def test_regime_router_reports_entropy_usage_and_roundtrips(tmp_path):
    model = RegimeRouter(expert_count=3, embedding_dim=4, random_seed=11).fit(
        ["entity_a", "entity_b", "entity_c", "entity_a"],
        context_features=[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        training_cutoff="2024-03-01",
    )

    route = model.route(
        ["entity_a", "entity_z"],
        context_features=[
            [0.5, 0.0],
            [0.0, 0.5],
        ],
    )
    path = tmp_path / "router.json"
    model.save(path)
    loaded = RegimeRouter.load(path)
    loaded_weights = loaded.predict_proba(
        ["entity_a", "entity_z"],
        context_features=[
            [0.5, 0.0],
            [0.0, 0.5],
        ],
    )

    assert route["expert_weights"].shape == (2, 3)
    np.testing.assert_allclose(route["expert_weights"].sum(axis=1), [1.0, 1.0])
    assert route["router_entropy"].shape == (2,)
    np.testing.assert_array_equal(loaded_weights, route["expert_weights"])
    artifact = loaded.artifact_metadata()
    assert artifact["model_class"] == "RegimeRouter"
    assert artifact["architecture"] == "regime_router"
    assert artifact["hash_bucket_config"]["expert_count"] == 3
    assert artifact["training_cutoff"] == "2024-03-01"
    assert artifact["save_load_parity_checked"] is True
    assert "mean_router_entropy" in artifact["training_metrics"]
    assert sum(artifact["expert_usage"].values()) == 1.0


def test_historical_analog_retriever_returns_explainable_cutoff_safe_neighbors(tmp_path):
    retriever = HistoricalAnalogRetriever().fit(
        ["early_a", "early_b", "future_c"],
        [[0.0, 0.0], [1.0, 1.0], [0.1, 0.0]],
        timestamps=["2024-01-01", "2024-01-02", "2024-02-01"],
        training_cutoff="2024-02-15",
    )

    unrestricted = retriever.query([[0.09, 0.0]], k=2)
    cutoff = retriever.query([[0.09, 0.0]], k=2, cutoff="2024-01-15")
    path = tmp_path / "retriever.json"
    retriever.save(path)
    loaded = HistoricalAnalogRetriever.load(path)

    assert unrestricted[0]["analog_ids"][0] == "future_c"
    assert "future_c" not in cutoff[0]["analog_ids"]
    assert cutoff[0]["analog_ids"][0] == "early_a"
    assert cutoff[0]["distances"][0] <= cutoff[0]["distances"][1]
    assert loaded.query([[0.09, 0.0]], k=2, cutoff="2024-01-15") == cutoff
    artifact = loaded.artifact_metadata()
    assert artifact["model_class"] == "HistoricalAnalogRetriever"
    assert artifact["architecture"] == "exact_knn_memory"
    assert artifact["hash_bucket_config"]["memory_size"] == 3
    assert artifact["training_metrics"]["memory_size"] == 3.0
    assert artifact["save_load_parity_checked"] is True


def test_historical_analog_retriever_supports_ann_compression_and_learned_projection(tmp_path):
    retriever = HistoricalAnalogRetriever(
        approximate=True,
        compressed=True,
        learned_projection_dim=2,
        random_seed=5,
    ).fit(
        ["a", "b", "c", "d"],
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [4.0, 4.0, 0.0], [4.1, 4.0, 0.0]],
        timestamps=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
    )

    result = retriever.query([[0.05, 0.0, 1.0]], k=2)
    path = tmp_path / "ann-retriever.json"
    retriever.save(path)
    loaded = HistoricalAnalogRetriever.load(path)
    loaded_result = loaded.query([[0.05, 0.0, 1.0]], k=2)

    assert result[0]["index_kind"] == "approximate_bucket"
    assert result[0]["compressed"] is True
    assert result[0]["learned_projection"] is True
    assert result[0]["analog_ids"][0] in {"a", "b"}
    assert loaded_result == result
    artifact = loaded.artifact_metadata()
    assert artifact["hash_bucket_config"]["ann_bucket_count"] > 0
    assert artifact["hash_bucket_config"]["compressed"] == 1
    assert artifact["hash_bucket_config"]["learned_projection_dim"] == 2


def test_retrieval_augmented_forecaster_improves_rare_pattern_and_roundtrips(tmp_path):
    model = RetrievalAugmentedForecaster(k=2).fit(
        ["common_a", "common_b", "rare_a", "rare_b", "future_rare"],
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [5.05, 5.0]],
        [1.0, 1.2, 10.0, 10.2, 99.0],
        timestamps=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-03-01"],
        training_cutoff="2024-02-01",
    )

    explained = model.predict([[5.05, 5.0]], cutoff="2024-02-01", return_explanation=True)
    benchmark = model.rare_pattern_benchmark([[5.05, 5.0]], [10.1], cutoff="2024-02-01")
    path = tmp_path / "retrieval_forecaster.json"
    model.save(path)
    loaded = RetrievalAugmentedForecaster.load(path)
    loaded_explained = loaded.predict([[5.05, 5.0]], cutoff="2024-02-01", return_explanation=True)

    assert explained["retrieval"][0]["analog_ids"][0] in {"rare_a", "rare_b"}
    assert "future_rare" not in explained["retrieval"][0]["analog_ids"]
    assert explained["retrieval"][0]["distances"][0] <= explained["retrieval"][0]["distances"][1]
    assert benchmark["retrieval_rmse"] < benchmark["global_mean_rmse"]
    assert benchmark["improvement"] > 0.0
    np.testing.assert_array_equal(loaded_explained["prediction"], explained["prediction"])
    assert KNNContextMemory is HistoricalAnalogRetriever


def test_retrieval_augmented_pair_model_uses_directional_pair_memory():
    model = RetrievalAugmentedPairModel(k=1).fit(
        ["A", "B", "A"],
        ["B", "A", "C"],
        [[1.0, 0.0], [5.0, 0.0], [1.1, 0.0]],
        [3.0, 20.0, 3.2],
        timestamps=["2024-01-01", "2024-01-02", "2024-01-03"],
    )
    explained = model.predict([[1.05, 0.0]], return_explanation=True)

    assert explained["retrieval"][0]["analog_ids"][0] in {"A->B", "A->C"}
    assert explained["prediction"][0] < 5.0


def test_self_supervised_pretrainer_respects_cutoff_and_roundtrips(tmp_path):
    pretrain_ids = [f"entity_{idx}" for idx in range(8)] + ["future_entity"]
    pretrain_features = [
        [float(idx), float((idx * idx) % 7), float(np.sin(idx))] for idx in range(8)
    ] + [[100.0, 100.0, 100.0]]
    pretrain_timestamps = [f"2024-01-{idx + 1:02d}" for idx in range(8)] + ["2024-03-01"]
    model = SelfSupervisedPretrainer(embedding_dim=3, random_seed=17).fit(
        pretrain_ids,
        pretrain_features,
        timestamps=pretrain_timestamps,
        training_cutoff="2024-02-01",
    )

    before = model.transform(["entity_0", "future_entity"])
    pair_embeddings = model.pretrained_pair_embeddings()
    node_embeddings = model.pretrained_node_embeddings()
    temporal_encoder = model.pretrained_temporal_encoder()
    benchmark_ids = [f"entity_{idx}" for idx in range(8)]
    benchmark_embedding = model.transform(benchmark_ids)
    benchmark_target = (
        benchmark_embedding[:, 0]
        - 0.5 * benchmark_embedding[:, 1]
        + 0.25 * benchmark_embedding[:, 2]
    )
    benchmark = model.downstream_embedding_benchmark(
        benchmark_ids,
        benchmark_target,
        train_size=4,
        random_seed=101,
    )
    path = tmp_path / "pretrainer.json"
    model.save(path)
    loaded = SelfSupervisedPretrainer.load(path)

    np.testing.assert_array_equal(loaded.transform(["entity_0", "future_entity"]), before)
    np.testing.assert_array_equal(loaded.pretrained_pair_embeddings(), pair_embeddings)
    np.testing.assert_array_equal(loaded.pretrained_node_embeddings(), node_embeddings)
    np.testing.assert_array_equal(loaded.pretrained_temporal_encoder(), temporal_encoder)
    artifact = loaded.artifact_metadata()
    assert artifact["model_class"] == "SelfSupervisedPretrainer"
    assert artifact["architecture"] == "deterministic_self_supervised_pretrainer"
    assert artifact["training_cutoff"] == "2024-02-01"
    assert artifact["training_metrics"]["pretraining_rows"] == 8.0
    assert "masked_pair_proxy_rmse" in artifact["training_metrics"]
    assert "graph_edge_denoising_proxy_auc" in artifact["training_metrics"]
    assert "temporal_order_contrastive_margin" in artifact["training_metrics"]
    assert "spatial_neighbor_contrastive_margin" in artifact["training_metrics"]
    assert "future_patch_reconstruction_proxy_rmse" in artifact["training_metrics"]
    assert artifact["save_load_parity_checked"] is True
    assert artifact["pretraining_tasks"] == [
        "masked_entity_time_modeling",
        "masked_pair_time_modeling",
        "graph_edge_denoising",
        "temporal_order_contrastive_loss",
        "spatial_neighbor_contrastive_loss",
        "future_patch_reconstruction",
    ]
    assert artifact["outputs"] == [
        "pretrained_entity_embeddings",
        "pretrained_pair_embeddings",
        "pretrained_node_embeddings",
        "pretrained_temporal_encoder",
    ]
    assert pair_embeddings.shape == (7, 3)
    assert node_embeddings.shape[1] == 3
    assert temporal_encoder.shape == (6, 3)
    assert benchmark["benchmark"] == "maintained_pretrained_embedding_holdout"
    assert benchmark["supervised_budget"] == 4
    assert benchmark["pretrained_beats_random"] is True
    assert benchmark["pretrained_rmse"] < benchmark["random_embedding_rmse"]
    assert benchmark["improvement"] > 0.0
    assert "future_entity" not in artifact["id_maps"]["entity"]
    assert MaskedEntityTimeModeling.task_name == "masked_entity_time_modeling"
    assert MaskedPairTimeModeling.task_name == "masked_pair_time_modeling"
    assert GraphEdgeDenoising.task_name == "graph_edge_denoising"
    assert TemporalOrderContrastiveLoss.task_name == "temporal_order_contrastive_loss"
    assert SpatialNeighborContrastiveLoss.task_name == "spatial_neighbor_contrastive_loss"
    assert FuturePatchReconstruction.task_name == "future_patch_reconstruction"
