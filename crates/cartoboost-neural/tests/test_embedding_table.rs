use cartoboost_neural::{
    build_embedding_table_artifact, write_embedding_table_artifact, ArtifactFallbackKind,
    EmbeddingIdType, EmbeddingRow, EmbeddingTable, NeuralEncoder,
};
use tempfile::tempdir;

#[test]
fn embedding_checksum_and_predictions_do_not_depend_on_artifact_row_order() {
    let rows = vec![
        EmbeddingRow {
            id: 7,
            values: vec![0.25, -0.75],
        },
        EmbeddingRow {
            id: 3,
            values: vec![-0.5, 0.0],
        },
    ];
    let artifact =
        build_embedding_table_artifact(2, rows.clone(), ArtifactFallbackKind::GlobalMeanVector)
            .unwrap();
    let sorted = build_embedding_table_artifact(
        2,
        rows.into_iter().rev().collect(),
        ArtifactFallbackKind::GlobalMeanVector,
    )
    .unwrap();
    assert_eq!(artifact.metadata.checksum, sorted.metadata.checksum);
    let original = EmbeddingTable::from_artifact(artifact).unwrap();
    let restored = EmbeddingTable::from_artifact(sorted).unwrap();
    assert_eq!(
        original.encode_ids(&[7, 3, 999], "ids").unwrap().values,
        restored.encode_ids(&[7, 3, 999], "ids").unwrap().values,
    );
}

#[test]
fn loads_embedding_table() {
    let rows = vec![
        EmbeddingRow {
            id: 100,
            values: vec![0.1, 0.2],
        },
        EmbeddingRow {
            id: 200,
            values: vec![0.3, 0.4],
        },
    ];
    let artifact =
        build_embedding_table_artifact(2, rows.clone(), ArtifactFallbackKind::GlobalMeanVector)
            .expect("artifact should build");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("table.json");
    write_embedding_table_artifact(&path, &artifact).expect("artifact should save");

    let table = EmbeddingTable::load(path).expect("table should load");
    let metadata = table.artifact_metadata();

    assert_eq!(table.dim(), 2);
    assert_eq!(table.row_count(), 2);
    assert_eq!(table.id_type(), EmbeddingIdType::U64);
    assert!(!metadata.checksum.is_empty());
}

#[test]
fn looks_up_known_id() {
    let rows = vec![
        EmbeddingRow {
            id: 5,
            values: vec![1.0, -1.0],
        },
        EmbeddingRow {
            id: 7,
            values: vec![2.0, 3.0],
        },
    ];
    let artifact = build_embedding_table_artifact(2, rows, ArtifactFallbackKind::ZeroVector)
        .expect("artifact should build");
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("table.json");
    write_embedding_table_artifact(&path, &artifact).expect("artifact should save");

    let table = EmbeddingTable::load(path).expect("table should load");

    assert_eq!(table.lookup(5), Some(&[1.0_f32, -1.0][..]));
    assert_eq!(table.lookup(7), Some(&[2.0_f32, 3.0][..]));
}

#[test]
fn uses_fallback_for_missing_id() {
    let rows = vec![
        EmbeddingRow {
            id: 10,
            values: vec![4.0, 2.0],
        },
        EmbeddingRow {
            id: 11,
            values: vec![0.0, 8.0],
        },
    ];
    let artifact = build_embedding_table_artifact(2, rows, ArtifactFallbackKind::GlobalMeanVector)
        .expect("artifact should build");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("table.json");
    write_embedding_table_artifact(&path, &artifact).expect("artifact should save");

    let table = EmbeddingTable::load(path).expect("table should load");
    let block = table
        .encode_ids(&[10, 999], "neural.cell")
        .expect("encoding should work");

    assert_eq!(block.values[0..2], [4.0_f32, 2.0_f32]);
    assert_eq!(block.values[2..4], [2.0_f32, 5.0_f32]);
}

#[test]
fn feature_names_and_order_are_deterministic() {
    let rows = vec![EmbeddingRow {
        id: 1,
        values: vec![0.5, 1.5, 2.5],
    }];
    let artifact = build_embedding_table_artifact(3, rows, ArtifactFallbackKind::ZeroVector)
        .expect("artifact should build");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("table.json");
    write_embedding_table_artifact(&path, &artifact).expect("artifact should save");

    let table = EmbeddingTable::load(path).expect("table should load");
    let encoder = cartoboost_neural::EmbeddingTableEncoder::new("neural.cell", table);
    let block = encoder.encode_ids(&[1]).expect("encoding should work");

    assert_eq!(
        block.feature_names(),
        vec![
            "neural.cell_00".to_string(),
            "neural.cell_01".to_string(),
            "neural.cell_02".to_string(),
        ]
    );
}

#[test]
fn appends_block_without_changing_row_count() {
    let rows = vec![
        EmbeddingRow {
            id: 1,
            values: vec![0.1, -0.1],
        },
        EmbeddingRow {
            id: 2,
            values: vec![0.2, -0.2],
        },
    ];
    let artifact = build_embedding_table_artifact(2, rows, ArtifactFallbackKind::ZeroVector)
        .expect("artifact should build");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("table.json");
    write_embedding_table_artifact(&path, &artifact).expect("artifact should save");

    let table = EmbeddingTable::load(path).expect("table should load");
    let block = table
        .encode_ids(&[1, 2], "neural.cell")
        .expect("encoding should work");

    let mut dense: Vec<Vec<f64>> = vec![vec![1.0_f64, 2.0], vec![3.0_f64, 4.0]];
    assert_eq!(dense.len(), 2);

    let before = dense.len();
    block
        .append_to_dense_f64(&mut dense)
        .expect("appended dense should preserve row count");

    assert_eq!(dense.len(), before);
    assert!((dense[0][0] - 1.0_f64).abs() < 1e-12);
    assert!((dense[0][1] - 2.0_f64).abs() < 1e-12);
    assert!((dense[0][2] - 0.1_f64).abs() < 1e-6);
    assert!((dense[0][3] - (-0.1_f64)).abs() < 1e-6);

    assert!((dense[1][0] - 3.0_f64).abs() < 1e-12);
    assert!((dense[1][1] - 4.0_f64).abs() < 1e-12);
    assert!((dense[1][2] - 0.2_f64).abs() < 1e-6);
    assert!((dense[1][3] - (-0.2_f64)).abs() < 1e-6);
}
