#![no_main]

use cartoboost_core::data::{FeatureKind, FeatureSchema, SparseSetColumn};
use cartoboost_core::tree::{FuzzyKernel, Node, Split, Tree, MODEL_ARTIFACT_VERSION};
use cartoboost_core::{Dataset, Model};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 24 {
        return;
    }
    let rows = 2 + (data[0] as usize % 6);
    let cols = 1 + (data[1] as usize % 3);
    let mut cursor = 2;
    let needed = 2 + rows * cols + rows * 4 + 1 + 18 + 5;
    if data.len() < needed {
        return;
    }

    let mut dense = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push(scaled_byte(data[cursor], 12.0));
            cursor += 1;
        }
        dense.push(row);
    }

    let mut sparse_rows = Vec::with_capacity(rows);
    for _ in 0..rows {
        let id_count = 1 + (data[cursor] as usize % 3);
        cursor += 1;
        let mut ids = Vec::with_capacity(id_count);
        for _ in 0..id_count {
            ids.push(u64::from(data[cursor] % 24));
            cursor += 1;
        }
        sparse_rows.push(ids);
    }

    let mut names = (0..cols)
        .map(|idx| format!("feature_{idx}"))
        .collect::<Vec<_>>();
    let mut kinds = (0..cols)
        .map(|idx| {
            if idx == 0 && data[cursor] % 2 == 0 {
                FeatureKind::Periodic { period: 24 }
            } else {
                FeatureKind::Numeric
            }
        })
        .collect::<Vec<_>>();
    cursor += 1;
    names.push("route_cells".to_string());
    kinds.push(FeatureKind::SparseSet);

    let Ok(dataset) = Dataset::mixed(
        dense,
        vec![SparseSetColumn::new(sparse_rows)],
        Some(FeatureSchema { names, kinds }),
    ) else {
        return;
    };

    let split_a = split_from_bytes(data, &mut cursor, cols);
    let split_b = split_from_bytes(data, &mut cursor, cols);
    let left_branch = branch(
        split_b,
        leaf(scaled_byte(data[cursor], 4.0)),
        leaf(scaled_byte(data[cursor + 1], 4.0)),
        rows,
    );
    cursor += 2;
    let right_leaf = leaf(scaled_byte(data[cursor], 4.0));
    cursor += 1;
    let root = branch(split_a, Box::new(left_branch), right_leaf, rows);

    let model = Model {
        artifact_version: MODEL_ARTIFACT_VERSION,
        metadata: None,
        init_prediction: scaled_byte(data[cursor], 2.0),
        learning_rate: 0.05 + f64::from(data[cursor + 1] % 10) / 10.0,
        feature_count: cols,
        feature_schema: Some(dataset.feature_schema_or_default()),
        target_name: None,
        training_config: None,
        training_history: Vec::new(),
        prediction_transform: cartoboost_core::tree::PredictionTransform::Identity,
        trees: vec![Tree { root }],
    };

    for row in 0..dataset.n_rows() {
        let _ = model.try_predict_dataset_row(&dataset, row);
    }
    let _ = model.try_predict(&dataset);
});

fn scaled_byte(byte: u8, scale: f64) -> f64 {
    (f64::from(byte) / 255.0) * 2.0 * scale - scale
}

fn split_from_bytes(data: &[u8], cursor: &mut usize, cols: usize) -> Split {
    let feature = data[*cursor] as usize % cols;
    let threshold = scaled_byte(data[*cursor + 1], 12.0);
    let missing_goes_left = data[*cursor + 2] % 2 == 0;
    let choice = data[*cursor + 3] % 5;
    let split = match choice {
        0 => Split::Axis {
            feature,
            threshold,
            missing_goes_left,
        },
        1 => Split::PeriodicInterval {
            feature,
            period: 24.0,
            start: f64::from(data[*cursor + 4] % 24),
            end: f64::from(data[*cursor + 5] % 24),
            missing_goes_left,
        },
        2 => Split::SparseListContainsAny {
            sparse_feature: 0,
            ids: vec![
                u64::from(data[*cursor + 4] % 24),
                u64::from(data[*cursor + 5] % 24),
            ],
            missing_goes_left,
        },
        3 if cols >= 2 => Split::Diagonal2D {
            x_feature: 0,
            y_feature: 1,
            normal_x: 1.0,
            normal_y: if data[*cursor + 4] % 2 == 0 {
                1.0
            } else {
                -1.0
            },
            threshold,
            missing_goes_left,
        },
        _ if cols >= 2 => Split::Gaussian2D {
            x_feature: 0,
            y_feature: 1,
            center_x: scaled_byte(data[*cursor + 4], 4.0),
            center_y: scaled_byte(data[*cursor + 5], 4.0),
            radius: 0.1 + f64::from(data[*cursor + 6] % 24) / 3.0,
            missing_goes_left,
        },
        _ => Split::Axis {
            feature,
            threshold,
            missing_goes_left,
        },
    };
    let fuzzy = data[*cursor + 7] % 3 == 0 && !split.contains_sparse_list_split();
    let bandwidth = 0.1 + f64::from(data[*cursor + 8] % 16) / 4.0;
    *cursor += 9;
    if fuzzy {
        Split::Fuzzy {
            base: Box::new(split),
            bandwidth,
            kernel: FuzzyKernel::Linear,
        }
    } else {
        split
    }
}

fn leaf(value: f64) -> Box<Node> {
    Box::new(Node::Leaf {
        value,
        sample_weight_sum: 1.0,
        training_loss: 0.0,
    })
}

fn branch(split: Split, left: Box<Node>, right: Box<Node>, rows: usize) -> Node {
    Node::Branch {
        split,
        left,
        right,
        gain: 0.0,
        sample_weight_sum: rows as f64,
    }
}
