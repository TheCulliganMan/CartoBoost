//! Shared H3/S2 helper contracts for sparse spatial features.
//!
//! This module intentionally owns deterministic behavior that must match across
//! Python and native callers: ID normalization, coordinate and level validation,
//! H3 scaffold hierarchy expansion, and sparse row sorting/deduplication. The
//! actual H3/S2 coordinate-to-cell algorithms are still provided by optional
//! Python extras because those dependencies are not core Rust dependencies.

use crate::{CartoBoostError, Result};
use rayon::prelude::*;

/// Supported sparse spatial grid families.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeoGridKind {
    H3,
    S2,
}

impl GeoGridKind {
    fn level_name(self) -> &'static str {
        match self {
            GeoGridKind::H3 => "H3 resolution",
            GeoGridKind::S2 => "S2 level",
        }
    }

    fn max_level(self) -> i64 {
        match self {
            GeoGridKind::H3 => 15,
            GeoGridKind::S2 => 30,
        }
    }
}

/// Normalize an H3 ID string into a non-negative integer cell ID.
///
/// Accepts decimal strings, `0x`-prefixed hexadecimal strings, and bare
/// hexadecimal H3 cell strings.
pub fn normalize_h3_id_text(value: &str) -> Result<u64> {
    let text = value.trim();
    if text.is_empty() {
        return Err(invalid_input("H3 IDs must not be empty"));
    }
    if text.starts_with('-') {
        return Err(invalid_input("H3 IDs must be non-negative"));
    }
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        parse_unsigned_id(
            hex,
            16,
            "H3 IDs must be decimal or hexadecimal integer strings",
        )
    } else if text.chars().all(|ch| ch.is_ascii_digit()) {
        parse_unsigned_id(
            text,
            10,
            "H3 IDs must be decimal or hexadecimal integer strings",
        )
    } else {
        parse_unsigned_id(
            text,
            16,
            "H3 IDs must be decimal or hexadecimal integer strings",
        )
    }
}

/// Normalize an S2 ID string into a non-negative integer cell ID.
///
/// Accepts decimal strings and `0x`-prefixed hexadecimal strings.
pub fn normalize_s2_id_text(value: &str) -> Result<u64> {
    let text = value.trim();
    if text.is_empty() {
        return Err(invalid_input("S2 IDs must not be empty"));
    }
    if text.starts_with('-') {
        return Err(invalid_input("S2 IDs must be non-negative"));
    }
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        parse_unsigned_id(
            hex,
            16,
            "S2 IDs must be decimal or 0x-prefixed integer strings",
        )
    } else {
        parse_unsigned_id(
            text,
            10,
            "S2 IDs must be decimal or 0x-prefixed integer strings",
        )
    }
}

/// Validate that a coordinate is finite.
pub fn normalize_coordinate(value: f64, field_name: &str) -> Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_input(format!(
            "{field_name} must be a finite coordinate"
        )))
    }
}

/// Validate and normalize an H3 resolution in the inclusive range 0..=15.
pub fn normalize_h3_resolution(value: i64, field_name: &str) -> Result<u8> {
    normalize_grid_level(value, field_name, GeoGridKind::H3)
}

/// Validate and normalize an S2 level in the inclusive range 0..=30.
pub fn normalize_s2_level(value: i64, field_name: &str) -> Result<u8> {
    normalize_grid_level(value, field_name, GeoGridKind::S2)
}

/// Validate that every parent level is strictly coarser than the child level.
pub fn validate_parent_levels(
    child_level: u8,
    parent_levels: &[u8],
    kind: GeoGridKind,
) -> Result<()> {
    for parent in parent_levels {
        if *parent >= child_level {
            let name = match kind {
                GeoGridKind::H3 => "parent_resolutions must be less than resolution",
                GeoGridKind::S2 => "parent_levels must be less than level",
            };
            return Err(invalid_input(name));
        }
    }
    Ok(())
}

/// Build a deterministic synthetic H3 parent ID for tests and schema fixtures.
///
/// This is not a real H3 parent calculation. Production parent cells should be
/// derived with the optional H3 library and passed through sparse-row assembly.
pub fn scaffold_h3_parent_id(cell: u64, resolution: u8, parent_resolution: u8) -> Result<u64> {
    validate_parent_levels(resolution, &[parent_resolution], GeoGridKind::H3)?;
    Ok(scaffold_h3_parent_id_unchecked(
        cell,
        resolution,
        parent_resolution,
    ))
}

fn scaffold_h3_parent_id_unchecked(cell: u64, resolution: u8, parent_resolution: u8) -> u64 {
    let resolution_gap = resolution - parent_resolution;
    let bucket = cell >> (u32::from(resolution_gap) * 3);
    (1_u64 << 63) | (u64::from(parent_resolution) << 56) | bucket
}

/// Expand H3 child IDs with deterministic synthetic parent IDs.
pub fn expand_h3_sparse_set(
    values: &[u64],
    resolution: u8,
    parent_resolutions: &[u8],
) -> Result<Vec<u64>> {
    validate_parent_levels(resolution, parent_resolutions, GeoGridKind::H3)?;
    let mut expanded = values
        .par_iter()
        .flat_map_iter(|cell| {
            std::iter::once(*cell).chain(
                parent_resolutions
                    .iter()
                    .map(move |parent| scaffold_h3_parent_id_unchecked(*cell, resolution, *parent)),
            )
        })
        .collect::<Vec<_>>();
    sort_dedup_ids(&mut expanded);
    Ok(expanded)
}

/// Return a sorted, deduplicated sparse row from one child ID and parent IDs.
pub fn assemble_sparse_row(child: u64, parents: &[u64]) -> Vec<u64> {
    let mut row = Vec::with_capacity(parents.len() + 1);
    row.push(child);
    row.extend_from_slice(parents);
    sort_dedup_ids(&mut row);
    row
}

/// Return sorted, deduplicated sparse rows for a full coordinate-derived column.
///
/// `parent_columns` is column-major: each inner vector contains one parent
/// level's IDs and must have the same length as `children`.
pub fn assemble_sparse_column(
    children: &[u64],
    parent_columns: &[Vec<u64>],
) -> Result<Vec<Vec<u64>>> {
    for parent_column in parent_columns {
        if parent_column.len() != children.len() {
            return Err(invalid_input(format!(
                "parent cell column has {} rows, expected {}",
                parent_column.len(),
                children.len()
            )));
        }
    }
    Ok(children
        .par_iter()
        .enumerate()
        .map(|(idx, child)| {
            let mut row = Vec::with_capacity(parent_columns.len() + 1);
            row.push(*child);
            for parent_column in parent_columns {
                row.push(parent_column[idx]);
            }
            sort_dedup_ids(&mut row);
            row
        })
        .collect())
}

/// Return sorted, deduplicated sparse rows for route-derived cell memberships.
///
/// Each input row is the sequence of grid cells touched by one decoded route.
/// Empty routes are invalid because silently treating a missing route as no
/// membership would weaken route-context features.
pub fn assemble_route_sparse_rows(route_cells: &[Vec<u64>]) -> Result<Vec<Vec<u64>>> {
    route_cells
        .par_iter()
        .enumerate()
        .map(|(idx, cells)| {
            if cells.is_empty() {
                return Err(invalid_input(format!(
                    "route cell row {idx} must contain at least one cell"
                )));
            }
            let mut row = cells.clone();
            sort_dedup_ids(&mut row);
            Ok(row)
        })
        .collect()
}

/// Validate a named coordinate feature against the expected row count.
pub fn validate_equal_row_count(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "coordinate feature '{name}' has {actual} rows, expected {expected}"
        )))
    }
}

fn normalize_grid_level(value: i64, field_name: &str, kind: GeoGridKind) -> Result<u8> {
    if value < 0 || value > kind.max_level() {
        return Err(invalid_input(format!(
            "{field_name} must be between 0 and {}",
            kind.max_level()
        )));
    }
    u8::try_from(value).map_err(|_| {
        invalid_input(format!(
            "{field_name} must be an integer {}",
            kind.level_name()
        ))
    })
}

fn parse_unsigned_id(text: &str, radix: u32, message: &'static str) -> Result<u64> {
    if text.is_empty() {
        return Err(invalid_input(message));
    }
    u64::from_str_radix(text, radix).map_err(|_| invalid_input(message))
}

fn sort_dedup_ids(values: &mut Vec<u64>) {
    values.sort_unstable();
    values.dedup();
}

fn invalid_input(message: impl Into<String>) -> CartoBoostError {
    CartoBoostError::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_h3_decimal_prefixed_hex_and_bare_hex() {
        assert_eq!(normalize_h3_id_text("12345").unwrap(), 12_345);
        assert_eq!(normalize_h3_id_text("0x2a").unwrap(), 42);
        assert_eq!(
            normalize_h3_id_text("8928308280fffff").unwrap(),
            u64::from_str_radix("8928308280fffff", 16).unwrap()
        );
    }

    #[test]
    fn normalizes_s2_decimal_and_prefixed_hex_only() {
        assert_eq!(normalize_s2_id_text("12345").unwrap(), 12_345);
        assert_eq!(normalize_s2_id_text("0x2a").unwrap(), 42);
        assert!(normalize_s2_id_text("2a").is_err());
    }

    #[test]
    fn expands_h3_scaffold_parents_deterministically() {
        let cell = normalize_h3_id_text("8928308280fffff").unwrap();
        let expanded = expand_h3_sparse_set(&[cell, cell], 9, &[5, 7]).unwrap();
        assert_eq!(
            expanded,
            vec![
                cell,
                scaffold_h3_parent_id(cell, 9, 5).unwrap(),
                scaffold_h3_parent_id(cell, 9, 7).unwrap(),
            ]
        );
    }

    #[test]
    fn validates_parent_levels_are_strictly_above_child_resolution() {
        assert!(validate_parent_levels(9, &[5, 7], GeoGridKind::H3).is_ok());
        assert!(validate_parent_levels(9, &[9], GeoGridKind::H3).is_err());
        assert!(validate_parent_levels(12, &[13], GeoGridKind::S2).is_err());
    }

    #[test]
    fn assembles_sparse_columns_in_one_native_pass() {
        let rows = assemble_sparse_column(&[10, 20], &[vec![1, 2], vec![10, 3]]).unwrap();
        assert_eq!(rows, vec![vec![1, 10], vec![2, 3, 20]]);
        assert!(assemble_sparse_column(&[10, 20], &[vec![1]]).is_err());
    }

    #[test]
    fn assemble_sparse_column_sorts_and_deduplicates_large_columns() {
        let children = (0..1_024).map(|idx| 10_000 + idx).collect::<Vec<_>>();
        let parent_a = children.iter().map(|child| child / 10).collect::<Vec<_>>();
        let parent_b = children.clone();
        let parent_c = children.iter().map(|child| child / 100).collect::<Vec<_>>();

        let rows = assemble_sparse_column(&children, &[parent_a, parent_b, parent_c]).unwrap();

        assert_eq!(rows.len(), children.len());
        for (idx, row) in rows.iter().enumerate() {
            let mut expected = vec![children[idx], children[idx] / 10, children[idx] / 100];
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(row, &expected);
        }
    }

    #[test]
    fn assembles_route_sparse_rows_without_silent_empty_routes() {
        let rows = assemble_route_sparse_rows(&[vec![3, 1, 3], vec![5, 4]]).unwrap();
        assert_eq!(rows, vec![vec![1, 3], vec![4, 5]]);
        assert!(assemble_route_sparse_rows(&[vec![]]).is_err());
    }
}
