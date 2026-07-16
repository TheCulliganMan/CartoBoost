use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum NeuralError {
    InvalidArgument(String),
    Io(std::io::Error),
    Json(serde_json::Error),
    ChecksumMismatch { expected: String, actual: String },
    DuplicateId(u64),
    InvalidRowCount { expected: usize, actual: usize },
}

impl Display for NeuralError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::Io(err) => write!(f, "io error: {err}"),
            Self::Json(err) => write!(f, "json error: {err}"),
            Self::ChecksumMismatch { expected, actual } => {
                write!(f, "checksum mismatch: expected {expected}, got {actual}")
            }
            Self::DuplicateId(id) => write!(f, "duplicate embedding id: {id}"),
            Self::InvalidRowCount { expected, actual } => {
                write!(f, "row count mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for NeuralError {}

impl From<std::io::Error> for NeuralError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

impl From<serde_json::Error> for NeuralError {
    fn from(err: serde_json::Error) -> Self {
        Self::Json(err)
    }
}

impl From<cartoboost_accelerator::AcceleratorError> for NeuralError {
    fn from(err: cartoboost_accelerator::AcceleratorError) -> Self {
        match err {
            cartoboost_accelerator::AcceleratorError::InvalidArgument(message) => {
                Self::InvalidArgument(message)
            }
        }
    }
}

pub type Result<T> = std::result::Result<T, NeuralError>;
