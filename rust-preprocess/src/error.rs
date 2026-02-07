/// Error types for the preprocessing service.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug)]
pub enum AppError {
    ImageDecode(String),
    TritonInference(String),
    OpenSearch(String),
    BadRequest(String),
    Internal(String),
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AppError::ImageDecode(msg) => write!(f, "Image decode error: {msg}"),
            AppError::TritonInference(msg) => write!(f, "Triton inference error: {msg}"),
            AppError::OpenSearch(msg) => write!(f, "OpenSearch error: {msg}"),
            AppError::BadRequest(msg) => write!(f, "Bad request: {msg}"),
            AppError::Internal(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::Internal(format!("IO error: {err}"))
    }
}

impl From<tonic::transport::Error> for AppError {
    fn from(err: tonic::transport::Error) -> Self {
        AppError::TritonInference(format!("Transport error: {err}"))
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            AppError::ImageDecode(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::TritonInference(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::OpenSearch(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
        };

        // Log error for debugging
        eprintln!("[ERROR] {} - {}", status.as_u16(), message);

        let body = json!({
            "error": message,
            "status": status.as_u16(),
        });

        (status, axum::Json(body)).into_response()
    }
}
