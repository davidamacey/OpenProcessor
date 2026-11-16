//! `/live`, the front's metrics path and the probe the container's health check runs.
//!
//! `/live` answers `{"status": "alive"}` as the Python API's route of the same name does: the
//! process is up, no dependency is asked. [`FRONT_METRICS_PATH`] serves the front's own registry
//! ([`crate::metrics`]). `/metrics` is not here: it stays the Python API's and is forwarded like
//! any other path, so every reader of the Python series on the API port (bulk ingest, the bench
//! scripts) gets them as before.

use std::error::Error;
use std::fmt;
use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use http::header::CONTENT_TYPE;
use http::{StatusCode, Uri};
use hyper_util::client::legacy::Client;
use hyper_util::rt::TokioExecutor;
use serde_json::json;

use crate::errors::ApiError;
use crate::metrics::CONTENT_TYPE as METRICS_CONTENT_TYPE;
use crate::routes::get_only;
use crate::state::AppState;

/// Where the front's own series are served. Prometheus scrapes it as the job `api-front`.
pub const FRONT_METRICS_PATH: &str = "/metrics/front";

/// The routes of this module, for [`crate::routes`].
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/live", get_only(live))
        .route(FRONT_METRICS_PATH, get_only(metrics))
}

async fn live() -> axum::Json<serde_json::Value> {
    axum::Json(json!({ "status": "alive" }))
}

async fn metrics(State(state): State<AppState>) -> Response {
    match state.metrics.render() {
        Ok(text) => ([(CONTENT_TYPE, METRICS_CONTENT_TYPE)], text).into_response(),
        Err(e) => {
            tracing::error!(error = %e, "the metrics registry did not encode");
            ApiError::detail(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
                .into_response()
        }
    }
}

/// Why a probe did not find the front healthy.
#[derive(Debug)]
pub enum ProbeError {
    /// The request did not complete, or no answer came in time.
    Unreachable(String),
    /// The front answered, with a status that is not a success.
    Status(StatusCode),
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unreachable(why) => write!(f, "no answer: {why}"),
            Self::Status(status) => write!(f, "answered {status}"),
        }
    }
}

impl Error for ProbeError {}

/// Ask the front on `addr` for `path` and succeed on a 2xx answer, within `timeout`. This is what
/// the container's health check runs (the image has no curl).
///
/// # Errors
///
/// [`ProbeError::Unreachable`] when the connection or the answer fails or takes too long,
/// [`ProbeError::Status`] when the answer is not a success.
pub async fn probe(addr: SocketAddr, path: &str, timeout: Duration) -> Result<(), ProbeError> {
    let uri = Uri::builder()
        .scheme("http")
        .authority(addr.to_string())
        .path_and_query(path)
        .build()
        .map_err(|e| ProbeError::Unreachable(e.to_string()))?;
    let client = Client::builder(TokioExecutor::new()).build_http::<Body>();
    let response = tokio::time::timeout(timeout, client.get(uri))
        .await
        .map_err(|_| ProbeError::Unreachable(format!("no answer within {timeout:?}")))?
        .map_err(|e| ProbeError::Unreachable(e.to_string()))?;
    if response.status().is_success() {
        Ok(())
    } else {
        Err(ProbeError::Status(response.status()))
    }
}
