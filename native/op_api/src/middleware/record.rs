//! Latency of the routes Rust owns, into `http_request_duration_seconds`.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::{MatchedPath, Request, State};
use axum::middleware::Next;
use axum::response::Response;

use crate::metrics::Metrics;

/// Observe the request when it matched one of the front's own routes; a forwarded request has no
/// route here and is observed by the Python API.
///
/// The layer must be applied with `Router::layer`, which runs it after routing: the matched path
/// is only known then.
pub async fn record(State(metrics): State<Arc<Metrics>>, request: Request, next: Next) -> Response {
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(|path| path.as_str().to_owned());
    let method = request.method().clone();
    let started = Instant::now();
    let response = next.run(request).await;
    if let Some(route) = route {
        metrics.observe_http(&method, &route, response.status(), started.elapsed());
    }
    response
}
