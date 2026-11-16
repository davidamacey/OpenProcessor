//! The routes Rust owns. Every request that matches none of them is forwarded to the Python API
//! ([`crate::forward`]).
//!
//! The probes sit outside the load-shed and concurrency layers ([`crate::middleware`]): a front
//! that is full must still answer its health check. A route family a later slice moves to Rust is
//! one line in a function of its own here, alphabetical, and goes through those layers.

use axum::Router;
use axum::handler::Handler;
use axum::response::{IntoResponse, Response};
use axum::routing::{MethodRouter, get};
use http::StatusCode;
use http::header::{ALLOW, HeaderValue};

use crate::errors::ApiError;
use crate::health;
use crate::state::AppState;

/// `/live` and the front's metrics: answered even when the front is at its concurrency limit.
pub fn probes() -> Router<AppState> {
    health::routes()
}

/// A route that answers `GET` only, as a FastAPI `@router.get` route does: any other method,
/// `HEAD` and `OPTIONS` included, is a 405 with `{"detail": "Method Not Allowed"}` and
/// `Allow: GET`. (axum's own `get` also answers `HEAD` and its 405 has no body.)
pub fn get_only<H, T, S>(handler: H) -> MethodRouter<S>
where
    H: Handler<T, S>,
    T: 'static,
    S: Clone + Send + Sync + 'static,
{
    get(handler)
        .head(method_not_allowed)
        .fallback(method_not_allowed)
}

async fn method_not_allowed() -> Response {
    let mut answer =
        ApiError::detail(StatusCode::METHOD_NOT_ALLOWED, "Method Not Allowed").into_response();
    answer
        .headers_mut()
        .insert(ALLOW, HeaderValue::from_static("GET"));
    answer
}
