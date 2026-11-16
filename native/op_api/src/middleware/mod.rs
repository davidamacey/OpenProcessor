//! The layers every request passes, in this order from the outside in:
//!
//! 1. [`request_id::drop_unsafe`] and tower-http's `SetRequestIdLayer`: the id of the request.
//! 2. tower-http's `TraceLayer`: one JSON log line per request with its `request_id`.
//! 3. tower-http's `PropagateRequestIdLayer`: the id on the response.
//! 4. tower-http's `CatchPanicLayer`: a panic in a handler answers 500 and the process lives.
//! 5. tower's `LoadShedLayer` over a `GlobalConcurrencyLimitLayer`: the request over the limit is
//!    answered 503 with `Retry-After` at once, not queued (uvicorn's `--limit-concurrency`). The
//!    permit is held until the response head is produced, so a long streamed body (a server-sent
//!    event stream, a big download) does not hold one; the open connections are bounded apart, by
//!    [`crate::serve`].
//! 6. [`record::record`]: latency of the routes Rust owns.
//!
//! Pages: <https://docs.rs/tower-http/0.7.1/tower_http/>, <https://docs.rs/tower/0.5.3/tower/>,
//! <https://docs.rs/axum/0.8.9/axum/middleware/index.html> (the order of `Router::layer`).

pub mod record;
pub mod request_id;

use std::any::Any;
use std::sync::Arc;

use axum::body::Body;
use axum::error_handling::HandleErrorLayer;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::{BoxError, Extension, Router};
use http::StatusCode;
use tower::ServiceBuilder;
use tower::limit::GlobalConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::load_shed::error::Overloaded;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::request_id::{
    MakeRequestUuid, PropagateRequestIdLayer, RequestId, SetRequestIdLayer,
};
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use tracing::Level;

use crate::errors::ApiError;
use crate::metrics::Metrics;
use crate::state::AppState;

/// Seconds the 503 of a shed request tells the client to wait.
const SHED_RETRY_AFTER_S: u32 = 1;

/// Wrap `router` in the layers above, allowing `concurrency_limit` requests at once.
pub fn apply(router: Router, state: &AppState, concurrency_limit: usize) -> Router {
    let shed_metrics = Arc::clone(&state.metrics);
    router.layer(
        ServiceBuilder::new()
            .layer(axum::middleware::from_fn(request_id::drop_unsafe))
            .layer(SetRequestIdLayer::new(
                request_id::X_REQUEST_ID,
                MakeRequestUuid,
            ))
            .layer(
                TraceLayer::new_for_http()
                    .make_span_with(|request: &Request<Body>| {
                        tracing::info_span!(
                            "request",
                            request_id = request_id::of(request).unwrap_or("-"),
                            method = %request.method(),
                            path = request.uri().path(),
                        )
                    })
                    .on_response(DefaultOnResponse::new().level(Level::INFO)),
            )
            .layer(PropagateRequestIdLayer::new(request_id::X_REQUEST_ID))
            .layer(CatchPanicLayer::custom(panic_response))
            .layer(HandleErrorLayer::new(
                move |id: Option<Extension<RequestId>>, error: BoxError| {
                    let metrics = Arc::clone(&shed_metrics);
                    async move { overload_response(&metrics, id.as_ref(), &error) }
                },
            ))
            .layer(LoadShedLayer::new())
            .layer(GlobalConcurrencyLimitLayer::new(concurrency_limit))
            .layer(axum::middleware::from_fn_with_state(
                Arc::clone(&state.metrics),
                record::record,
            )),
    )
}

fn overload_response(
    metrics: &Metrics,
    id: Option<&Extension<RequestId>>,
    error: &BoxError,
) -> Response {
    let (status, detail) = if error.is::<Overloaded>() {
        metrics.shed();
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "The server is at its concurrency limit; retry shortly",
        )
    } else {
        tracing::error!(%error, "a layer of the front failed");
        (StatusCode::INTERNAL_SERVER_ERROR, "Internal server error")
    };
    let mut answer = ApiError::detail(status, detail);
    if status == StatusCode::SERVICE_UNAVAILABLE {
        answer = answer.with_retry_after(SHED_RETRY_AFTER_S);
    }
    if let Some(Extension(id)) = id
        && let Ok(id) = id.header_value().to_str()
    {
        answer = answer.with_request_id(id);
    }
    answer.into_response()
}

fn panic_response(payload: Box<dyn Any + Send + 'static>) -> Response {
    let what = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("not a string");
    tracing::error!(panic = what, "a handler panicked");
    ApiError::detail(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error").into_response()
}
