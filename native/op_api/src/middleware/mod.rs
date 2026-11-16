//! The layers every request passes, in this order from the outside in:
//!
//! 1. [`request_id::assign`]: the id of the request, on the request and on the response.
//! 2. tower-http's `TraceLayer`: one JSON log line per request with its `request_id`.
//! 3. [`crate::cors::on_front_answers`]: CORS headers on the answers the front builds itself
//!    (the 503s, 502, 400 and the 500 below), so a browser on another origin can read them.
//! 4. `catch_panic`: a panic in a handler answers 500 with the Python 500 body, request id
//!    included, and the process lives.
//! 5. tower's `LoadShedLayer` over a `GlobalConcurrencyLimitLayer`: the request over the limit is
//!    answered 503 with `Retry-After` at once, not queued. The permit is held until the response
//!    head is produced, so a long streamed body (a server-sent event stream, a big download) does
//!    not hold one; the open connections are bounded apart, by [`crate::serve`]. Layers 5 and 6
//!    wrap what is forwarded ([`limit`]); `/live` and the front's metrics path ([`observe`]) are
//!    outside layer 5, so a front at its limit still answers its health probe.
//! 6. [`record::record`]: latency of the routes Rust owns.
//!
//! Pages: <https://docs.rs/tower-http/0.7.1/tower_http/>, <https://docs.rs/tower/0.5.3/tower/>,
//! <https://docs.rs/axum/0.8.9/axum/middleware/index.html> (the order of `Router::layer`).

pub mod record;
pub mod request_id;

use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use axum::body::Body;
use axum::error_handling::HandleErrorLayer;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::{BoxError, Extension, Router};
use futures_util::FutureExt;
use http::StatusCode;
use regex::Regex;
use tower::ServiceBuilder;
use tower::limit::GlobalConcurrencyLimitLayer;
use tower::load_shed::LoadShedLayer;
use tower::load_shed::error::Overloaded;
use tower_http::trace::{DefaultOnResponse, TraceLayer};
use tracing::Level;

use crate::cors;
use crate::errors::ApiError;
use crate::metrics::Metrics;
use crate::middleware::request_id::RequestId;
use crate::state::AppState;

/// Seconds the 503 of a shed request tells the client to wait.
const SHED_RETRY_AFTER_S: u32 = 1;

/// The layers every request passes, around the whole front: 1 to 4 above.
pub fn common(router: Router, origins: &Regex) -> Router {
    router.layer(
        ServiceBuilder::new()
            .layer(axum::middleware::from_fn(request_id::assign))
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
            .layer(axum::middleware::from_fn_with_state(
                origins.clone(),
                cors::on_front_answers,
            ))
            .layer(axum::middleware::from_fn(catch_panic)),
    )
}

/// Answer 503 at once to the request that finds `concurrency_limit` requests in flight (layer 5),
/// then record latency (layer 6). Only what the Python API answers goes through here; the probes
/// do not, so a front at its limit still answers its health check.
pub fn limit(
    router: Router<AppState>,
    state: &AppState,
    concurrency_limit: usize,
) -> Router<AppState> {
    let shed_metrics = Arc::clone(&state.metrics);
    router.layer(
        ServiceBuilder::new()
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

/// Record latency of the routes in `router` (layer 6) without limiting them.
pub fn observe(router: Router<AppState>, state: &AppState) -> Router<AppState> {
    router.layer(axum::middleware::from_fn_with_state(
        Arc::clone(&state.metrics),
        record::record,
    ))
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
    if let Some(Extension(RequestId(id))) = id {
        answer = answer.with_request_id(id);
    }
    answer.into_response()
}

/// Run the rest of the stack and answer 500 with the Python body (`{"detail": "Internal server
/// error", "request_id": ...}`) if it panics. tower-http's `CatchPanicLayer` cannot do this: its
/// handler sees the panic payload and not the request, so it cannot put the id in the body.
async fn catch_panic(request: Request, next: axum::middleware::Next) -> Response {
    let id = request.extensions().get::<RequestId>().cloned();
    match AssertUnwindSafe(next.run(request)).catch_unwind().await {
        Ok(response) => response,
        Err(payload) => {
            let what = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("not a string");
            tracing::error!(panic = what, "a handler panicked");
            let answer =
                ApiError::detail(StatusCode::INTERNAL_SERVER_ERROR, "Internal server error");
            match id {
                Some(RequestId(id)) => answer.with_request_id(&id),
                None => answer,
            }
            .into_response()
        }
    }
}
