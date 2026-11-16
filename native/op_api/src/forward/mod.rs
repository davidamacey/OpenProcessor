//! The forwarder: every request no Rust route owns goes to the Python API, and its answer comes
//! back, so the person sees the API they had.
//!
//! What it keeps: method, path and query, the headers (hop-by-hop ones removed as RFC 9110 says,
//! see [`headers`]), the request body and the response body, both streamed frame by frame with the
//! back pressure of the two sockets (nothing is read ahead of the side that is writing, and
//! nothing is held to count or hash it), and server-sent events, which are just a response body
//! that does not end. It adds `x-forwarded-for` and `x-forwarded-proto`; `host`,
//! `x-forwarded-host` and `x-request-id` pass as they are, so `src/core/request_origin.py` and the
//! logs see what they saw before.
//!
//! The client is hyper-util's legacy client
//! (<https://docs.rs/hyper-util/0.1.21/hyper_util/client/legacy/struct.Client.html>): a pool of
//! keep-alive connections to the Python API, shared by every request. A connection is never
//! upgraded: the `Upgrade` and `Connection` fields are not sent on, so a request for a protocol
//! the front does not understand reaches the Python API as a plain request and is answered as
//! one.

pub mod headers;

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::response::{IntoResponse, Response};
use http::{StatusCode, Uri, Version};
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::{TokioExecutor, TokioTimer};

use crate::errors::ApiError;
use crate::metrics::{ForwardFailure, Metrics};
use crate::middleware::request_id;
use crate::settings::Upstream;
use crate::state::AppState;

/// Idle connections kept per upstream address. Bounded: the number in use is bounded by the
/// concurrency limit.
const POOL_MAX_IDLE: usize = 256;
/// An idle pooled connection is dropped after this long: shorter than the 75 s uvicorn keeps an
/// idle connection, so a reused connection is not one the upstream has already closed.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(30);
/// Seconds a client is told to wait when the Python API cannot be reached.
const UNAVAILABLE_RETRY_AFTER_S: u32 = 5;

/// Sends requests to the Python API.
#[derive(Clone)]
pub struct Forwarder {
    client: Client<HttpConnector, Body>,
    upstream: Upstream,
    metrics: Arc<Metrics>,
}

impl Forwarder {
    /// A forwarder to `upstream`, giving up on a connection after `connect_timeout`.
    #[must_use]
    pub fn new(upstream: Upstream, connect_timeout: Duration, metrics: Arc<Metrics>) -> Self {
        let mut connector = HttpConnector::new();
        connector.set_connect_timeout(Some(connect_timeout));
        connector.set_nodelay(true);
        connector.enforce_http(true);
        let client = Client::builder(TokioExecutor::new())
            .pool_timer(TokioTimer::new())
            .pool_idle_timeout(POOL_IDLE_TIMEOUT)
            .pool_max_idle_per_host(POOL_MAX_IDLE)
            .build(connector);
        Self {
            client,
            upstream,
            metrics,
        }
    }

    fn target(&self, uri: &Uri) -> Option<Uri> {
        Uri::builder()
            .scheme(self.upstream.scheme.clone())
            .authority(self.upstream.authority.clone())
            .path_and_query(uri.path_and_query()?.clone())
            .build()
            .ok()
    }

    /// Send `request` on as `peer` sent it, and return what the Python API answers.
    pub async fn forward(&self, peer: SocketAddr, request: Request) -> Response {
        let id = request_id::of(&request).map(str::to_owned);
        let (mut parts, body) = request.into_parts();
        let Some(target) = self.target(&parts.uri) else {
            return with_id(
                ApiError::detail(StatusCode::BAD_REQUEST, "Request target not understood"),
                id.as_deref(),
            )
            .into_response();
        };
        headers::strip_hop_by_hop(&mut parts.headers);
        headers::add_forwarded(&mut parts.headers, peer.ip());
        parts.uri = target;
        parts.version = Version::HTTP_11;
        match self.client.request(Request::from_parts(parts, body)).await {
            Ok(response) => {
                let (mut parts, body) = response.into_parts();
                headers::strip_hop_by_hop(&mut parts.headers);
                Response::from_parts(parts, Body::new(body))
            }
            Err(error) => self.failed(&error, id.as_deref()).into_response(),
        }
    }

    fn failed(&self, error: &hyper_util::client::legacy::Error, id: Option<&str>) -> ApiError {
        let (failure, answer) = if error.is_connect() {
            (
                ForwardFailure::Connect,
                ApiError::detail(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "The API is starting or restarting; retry shortly",
                )
                .with_retry_after(UNAVAILABLE_RETRY_AFTER_S),
            )
        } else {
            (
                ForwardFailure::Upstream,
                ApiError::detail(StatusCode::BAD_GATEWAY, "The API did not answer correctly"),
            )
        };
        tracing::warn!(error = %error, ?failure, "the Python API could not answer");
        self.metrics.forward_failed(failure);
        with_id(answer, id)
    }
}

fn with_id(error: ApiError, id: Option<&str>) -> ApiError {
    match id {
        Some(id) => error.with_request_id(id),
        None => error,
    }
}

/// The fallback of the router: forward the request.
pub async fn handler(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
) -> Response {
    state.forwarder.forward(peer, request).await
}
