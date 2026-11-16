//! The routes Rust owns. Every request that matches none of them is forwarded to the Python API
//! ([`crate::forward`]).
//!
//! The probes sit outside the load-shed and concurrency layers ([`crate::middleware`]): a front
//! that is full must still answer its health check. A route family a later slice moves to Rust is
//! one line in a function of its own here, alphabetical, and goes through those layers.

use axum::Router;

use crate::health;
use crate::state::AppState;

/// `/live` and the front's metrics: answered even when the front is at its concurrency limit.
pub fn probes() -> Router<AppState> {
    health::routes()
}
