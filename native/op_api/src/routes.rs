//! The routes Rust owns: one line per route family, alphabetical. Every request that matches
//! none of them is forwarded to the Python API ([`crate::forward`]).

use axum::Router;

use crate::health;
use crate::state::AppState;

/// All routes the front answers itself.
pub fn owned() -> Router<AppState> {
    Router::new().merge(health::routes())
}
