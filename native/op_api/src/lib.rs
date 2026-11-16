//! The front of the API: router assembly, middleware, error shapes, health, metrics and the
//! forwarder, with one module per route family.
//!
//! The front listens on the API's port. It answers the routes Rust owns ([`routes`]) and forwards
//! every other request to the Python API ([`forward`]) unchanged, so the person sees the API they
//! had. [`build`] makes the router; [`serve`] runs it.
#![forbid(unsafe_code)]

pub mod cors;
pub mod errors;
pub mod forward;
pub mod health;
pub mod metrics;
pub mod middleware;
pub mod routes;
pub mod serve;
pub mod settings;
pub mod state;

use axum::Router;

use crate::settings::ApiSettings;
use crate::state::AppState;

/// The front's router: the probes with their CORS rule, the forwarder for everything else behind
/// the concurrency limit, and the layers every request passes ([`middleware`]).
///
/// # Errors
///
/// The regex error when the CORS origin pattern does not compile, which a test rules out.
pub fn build(settings: &ApiSettings) -> Result<Router, regex::Error> {
    let state = AppState::new(settings);
    let origins = cors::origin_pattern()?;
    let probes = middleware::observe(routes::probes().layer(cors::layer(origins.clone())), &state);
    let limited = middleware::limit(
        Router::new().fallback(forward::handler),
        &state,
        settings.concurrency_limit,
    );
    let app = probes.merge(limited).with_state(state);
    Ok(middleware::common(app, &origins))
}
