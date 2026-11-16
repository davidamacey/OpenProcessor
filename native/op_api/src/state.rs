//! What every handler of the front shares.

use std::sync::Arc;

use crate::forward::Forwarder;
use crate::metrics::Metrics;
use crate::settings::ApiSettings;

/// The state of the router: the metrics registry and the forwarder, cheap to clone (both are
/// shared handles). It holds no data a request could change, so several processes behind a load
/// balancer need nothing from each other.
#[derive(Clone)]
pub struct AppState {
    /// The registry served on `/metrics`.
    pub metrics: Arc<Metrics>,
    /// Sends the requests Rust does not own to the Python API.
    pub forwarder: Forwarder,
}

impl AppState {
    /// The state of a front for `settings`: an empty registry and a forwarder to the upstream.
    #[must_use]
    pub fn new(settings: &ApiSettings) -> Self {
        let metrics = Arc::new(Metrics::new());
        let forwarder = Forwarder::new(
            settings.upstream.clone(),
            settings.upstream_connect_timeout,
            Arc::clone(&metrics),
        );
        Self { metrics, forwarder }
    }
}
