//! The front's Prometheus series, from one registry served on [`crate::health::FRONT_METRICS_PATH`].
//!
//! `http_request_duration_seconds{method, route, status}` has the name, labels and buckets of
//! `src/core/metrics.py`. A request is observed by the process that answers it: the front observes
//! the routes it owns, and the requests it forwards are observed by the Python API (scraped on its
//! own port), so a dashboard that sums both counts each request once.
//!
//! The text is OpenMetrics (`prometheus-client` 0.26 writes no other); a counter `x` appears as
//! `x_total`.

use std::fmt;
use std::time::Duration;

use http::{Method, StatusCode};
use prometheus_client::encoding::EncodeLabelSet;
use prometheus_client::encoding::text::encode;
use prometheus_client::metrics::counter::Counter;
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::histogram::Histogram;
use prometheus_client::registry::Registry;

/// The content type of [`Metrics::render`] (OpenMetrics 1.0).
pub const CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

/// The latency buckets of `src/core/metrics.py`, in seconds.
const BUCKETS: [f64; 12] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
];

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct HttpLabels {
    method: String,
    route: String,
    status: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct ByReason {
    reason: String,
}

/// Why a forwarded request got no answer from the Python API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForwardFailure {
    /// The connection could not be made, or timed out being made.
    Connect,
    /// The connection was made and then broke or sent something that is not HTTP.
    Upstream,
}

impl ForwardFailure {
    fn label(self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Upstream => "upstream",
        }
    }
}

/// Every series of one front process.
pub struct Metrics {
    registry: Registry,
    http_duration: Family<HttpLabels, Histogram>,
    forward_failures: Family<ByReason, Counter>,
    shed: Counter,
}

impl Metrics {
    /// A registry holding the front's series, all at zero.
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let http_duration =
            Family::<HttpLabels, Histogram>::new_with_constructor(|| Histogram::new(BUCKETS));
        registry.register(
            "http_request_duration_seconds",
            "HTTP request latency by method, route template, and status code",
            http_duration.clone(),
        );
        let forward_failures = Family::<ByReason, Counter>::default();
        registry.register(
            "op_api_forward_failures",
            "Requests the front could not get answered by the Python API, by reason",
            forward_failures.clone(),
        );
        let shed = Counter::default();
        registry.register(
            "op_api_requests_shed",
            "Requests answered 503 at once because the concurrency limit was reached",
            shed.clone(),
        );
        Self {
            registry,
            http_duration,
            forward_failures,
            shed,
        }
    }

    /// Record a request the front answered: its method, the route template, its status and how
    /// long it took.
    pub fn observe_http(
        &self,
        method: &Method,
        route: &str,
        status: StatusCode,
        elapsed: Duration,
    ) {
        self.http_duration
            .get_or_create(&HttpLabels {
                method: method.as_str().to_owned(),
                route: route.to_owned(),
                status: status.as_str().to_owned(),
            })
            .observe(elapsed.as_secs_f64());
    }

    /// Record a forwarded request the Python API did not answer.
    pub fn forward_failed(&self, failure: ForwardFailure) {
        self.forward_failures
            .get_or_create(&ByReason {
                reason: failure.label().to_owned(),
            })
            .inc();
    }

    /// Record a request shed at the concurrency limit.
    pub fn shed(&self) {
        self.shed.inc();
    }

    /// The exposition text of every series.
    ///
    /// # Errors
    ///
    /// The encoder's formatting error, which does not happen for these series.
    pub fn render(&self) -> Result<String, fmt::Error> {
        let mut text = String::new();
        encode(&mut text, &self.registry)?;
        Ok(text)
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_observed_request_appears_with_the_python_series_name_and_labels() {
        let metrics = Metrics::new();
        metrics.observe_http(
            &Method::GET,
            "/live",
            StatusCode::OK,
            Duration::from_millis(7),
        );
        let text = metrics.render().unwrap();
        assert!(
            text.contains(
                "http_request_duration_seconds_count{method=\"GET\",route=\"/live\",status=\"200\"} 1"
            ),
            "{text}"
        );
        assert!(
            text.contains("http_request_duration_seconds_bucket{le=\"0.01\",method=\"GET\",route=\"/live\",status=\"200\"} 1"),
            "{text}"
        );
        assert!(text.contains("le=\"30.0\""), "{text}");
        assert!(
            text.contains("# TYPE http_request_duration_seconds histogram"),
            "{text}"
        );
    }

    #[test]
    fn the_counters_start_at_zero_series_free_and_count_what_happens() {
        let metrics = Metrics::new();
        metrics.shed();
        metrics.shed();
        metrics.forward_failed(ForwardFailure::Connect);
        let text = metrics.render().unwrap();
        assert!(text.contains("op_api_requests_shed_total 2"), "{text}");
        assert!(
            text.contains("op_api_forward_failures_total{reason=\"connect\"} 1"),
            "{text}"
        );
        assert!(!text.contains("reason=\"upstream\""), "{text}");
    }
}
