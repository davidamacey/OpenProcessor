//! What the server counts about the engine: the hooks for its metrics.

/// How a readiness probe ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeOutcome {
    /// The engine said ready at the first call.
    Ready,
    /// Ready on a later attempt, after the transport refused an earlier one.
    Recovered,
    /// The engine answered, and the answer was "not ready".
    NotReady,
    /// No attempt got an answer: the engine could not be asked.
    Undelivered,
}

impl ProbeOutcome {
    /// The label `op_engine_probe_total` carries (`PROBE_*` of `src/clients/frame_engine.py`).
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Recovered => "recovered",
            Self::NotReady => "not_ready",
            Self::Undelivered => "undelivered",
        }
    }
}

/// Counters the client updates as it works; the server's metrics registry implements it
/// (`op_engine_call_failures_total{code, outcome}` and the probe counter).
pub trait CallObserver: Send + Sync {
    /// A failed call is being sent again; `code` is the status name (`UNAVAILABLE`, ...).
    fn retried(&self, code: &str);
    /// A failed call was not sent again and the photo failed.
    fn exhausted(&self, code: &str);
    /// A readiness probe ended.
    fn probed(&self, outcome: ProbeOutcome);
}
