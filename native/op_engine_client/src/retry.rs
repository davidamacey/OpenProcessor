//! Which failures are sent again, and when to give up: the policy of
//! `src/clients/frame_engine.py` (`_retry_again`, `_BACKOFF`, tenacity `stop_after_delay`,
//! `wait_exponential_jitter`; the gRPC status-code guide, <https://grpc.io/docs/guides/status-codes/>,
//! calls `UNAVAILABLE` "most likely a transient condition, which can be corrected by retrying
//! with a backoff").
//!
//! The engine's call is a pure function of the photo and the plan, so every repeat is safe.
//! The policy is kept here, not in a retry crate: it has a limit for each class of error (an
//! overload is retried until the photo's deadline, an unavailable engine a fixed number of times,
//! nothing else), and the tests replay the decisions tenacity makes call by call.

use std::time::Duration;

use tonic::{Code, Status};

/// What the engine's own `INTERNAL` details start with (`ENGINE_ERROR_PREFIX`): the photo's own
/// failure, which a repeat does not change. An `INTERNAL` without it was made by the transport.
pub const ENGINE_ERROR_PREFIX: &str = "frame engine: ";
/// Retries of an unavailable engine or a lost call, after the first try.
pub const RETRIES: u32 = 3;
/// How long a photo waits for the engine across retries.
pub const DEADLINE: Duration = Duration::from_secs(120);

const BACKOFF_FIRST: f64 = 0.05;
const BACKOFF_MAX: f64 = 2.0;
const BACKOFF_JITTER: f64 = 0.2;

/// How a failed call is treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// `RESOURCE_EXHAUSTED`: the engine is busy; retried until the deadline.
    Overload,
    /// `UNAVAILABLE`, or an `INTERNAL` the transport made: retried [`RETRIES`] times.
    Transient,
    /// Everything else: the answer stands.
    Final,
}

/// The class of a failed call's status.
#[must_use]
pub fn classify(status: &Status) -> Class {
    match status.code() {
        Code::ResourceExhausted => Class::Overload,
        Code::Unavailable => Class::Transient,
        Code::Internal if !status.message().starts_with(ENGINE_ERROR_PREFIX) => Class::Transient,
        _ => Class::Final,
    }
}

/// What to do after a failed call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Give up and report the failure.
    Stop,
    /// Call again after this wait.
    After(Duration),
}

/// `min(2 s, 0.05 s * 2^(attempt - 1) + jitter)` with `jitter` in `[0, 0.2]` s, as tenacity's
/// `wait_exponential_jitter(multiplier=0.05, max=2.0, jitter=0.2)` computes; `unit` in `[0, 1]`
/// scales the jitter.
#[must_use]
pub fn backoff(attempt: u32, unit: f64) -> Duration {
    let exponent = i32::try_from(attempt.saturating_sub(1)).unwrap_or(i32::MAX);
    let seconds = BACKOFF_FIRST * 2f64.powi(exponent) + BACKOFF_JITTER * unit.clamp(0.0, 1.0);
    Duration::from_secs_f64(seconds.clamp(0.0, BACKOFF_MAX))
}

/// The decision after failed call number `attempt` (the first is 1), `elapsed` since the first
/// began: tenacity's order, retry predicate first, then the deadline (`stop_after_delay` stops
/// once `elapsed >= deadline`, so the last wait may run past it).
#[must_use]
pub fn decide(
    class: Class,
    attempt: u32,
    elapsed: Duration,
    deadline: Duration,
    unit: f64,
) -> Verdict {
    let again = match class {
        Class::Overload => true,
        Class::Transient => attempt <= RETRIES,
        Class::Final => false,
    };
    if !again || elapsed >= deadline {
        Verdict::Stop
    } else {
        Verdict::After(backoff(attempt, unit))
    }
}
