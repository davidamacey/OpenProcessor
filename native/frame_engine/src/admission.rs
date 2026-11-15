//! Admission as the front needs it (plan 17 section 2.2, "Admission" and "Request"): the cap on
//! photos in flight, answered `RESOURCE_EXHAUSTED` at once when full, and the arithmetic of the
//! request deadline that the stages and the model calls share.
//!
//! The frame budget (decoded bytes, a fair semaphore with a bounded wait) is `gpu_frame`'s
//! (`Admission::admit` of plan section 2.0). It is reached through [`FrameBudget`], so the
//! pipeline that joins the two crates plugs the real one in; this crate holds only the shape.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

use crate::pyfmt::format_g;

/// The engine is full right now: a retryable answer (`RESOURCE_EXHAUSTED`), counted by `cause`
/// (`in_flight`, `budget`, `slab`, `colour`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Overload {
    pub cause: String,
    pub detail: String,
}

/// `cause: detail`, or the cause alone (Python's `OverloadError`).
impl std::fmt::Display for Overload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.detail.is_empty() {
            f.write_str(&self.cause)
        } else {
            write!(f, "{}: {}", self.cause, self.detail)
        }
    }
}

impl std::error::Error for Overload {}

/// What the front assumes of the frame budget in `gpu_frame` (plan section 2.0, `Admission::admit`):
/// reserve `bytes` of decoded frame for a photo, waiting no later than `deadline`, and hold them
/// until the permit is dropped. `Err` is [`Overload`] with cause `budget`; a need larger than the
/// whole budget is the pipeline's `over_budget` status, decided before it asks.
pub trait FrameBudget: Send + Sync + 'static {
    type Permit: Send + 'static;

    fn admit(
        &self,
        bytes: u64,
        deadline: Instant,
    ) -> impl Future<Output = Result<Self::Permit, Overload>> + Send;
}

/// `OP_FRAME_ENGINE_MAX_IN_FLIGHT` photos admitted at once, running or queued.
#[derive(Debug)]
pub struct InFlightCap {
    count: AtomicU32,
    cap: u32,
    idle: Notify,
}

impl InFlightCap {
    pub fn new(cap: u32) -> Arc<Self> {
        Arc::new(Self {
            count: AtomicU32::new(0),
            cap,
            idle: Notify::new(),
        })
    }

    pub fn cap(&self) -> u32 {
        self.cap
    }

    pub fn current(&self) -> u32 {
        self.count.load(Ordering::Acquire)
    }

    /// Admits one photo, or says how many are in flight when the cap is reached. Never waits.
    pub fn try_admit(self: &Arc<Self>) -> Result<InFlightPermit, u32> {
        self.count
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.cap).then_some(n + 1)
            })
            .map(|_| InFlightPermit(Arc::clone(self)))
    }

    /// Completes when no photo is in flight.
    pub async fn wait_idle(&self) {
        loop {
            let released = self.idle.notified();
            tokio::pin!(released);
            // enabled before the check: a release between check and wait still wakes it
            released.as_mut().enable();
            if self.current() == 0 {
                return;
            }
            released.await;
        }
    }
}

/// One admitted photo; dropping it frees the slot.
#[derive(Debug)]
pub struct InFlightPermit(Arc<InFlightCap>);

impl Drop for InFlightPermit {
    fn drop(&mut self) {
        self.0.count.fetch_sub(1, Ordering::AcqRel);
        self.0.idle.notify_waiters();
    }
}

/// The deadline of one request: `OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS` after it arrived.
#[derive(Debug, Clone, Copy)]
pub struct RequestClock {
    pub arrived: Instant,
    pub deadline: Instant,
    seconds: f64,
}

impl RequestClock {
    pub fn start(arrived: Instant, deadline_seconds: f64) -> Self {
        Self {
            arrived,
            deadline: arrived + Duration::from_secs_f64(deadline_seconds),
            seconds: deadline_seconds,
        }
    }

    /// What is left of the request (zero when it is past).
    pub fn remaining(&self, now: Instant) -> Duration {
        self.deadline.saturating_duration_since(now)
    }

    /// The time one model call may take: `OP_FRAME_ENGINE_TRITON_TIMEOUT_SECONDS` or what is left
    /// of the request, whichever is less (plan 17 section 2.2, "Model calls").
    pub fn call_timeout(&self, per_call: Duration, now: Instant) -> Duration {
        per_call.min(self.remaining(now))
    }

    /// The status details of a request answered at its deadline, as the Python front words them.
    pub fn exceeded_message(&self, stage: &str) -> String {
        format!(
            "no answer within {} s (stage {stage})",
            format_g(self.seconds)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_admits_up_to_its_size_and_a_drop_frees_a_slot() {
        let cap = InFlightCap::new(2);
        let a = cap.try_admit().expect("first");
        let _b = cap.try_admit().expect("second");
        assert_eq!(cap.try_admit().expect_err("full"), 2);
        assert_eq!(cap.current(), 2);
        drop(a);
        assert_eq!(cap.current(), 1);
        let _c = cap.try_admit().expect("a slot came free");
        assert_eq!(cap.cap(), 2);
    }

    #[test]
    fn the_cap_holds_under_contention() {
        let cap = InFlightCap::new(5);
        let admitted = std::sync::atomic::AtomicU32::new(0);
        std::thread::scope(|s| {
            for _ in 0..16 {
                s.spawn(|| {
                    let held: Vec<_> = (0..50).filter_map(|_| cap.try_admit().ok()).collect();
                    admitted.fetch_add(u32::try_from(held.len()).expect("small"), Ordering::SeqCst);
                    assert!(cap.current() <= 5);
                    // the permits are held to the end of the closure: no slot is handed twice
                    drop(held);
                });
            }
        });
        assert_eq!(cap.current(), 0);
        assert!(admitted.load(Ordering::SeqCst) >= 5);
    }

    #[tokio::test]
    async fn wait_idle_ends_when_the_last_photo_leaves() {
        let cap = InFlightCap::new(4);
        cap.wait_idle().await;
        let a = cap.try_admit().expect("a");
        let b = cap.try_admit().expect("b");
        let waiter = tokio::spawn({
            let cap = Arc::clone(&cap);
            async move { cap.wait_idle().await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());
        drop(a);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "one photo is still in flight");
        drop(b);
        tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("idle in time")
            .expect("no panic");
    }

    #[test]
    fn deadline_arithmetic_never_goes_below_zero_and_caps_each_model_call() {
        let t0 = Instant::now();
        let clock = RequestClock::start(t0, 60.0);
        assert_eq!(clock.deadline - clock.arrived, Duration::from_secs(60));
        assert_eq!(
            clock.remaining(t0 + Duration::from_secs(45)),
            Duration::from_secs(15)
        );
        assert_eq!(
            clock.remaining(t0 + Duration::from_secs(61)),
            Duration::ZERO
        );
        let per_call = Duration::from_secs(15);
        assert_eq!(clock.call_timeout(per_call, t0), per_call);
        assert_eq!(
            clock.call_timeout(per_call, t0 + Duration::from_secs(50)),
            Duration::from_secs(10),
            "what is left of the request wins"
        );
        assert_eq!(
            clock.call_timeout(per_call, t0 + Duration::from_secs(70)),
            Duration::ZERO
        );
    }

    #[test]
    fn the_deadline_message_is_the_python_front_text() {
        let t0 = Instant::now();
        assert_eq!(
            RequestClock::start(t0, 60.0).exceeded_message("embed_pe"),
            "no answer within 60 s (stage embed_pe)"
        );
        assert_eq!(
            RequestClock::start(t0, 0.3).exceeded_message("detect"),
            "no answer within 0.3 s (stage detect)"
        );
    }

    #[test]
    fn an_overload_reads_cause_then_detail() {
        let o = Overload {
            cause: "budget".into(),
            detail: "full".into(),
        };
        assert_eq!(o.to_string(), "budget: full");
        assert_eq!(
            Overload {
                cause: "slab".into(),
                detail: String::new()
            }
            .to_string(),
            "slab"
        );
    }
}
