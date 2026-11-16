//! What a running job's handler can see of its lease, and the loop that keeps the lease alive.
//!
//! A job holds a lease on its row. [`heartbeat_loop`] renews it every tick with the fencing token;
//! the answer carries any cancel or pause request, which the handler reads through
//! [`LeaseWatch`]. `may_write` is the self-fence for writes outside the database: false once the
//! last good renewal is older than `lease_ttl - fence_margin`, so a holder that cannot reach the
//! database stops acting before another runner can take the job over (plan 02b section 3.1,
//! step 3). Time is tokio's, so tests run it on a paused clock.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use tokio::time::{Instant, MissedTickBehavior};

use crate::store::{Control, JobStore, Lease};

/// The timing of one lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeaseTiming {
    /// How long a lease lasts without a renewal.
    pub ttl: Duration,
    /// Between renewals.
    pub tick: Duration,
    /// A holder stops writing this long before its lease can expire.
    pub fence_margin: Duration,
}

impl LeaseTiming {
    /// How long after the last good renewal the holder may still write.
    #[must_use]
    pub fn write_window(&self) -> Duration {
        self.ttl.saturating_sub(self.fence_margin)
    }
}

#[derive(Debug)]
struct Shared {
    lost: AtomicBool,
    cancel: AtomicBool,
    pause: AtomicBool,
    shutdown: AtomicBool,
    base: Instant,
    last_good_ms: AtomicU64,
    timing: LeaseTiming,
}

/// The handler's and the wrapper's shared view of one lease. Cloning shares it.
#[derive(Debug, Clone)]
pub struct LeaseWatch {
    shared: Arc<Shared>,
}

impl LeaseWatch {
    /// A watch for a lease taken now.
    #[must_use]
    pub fn new(timing: LeaseTiming) -> Self {
        Self {
            shared: Arc::new(Shared {
                lost: AtomicBool::new(false),
                cancel: AtomicBool::new(false),
                pause: AtomicBool::new(false),
                shutdown: AtomicBool::new(false),
                base: Instant::now(),
                last_good_ms: AtomicU64::new(0),
                timing,
            }),
        }
    }

    fn elapsed_ms(&self) -> u64 {
        u64::try_from(self.shared.base.elapsed().as_millis()).unwrap_or(u64::MAX)
    }

    /// Record a fenced write that the database accepted, with what it said about the row.
    pub fn note_good(&self, control: Control) {
        self.shared
            .last_good_ms
            .store(self.elapsed_ms(), Ordering::Relaxed);
        if control.cancel_requested {
            self.shared.cancel.store(true, Ordering::Relaxed);
        }
        if control.pause_requested {
            self.shared.pause.store(true, Ordering::Relaxed);
        }
    }

    /// The token no longer holds the row.
    pub fn note_lost(&self) {
        self.shared.lost.store(true, Ordering::Relaxed);
    }

    /// The runner is shutting down.
    pub fn note_shutdown(&self) {
        self.shared.shutdown.store(true, Ordering::Relaxed);
    }

    /// The handler asked for its own pause and the database recorded it.
    pub fn note_pause_requested(&self) {
        self.shared.pause.store(true, Ordering::Relaxed);
    }

    /// A cancel was requested.
    #[must_use]
    pub fn cancel_requested(&self) -> bool {
        self.shared.cancel.load(Ordering::Relaxed)
    }

    /// A pause was requested.
    #[must_use]
    pub fn pause_requested(&self) -> bool {
        self.shared.pause.load(Ordering::Relaxed)
    }

    /// The lease is lost: the job must stop, and its writes are refused anyway.
    #[must_use]
    pub fn lost(&self) -> bool {
        self.shared.lost.load(Ordering::Relaxed)
    }

    /// The runner is draining.
    #[must_use]
    pub fn shutting_down(&self) -> bool {
        self.shared.shutdown.load(Ordering::Relaxed)
    }

    /// Whether a write outside the database is still safe: not lost, and the last good renewal
    /// younger than the write window.
    #[must_use]
    pub fn may_write(&self) -> bool {
        if self.lost() {
            return false;
        }
        let age = self.elapsed_ms()
            - self
                .shared
                .last_good_ms
                .load(Ordering::Relaxed)
                .min(self.elapsed_ms());
        Duration::from_millis(age) < self.shared.timing.write_window()
    }
}

/// Renew `lease` every tick until it is lost, then return. The caller aborts the task when the job
/// ends.
///
/// A failed renewal (database unreachable, or no answer within one tick) is retried at the next
/// tick; once the last good renewal is older than the write window the lease counts as lost, so
/// the handler stops before the reaper can hand the job to another runner.
pub async fn heartbeat_loop<S: JobStore>(store: Arc<S>, lease: Lease, watch: LeaseWatch) {
    let timing = watch.shared.timing;
    let mut ticker = tokio::time::interval_at(Instant::now() + timing.tick, timing.tick);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        ticker.tick().await;
        let beat = tokio::time::timeout(timing.tick, store.heartbeat(&lease, timing.ttl)).await;
        match beat {
            Ok(Ok(Some(control))) => watch.note_good(control),
            Ok(Ok(None)) => {
                tracing::warn!(job = %lease.job_id, token = lease.fencing_token, "lease lost");
                watch.note_lost();
                return;
            }
            failure => {
                match failure {
                    Ok(Err(error)) => {
                        tracing::warn!(job = %lease.job_id, %error, "heartbeat failed");
                    }
                    _ => tracing::warn!(job = %lease.job_id, "heartbeat did not answer in time"),
                }
                if !watch.may_write() {
                    watch.note_lost();
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::fake::FakeStore;
    use crate::store::{ClaimRequest, JobStore};

    const TIMING: LeaseTiming = LeaseTiming {
        ttl: Duration::from_secs(120),
        tick: Duration::from_secs(15),
        fence_margin: Duration::from_secs(30),
    };

    async fn claimed(store: &Arc<FakeStore>) -> Lease {
        let id = store.insert("noop", serde_json::json!({}), None).unwrap();
        let jobs = store
            .claim(&ClaimRequest {
                kinds: vec!["noop".into()],
                limit: 1,
                owner: "w".into(),
                lease_ttl: TIMING.ttl,
            })
            .await
            .unwrap();
        assert_eq!(jobs[0].lease.job_id, id);
        jobs[0].lease
    }

    async fn pass(seconds: u64) {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
    }

    #[tokio::test(start_paused = true)]
    async fn the_heartbeat_keeps_the_lease_alive_and_carries_a_cancel_to_the_handler() {
        let store = Arc::new(FakeStore::new());
        let lease = claimed(&store).await;
        let watch = LeaseWatch::new(TIMING);
        let beat = tokio::spawn(heartbeat_loop(Arc::clone(&store), lease, watch.clone()));
        pass(100).await;
        assert!(store.reap_expired().is_empty());
        assert!(watch.may_write() && !watch.lost() && !watch.cancel_requested());
        store.cancel(lease.job_id);
        pass(14).await;
        assert!(
            watch.cancel_requested(),
            "seen at the next heartbeat, within one tick"
        );
        beat.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn a_token_that_no_longer_holds_the_row_ends_the_loop_with_the_lease_lost() {
        let store = Arc::new(FakeStore::new());
        let lease = claimed(&store).await;
        let watch = LeaseWatch::new(TIMING);
        let beat = tokio::spawn(heartbeat_loop(Arc::clone(&store), lease, watch.clone()));
        store.lose_and_requeue(lease.job_id);
        pass(16).await;
        assert!(watch.lost() && !watch.may_write());
        assert!(beat.is_finished(), "a lost lease stops renewing");
    }

    #[tokio::test(start_paused = true)]
    async fn a_database_outage_shorter_than_the_write_window_is_survived() {
        let store = Arc::new(FakeStore::new());
        let lease = claimed(&store).await;
        let watch = LeaseWatch::new(TIMING);
        let beat = tokio::spawn(heartbeat_loop(Arc::clone(&store), lease, watch.clone()));
        pass(20).await; // one good beat at 15 s
        store.down.store(true, Ordering::Relaxed);
        pass(60).await; // down from 20 s to 80 s: 65 s since the last good beat, window 90 s
        assert!(watch.may_write() && !watch.lost());
        store.down.store(false, Ordering::Relaxed);
        pass(16).await;
        assert!(watch.may_write() && !beat.is_finished());
        beat.abort();
    }

    #[tokio::test(start_paused = true)]
    async fn a_holder_that_cannot_reach_the_database_stops_writing_before_the_lease_can_expire() {
        let store = Arc::new(FakeStore::new());
        let lease = claimed(&store).await;
        let watch = LeaseWatch::new(TIMING);
        let beat = tokio::spawn(heartbeat_loop(Arc::clone(&store), lease, watch.clone()));
        pass(20).await;
        store.down.store(true, Ordering::Relaxed);
        pass(90).await; // 105 s since the last good beat at 15 s: past the 90 s window, before 120 s
        assert!(
            !watch.may_write(),
            "the self-fence closed before the lease could expire"
        );
        assert!(watch.lost());
        assert!(beat.is_finished());
    }
}
