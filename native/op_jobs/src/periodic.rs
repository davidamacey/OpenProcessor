//! The periodic framework: `run_periodic` runs ticks (the reaper, sweeps, the ticks that start
//! work as a queued job row) under a single-ticker lock.
//!
//! Exactly one process per database ticks: it holds the session advisory lock `BEAT`
//! (`src/tasks/beat_lock.py`, class from the generated contract) on a dedicated direct connection,
//! and probes that connection while it runs, so a lost connection ends the ticker (the server
//! frees the lock when the session dies) and compose restarts it. A second ticker finds the lock
//! held and returns [`PeriodicError::AlreadyRunning`] (the process exits with code 3, as beat did).
//!
//! Each tick runs on its own interval, never overlaps itself (an interval that was missed is not
//! made up: tokio's `MissedTickBehavior::Delay`), and has a time limit. A failing tick is logged and
//! tried again at the next interval. Time is tokio's, so tests run it on a paused clock.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use op_db::{DbError, DbSettings, SessionLock};
use tokio::sync::watch;
use tokio::task::JoinSet;
use tokio::time::{Instant, MissedTickBehavior};

use crate::kinds::advisory_lock;

/// What a tick's body returns on failure.
pub type TickError = Box<dyn std::error::Error + Send + Sync>;

type TickRun =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<(), TickError>> + Send>> + Send + Sync>;

/// One periodic task.
#[derive(Clone)]
pub struct Tick {
    name: &'static str,
    interval: Duration,
    timeout: Duration,
    run: TickRun,
}

impl Tick {
    /// A tick that runs `body` every `interval`, first after one interval, and is abandoned when a
    /// run lasts longer than `timeout`.
    pub fn new<F, Fut>(name: &'static str, interval: Duration, timeout: Duration, body: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), TickError>> + Send + 'static,
    {
        Self {
            name,
            interval,
            timeout,
            run: Arc::new(move || Box::pin(body())),
        }
    }
}

/// Why `run_periodic` ended without being told to stop.
#[derive(Debug, thiserror::Error)]
pub enum PeriodicError {
    /// Another process holds the single-ticker lock.
    #[error("another process holds the single-ticker lock; exiting")]
    AlreadyRunning,
    /// The lock's connection stopped answering: the lock is gone, and so is the right to tick.
    #[error("the single-ticker lock connection was lost: {0}")]
    LockLost(String),
    /// The lock could not be asked for.
    #[error(transparent)]
    Db(#[from] DbError),
    /// The contract has no `BEAT` advisory lock class.
    #[error("contracts/json/job_kinds.json has no BEAT advisory lock")]
    NoBeatLock,
}

/// The single-ticker lock, as a trait so the framework is tested without a database.
pub trait TickerLock: Send + 'static {
    /// Take the lock; `false` when another process holds it.
    ///
    /// # Errors
    ///
    /// When the database cannot be asked.
    fn acquire(&mut self) -> impl Future<Output = Result<bool, DbError>> + Send;

    /// Check that the lock is still held (its connection still answers).
    ///
    /// # Errors
    ///
    /// When it is not.
    fn probe(&self) -> impl Future<Output = Result<(), DbError>> + Send;
}

/// The real lock: `pg_try_advisory_lock(BEAT, 0)` on a dedicated direct connection.
pub struct PgTickerLock {
    settings: DbSettings,
    held: Option<SessionLock>,
}

impl PgTickerLock {
    /// A lock that connects with `settings` when asked.
    #[must_use]
    pub fn new(settings: DbSettings) -> Self {
        Self {
            settings,
            held: None,
        }
    }
}

impl TickerLock for PgTickerLock {
    async fn acquire(&mut self) -> Result<bool, DbError> {
        let class = advisory_lock("BEAT").ok_or(DbError::Data {
            what: "advisory_locks",
            detail: "the contract has no BEAT class".into(),
        })?;
        let name = format!("op-ticker-{}", std::process::id());
        self.held = SessionLock::try_acquire(&self.settings, &name, class).await?;
        Ok(self.held.is_some())
    }

    async fn probe(&self) -> Result<(), DbError> {
        match &self.held {
            Some(lock) => lock.probe().await,
            None => Err(DbError::Finished {
                what: "ticker lock probe",
            }),
        }
    }
}

/// How often the lock's connection is probed.
pub const PROBE_INTERVAL: Duration = Duration::from_secs(10);

/// Take the single-ticker lock and run `ticks` until `shutdown` turns true (then wait for running
/// ticks, up to their own time limits) or the lock is lost.
///
/// # Errors
///
/// [`PeriodicError::AlreadyRunning`] when another process ticks; [`PeriodicError::LockLost`] when
/// the lock's connection stops answering (the ticks are aborted).
pub async fn run_periodic<L: TickerLock>(
    mut lock: L,
    ticks: Vec<Tick>,
    mut shutdown: watch::Receiver<bool>,
) -> Result<(), PeriodicError> {
    if !lock.acquire().await? {
        return Err(PeriodicError::AlreadyRunning);
    }
    let mut running: JoinSet<()> = JoinSet::new();
    for tick in ticks {
        running.spawn(tick_loop(tick, shutdown.clone()));
    }
    let mut probe = tokio::time::interval_at(Instant::now() + PROBE_INTERVAL, PROBE_INTERVAL);
    probe.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    break;
                }
            }
            _ = probe.tick() => {
                if let Err(error) = lock.probe().await {
                    running.abort_all();
                    return Err(PeriodicError::LockLost(error.to_string()));
                }
            }
        }
    }
    while running.join_next().await.is_some() {}
    Ok(())
}

async fn tick_loop(tick: Tick, mut shutdown: watch::Receiver<bool>) {
    let mut ticker = tokio::time::interval_at(Instant::now() + tick.interval, tick.interval);
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = ticker.tick() => {}
            _ = shutdown.changed() => return,
        }
        match tokio::time::timeout(tick.timeout, (tick.run)()).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => tracing::warn!(tick = tick.name, %error, "a tick failed"),
            Err(_) => {
                tracing::warn!(tick = tick.name, limit = ?tick.timeout, "a tick ran out of time")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use super::*;

    struct FakeLock {
        free: bool,
        alive: Arc<AtomicBool>,
    }

    impl TickerLock for FakeLock {
        async fn acquire(&mut self) -> Result<bool, DbError> {
            Ok(self.free)
        }

        async fn probe(&self) -> Result<(), DbError> {
            if self.alive.load(Ordering::Relaxed) {
                Ok(())
            } else {
                Err(DbError::Finished { what: "fake probe" })
            }
        }
    }

    fn lock() -> (FakeLock, Arc<AtomicBool>) {
        let alive = Arc::new(AtomicBool::new(true));
        (
            FakeLock {
                free: true,
                alive: Arc::clone(&alive),
            },
            alive,
        )
    }

    fn counting(name: &'static str, every: u64) -> (Tick, Arc<AtomicUsize>) {
        let runs = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&runs);
        let tick = Tick::new(
            name,
            Duration::from_secs(every),
            Duration::from_secs(every),
            move || {
                let seen = Arc::clone(&seen);
                async move {
                    seen.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }
            },
        );
        (tick, runs)
    }

    async fn pass(seconds: u64) {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_second_ticker_finds_the_lock_held_and_does_not_tick() {
        let (mut taken, _) = lock();
        taken.free = false;
        let (tick, runs) = counting("t", 1);
        let (_stop, shutdown) = watch::channel(false);
        let result = run_periodic(taken, vec![tick], shutdown).await;
        assert!(matches!(result, Err(PeriodicError::AlreadyRunning)));
        pass(10).await;
        assert_eq!(runs.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn each_tick_runs_on_its_own_interval_starting_after_the_first_one() {
        let (lock, _) = lock();
        let (fast, fast_runs) = counting("fast", 30);
        let (slow, slow_runs) = counting("slow", 60);
        let (stop, shutdown) = watch::channel(false);
        let ticker = tokio::spawn(run_periodic(lock, vec![fast, slow], shutdown));
        pass(29).await;
        assert_eq!(
            fast_runs.load(Ordering::Relaxed),
            0,
            "not before one interval"
        );
        pass(72).await; // t = 101 s
        assert_eq!(fast_runs.load(Ordering::Relaxed), 3);
        assert_eq!(slow_runs.load(Ordering::Relaxed), 1);
        stop.send(true).unwrap();
        ticker.await.unwrap().unwrap();
        pass(120).await;
        assert_eq!(
            fast_runs.load(Ordering::Relaxed),
            3,
            "no tick after shutdown"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_tick_never_overlaps_itself_and_a_failing_tick_does_not_stop_the_others() {
        let (lock, _) = lock();
        let active = Arc::new(AtomicUsize::new(0));
        let overlapped = Arc::new(AtomicBool::new(false));
        let slow = {
            let (active, overlapped) = (Arc::clone(&active), Arc::clone(&overlapped));
            Tick::new(
                "slow",
                Duration::from_secs(10),
                Duration::from_secs(100),
                move || {
                    let (active, overlapped) = (Arc::clone(&active), Arc::clone(&overlapped));
                    async move {
                        if active.fetch_add(1, Ordering::SeqCst) > 0 {
                            overlapped.store(true, Ordering::SeqCst);
                        }
                        tokio::time::sleep(Duration::from_secs(25)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok(())
                    }
                },
            )
        };
        let failing = Tick::new(
            "failing",
            Duration::from_secs(10),
            Duration::from_secs(10),
            || async { Err::<(), TickError>("boom".into()) },
        );
        let (healthy, healthy_runs) = counting("healthy", 10);
        let (stop, shutdown) = watch::channel(false);
        let ticker = tokio::spawn(run_periodic(lock, vec![slow, failing, healthy], shutdown));
        pass(205).await;
        assert!(!overlapped.load(Ordering::SeqCst));
        assert_eq!(healthy_runs.load(Ordering::Relaxed), 20);
        stop.send(true).unwrap();
        ticker.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn a_tick_that_runs_past_its_time_limit_is_abandoned_and_tried_again() {
        let (lock, _) = lock();
        let starts = Arc::new(AtomicUsize::new(0));
        let hanging = {
            let starts = Arc::clone(&starts);
            Tick::new(
                "hang",
                Duration::from_secs(30),
                Duration::from_secs(10),
                move || {
                    let starts = Arc::clone(&starts);
                    async move {
                        starts.fetch_add(1, Ordering::Relaxed);
                        std::future::pending().await
                    }
                },
            )
        };
        let (stop, shutdown) = watch::channel(false);
        let ticker = tokio::spawn(run_periodic(lock, vec![hanging], shutdown));
        pass(95).await;
        assert_eq!(starts.load(Ordering::Relaxed), 3);
        stop.send(true).unwrap();
        ticker.await.unwrap().unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn a_lost_lock_connection_ends_the_ticker_and_its_ticks_within_one_probe() {
        let (lock, alive) = lock();
        let (tick, runs) = counting("t", 1);
        let (_stop, shutdown) = watch::channel(false);
        let ticker = tokio::spawn(run_periodic(lock, vec![tick], shutdown));
        pass(5).await;
        let before = runs.load(Ordering::Relaxed);
        assert!(before >= 4);
        alive.store(false, Ordering::Relaxed);
        pass(PROBE_INTERVAL.as_secs()).await;
        let result = ticker.await.unwrap();
        assert!(
            matches!(result, Err(PeriodicError::LockLost(_))),
            "{result:?}"
        );
        let after = runs.load(Ordering::Relaxed);
        pass(30).await;
        assert_eq!(
            runs.load(Ordering::Relaxed),
            after,
            "no tick runs without the lock"
        );
    }
}
