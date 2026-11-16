//! The worker: `run_worker` claims jobs of the kinds it has handlers for and runs them.
//!
//! Wake-ups: `LISTEN op_jobs` on the dedicated connection (`op_db::Listener`) plus a poll every
//! `poll_interval` (two seconds by default), because a notification is only delivered to a session
//! that is listening when the notifying transaction commits; after every reconnect the runner polls
//! at once. A job that finishes frees a slot and polls at once too.
//!
//! The number of running jobs is bounded by `max_running`, with one semaphore permit per job; a
//! claim asks for no more rows than there are free permits, so nothing is claimed that cannot run.
//! Each running job has its own heartbeat and its handler task, both aborted when the job's task is
//! dropped, so a worker that is killed or whose future is dropped leaves nothing renewing a lease.
//!
//! Shutdown (`shutdown` turns true): claiming stops; every running handler is told through
//! [`JobContext::shutting_down`] and has `drain` to reach a boundary and return; one that does not
//! is aborted. Either way the job is written `interrupted` (a `cancelling` job `cancelled`), which
//! spends a retry, and the reaper or a person resumes it (plan 02b section 3.1, step 8).

use std::sync::Arc;
use std::time::Duration;

use op_config::{Env, EnvError};
use op_db::ListenEvents;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio::task::{JoinHandle, JoinSet};
use tokio::time::{Instant, MissedTickBehavior};

use crate::handler::{ContextParts, Handlers, JobContext, Outcome, Registered};
use crate::kinds::{KindSpec, contract};
use crate::retry::through_outage;
use crate::store::{ClaimRequest, ClaimedJob, Conclusion, JobStore, Lease};
use crate::terminal::{Flags, RunEnd, plan_conclusion};
use crate::watch::{LeaseTiming, LeaseWatch, heartbeat_loop};

/// How a worker runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunnerConfig {
    /// Who is claiming (`host:pid`), stored in `jobs.owner`.
    pub owner: String,
    /// The most jobs running at once.
    pub max_running: usize,
    /// How often to poll when no notification arrives.
    pub poll_interval: Duration,
    /// The lease timing.
    pub timing: LeaseTiming,
    /// At most one progress write per job per this long, except on a stage change.
    pub progress_min_interval: Duration,
    /// How long a running handler gets to reach a boundary after shutdown starts.
    pub drain: Duration,
}

fn at_least(key: &str, value: u64, minimum: u64) -> Result<u64, EnvError> {
    if value >= minimum {
        Ok(value)
    } else {
        Err(EnvError::Invalid {
            key: key.into(),
            value: value.to_string(),
            reason: format!("must be at least {minimum}"),
        })
    }
}

impl RunnerConfig {
    /// Read the settings. The lease variables are the Python side's (`OP_JOB_LEASE_TTL_S`,
    /// `OP_JOB_HEARTBEAT_TICK_S`, `OP_JOB_FENCE_MARGIN_S`, defaults from the contract); the others
    /// are `OP_JOB_MAX_RUNNING` (4), `OP_JOB_POLL_INTERVAL_S` (2) and `OP_JOB_DRAIN_S` (30).
    ///
    /// # Errors
    ///
    /// An [`EnvError`] naming the variable when a value does not parse, is below its minimum, or
    /// leaves no room to renew (a fence margin that is not shorter than the lease).
    pub fn from_env(env: &Env, owner: String) -> Result<Self, EnvError> {
        let defaults = &contract().timing;
        let ttl = at_least(
            "OP_JOB_LEASE_TTL_S",
            env.get("OP_JOB_LEASE_TTL_S", defaults.lease_ttl_s)?,
            10,
        )?;
        let tick = at_least(
            "OP_JOB_HEARTBEAT_TICK_S",
            env.get("OP_JOB_HEARTBEAT_TICK_S", defaults.heartbeat_tick_s)?,
            1,
        )?;
        let margin = at_least(
            "OP_JOB_FENCE_MARGIN_S",
            env.get("OP_JOB_FENCE_MARGIN_S", defaults.fence_margin_s)?,
            1,
        )?;
        if margin >= ttl || tick >= ttl - margin {
            return Err(EnvError::Invalid {
                key: "OP_JOB_FENCE_MARGIN_S".into(),
                value: margin.to_string(),
                reason: format!(
                    "the lease ({ttl} s) minus the margin must be longer than one heartbeat tick ({tick} s)"
                ),
            });
        }
        let max_running = at_least("OP_JOB_MAX_RUNNING", env.get("OP_JOB_MAX_RUNNING", 4)?, 1)?;
        Ok(Self {
            owner,
            max_running: usize::try_from(max_running).unwrap_or(usize::MAX),
            poll_interval: Duration::from_secs(at_least(
                "OP_JOB_POLL_INTERVAL_S",
                env.get("OP_JOB_POLL_INTERVAL_S", 2)?,
                1,
            )?),
            timing: LeaseTiming {
                ttl: Duration::from_secs(ttl),
                tick: Duration::from_secs(tick),
                fence_margin: Duration::from_secs(margin),
            },
            progress_min_interval: Duration::from_secs(contract().timing.progress_min_interval_s),
            drain: Duration::from_secs(env.get("OP_JOB_DRAIN_S", 30)?),
        })
    }
}

/// The name a worker puts in `jobs.owner`: the host name and the process id.
#[must_use]
pub fn default_owner() -> String {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|h| h.trim().to_owned())
        .ok()
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "unknown-host".to_owned());
    format!("{host}:{}", std::process::id())
}

/// A task that is aborted when its guard is dropped.
struct AbortOnDrop<T>(JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// The claim loop's state: the slots, the running jobs, and what a claim needs.
struct Worker<S: JobStore> {
    store: Arc<S>,
    handlers: Arc<Handlers<S>>,
    config: Arc<RunnerConfig>,
    kinds: Vec<String>,
    slots: Arc<Semaphore>,
    running: JoinSet<()>,
    shutdown: watch::Receiver<bool>,
}

impl<S: JobStore> Worker<S> {
    /// Claim as many jobs as there are free slots and start them. Returns whether the claim was
    /// full, so more may be waiting and the next claim should not wait for a wake-up.
    async fn claim_and_spawn(&mut self) -> bool {
        let free = self.slots.available_permits();
        if free == 0 || self.kinds.is_empty() {
            return false;
        }
        let request = ClaimRequest {
            kinds: self.kinds.clone(),
            limit: free,
            owner: self.config.owner.clone(),
            lease_ttl: self.config.timing.ttl,
        };
        let jobs = match self.store.claim(&request).await {
            Ok(jobs) => jobs,
            Err(error) => {
                tracing::warn!(%error, "the claim failed; trying again at the next poll");
                return false;
            }
        };
        let full = jobs.len() == free;
        for job in jobs {
            let Ok(slot) = Arc::clone(&self.slots).acquire_owned().await else {
                return false;
            };
            self.running.spawn(run_job(
                Arc::clone(&self.store),
                Arc::clone(&self.handlers),
                job,
                Arc::clone(&self.config),
                self.shutdown.clone(),
                slot,
            ));
        }
        full
    }

    fn note_finished(done: Result<(), tokio::task::JoinError>) {
        if let Err(error) = done {
            tracing::error!(%error, "a job's task ended abnormally");
        }
    }

    async fn drain(&mut self) {
        while let Some(done) = self.running.join_next().await {
            Self::note_finished(done);
        }
    }
}

/// Claim and run jobs until `shutdown` turns true, then drain.
///
/// `wakeups` is the `LISTEN` connection (an `op_db::Listener`), or any source of
/// [`op_db::ListenerEvent`]s. Returns when the loop has stopped claiming and every running job has been
/// written.
pub async fn run_worker<S, W>(
    store: Arc<S>,
    handlers: Handlers<S>,
    config: RunnerConfig,
    mut wakeups: W,
    mut shutdown: watch::Receiver<bool>,
) where
    S: JobStore,
    W: ListenEvents,
{
    let mut poll = tokio::time::interval(config.poll_interval);
    poll.set_missed_tick_behavior(MissedTickBehavior::Delay);
    let mut worker = Worker {
        kinds: handlers.kinds(),
        slots: Arc::new(Semaphore::new(config.max_running)),
        handlers: Arc::new(handlers),
        config: Arc::new(config),
        store,
        running: JoinSet::new(),
        shutdown: shutdown.clone(),
    };
    let mut claim_now = true;
    let mut listening = true;
    while !*shutdown.borrow() {
        if claim_now {
            claim_now = worker.claim_and_spawn().await;
        }
        tokio::select! {
            changed = shutdown.changed() => if changed.is_err() { break },
            event = wakeups.recv(), if listening => {
                listening = event.is_some();
                claim_now = true;
            },
            _ = poll.tick() => claim_now = true,
            Some(done) = worker.running.join_next(), if !worker.running.is_empty() => {
                Worker::<S>::note_finished(done);
                claim_now = true;
            }
        }
    }
    worker.drain().await;
}

async fn run_job<S: JobStore>(
    store: Arc<S>,
    handlers: Arc<Handlers<S>>,
    job: ClaimedJob,
    config: Arc<RunnerConfig>,
    mut shutdown: watch::Receiver<bool>,
    _slot: OwnedSemaphorePermit,
) {
    let message_limit = contract().wrapper.message_chars;
    let Some(registered) = handlers.get(&job.kind) else {
        let refusal = Conclusion::Finish {
            status: crate::state::JobStatus::Failed,
            outcome: crate::state::AttemptOutcome::Failed,
            result: None,
            error_code: Some("kind_not_registered".into()),
            error_message: Some(format!("this worker has no handler for {}", job.kind)),
        };
        write_conclusion(&*store, &job.lease, &refusal, config.timing).await;
        return;
    };
    let watch = LeaseWatch::new(config.timing);
    let _heartbeat = AbortOnDrop(tokio::spawn(heartbeat_loop(
        Arc::clone(&store),
        job.lease,
        watch.clone(),
    )));
    let ctx = JobContext::new(ContextParts {
        store: Arc::clone(&store),
        lease: job.lease,
        kind: registered.spec,
        params: job.params.clone(),
        resume_from: job.checkpoint.clone(),
        watch: watch.clone(),
        progress_min_interval: config.progress_min_interval,
        outage_budget: config.timing.write_window(),
    });
    let handler = AbortOnDrop(tokio::spawn((registered.run)(ctx.clone())));
    let end = drive(handler, registered, &watch, &mut shutdown, config.drain).await;
    if matches!(end, RunEnd::Returned(_)) {
        ctx.flush_progress().await;
    }
    let flags = Flags {
        cancel_requested: watch.cancel_requested(),
        pause_requested: watch.pause_requested(),
    };
    let conclusion = plan_conclusion(registered.spec, &end, flags, message_limit);
    write_conclusion(&*store, &job.lease, &conclusion, config.timing).await;
}

/// Wait for the handler, the soft time limit and the drain. The soft timer fires a little early
/// (two seconds, at most a tenth of the limit) so the terminal write lands before the limit.
async fn drive<S: JobStore>(
    mut handler: AbortOnDrop<Outcome>,
    registered: &Registered<S>,
    watch: &LeaseWatch,
    shutdown: &mut watch::Receiver<bool>,
    drain: Duration,
) -> RunEnd {
    let spec: &KindSpec = registered.spec;
    let soft = spec.soft_limit();
    let lead = Duration::from_secs_f64(contract().wrapper.soft_limit_lead_s).min(soft / 10);
    let soft_timer = tokio::time::sleep(soft - lead);
    tokio::pin!(soft_timer);
    let mut drain_at: Option<Instant> = None;
    if *shutdown.borrow() {
        watch.note_shutdown();
        drain_at = Some(Instant::now() + drain);
    }
    loop {
        let drained = async {
            match drain_at {
                Some(at) => tokio::time::sleep_until(at).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            joined = &mut handler.0 => return end_of(joined),
            () = &mut soft_timer => return RunEnd::TimeLimit,
            _ = shutdown.changed(), if drain_at.is_none() => {
                watch.note_shutdown();
                drain_at = Some(Instant::now() + drain);
            },
            () = drained => return RunEnd::DrainExpired,
        }
    }
}

fn end_of(joined: Result<Outcome, tokio::task::JoinError>) -> RunEnd {
    match joined {
        Ok(outcome) => RunEnd::Returned(outcome),
        Err(error) if error.is_panic() => {
            let payload = error.into_panic();
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic with a payload that is not text".to_owned());
            RunEnd::Panicked(message)
        }
        Err(_) => RunEnd::Panicked("the handler task was cancelled".to_owned()),
    }
}

async fn write_conclusion<S: JobStore>(
    store: &S,
    lease: &Lease,
    conclusion: &Conclusion,
    timing: LeaseTiming,
) {
    let written = through_outage(timing.write_window(), || store.conclude(lease, conclusion)).await;
    match written {
        Ok(Some(status)) => tracing::debug!(job = %lease.job_id, %status, "job written"),
        Ok(None) => {
            tracing::warn!(job = %lease.job_id, token = lease.fencing_token, "lease lost; nothing written");
        }
        Err(error) => tracing::error!(
            job = %lease.job_id, %error,
            "the terminal write failed; the lease will expire and the reaper decides"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use serde_json::{Map, json};
    use tokio::sync::{Notify, mpsc};

    use op_db::ListenerEvent;

    use super::*;
    use crate::fake::FakeStore;
    use crate::state::JobStatus;
    use crate::store::JobId;

    fn config(max_running: usize) -> RunnerConfig {
        RunnerConfig {
            owner: "w1".into(),
            max_running,
            poll_interval: Duration::from_secs(2),
            timing: LeaseTiming {
                ttl: Duration::from_secs(120),
                tick: Duration::from_secs(15),
                fence_margin: Duration::from_secs(30),
            },
            progress_min_interval: Duration::from_secs(1),
            drain: Duration::from_secs(30),
        }
    }

    struct Running {
        task: JoinHandle<()>,
        wake: mpsc::Sender<ListenerEvent>,
        stop: watch::Sender<bool>,
    }

    fn start(store: &Arc<FakeStore>, handlers: Handlers<FakeStore>, cfg: RunnerConfig) -> Running {
        let (wake, events) = mpsc::channel(8);
        let (stop, shutdown) = watch::channel(false);
        let task = tokio::spawn(run_worker(
            Arc::clone(store),
            handlers,
            cfg,
            events,
            shutdown,
        ));
        Running { task, wake, stop }
    }

    fn done() -> Outcome {
        Outcome::Completed(Map::new())
    }

    fn handlers_of<F, Fut>(kind: &str, run: F) -> Handlers<FakeStore>
    where
        F: Fn(JobContext<FakeStore>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Outcome> + Send + 'static,
    {
        let mut handlers = Handlers::new();
        handlers.register(kind, run).unwrap();
        handlers
    }

    fn handlers_noop<F, Fut>(run: F) -> Handlers<FakeStore>
    where
        F: Fn(JobContext<FakeStore>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Outcome> + Send + 'static,
    {
        handlers_of("noop", run)
    }

    async fn pass(seconds: u64) {
        tokio::time::sleep(Duration::from_secs(seconds)).await;
    }

    fn queued(store: &FakeStore) -> JobId {
        store.insert("noop", json!({}), None).unwrap()
    }

    #[tokio::test(start_paused = true)]
    async fn a_queued_job_is_claimed_run_and_completed_with_its_attempt_recorded() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        let _worker = start(&store, handlers_noop(|_| async { done() }), config(2));
        pass(1).await;
        let job = store.job(id);
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!((job.attempt, job.fencing_token), (1, 1));
        assert_eq!(job.attempts, vec![Some("completed".to_owned())]);
        assert!(job.owner.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn the_number_of_running_jobs_is_bounded_and_a_finished_job_frees_its_slot_at_once() {
        let store = Arc::new(FakeStore::new());
        let ids: Vec<JobId> = (0..3).map(|_| queued(&store)).collect();
        let gate = Arc::new(Notify::new());
        let handlers = {
            let gate = Arc::clone(&gate);
            handlers_noop(move |_| {
                let gate = Arc::clone(&gate);
                async move {
                    gate.notified().await;
                    done()
                }
            })
        };
        let _worker = start(&store, handlers, config(2));
        pass(5).await;
        assert_eq!(store.running_count(), 2, "two slots, three jobs");
        assert_eq!(store.status(ids[2]), JobStatus::Queued);
        gate.notify_one();
        pass(1).await;
        assert_eq!(
            store.running_count(),
            2,
            "the freed slot took the third job without waiting for a poll"
        );
        assert_eq!(store.status(ids[2]), JobStatus::Running);
    }

    #[tokio::test(start_paused = true)]
    async fn a_notification_claims_at_once_and_without_one_the_poll_claims_within_its_interval() {
        let store = Arc::new(FakeStore::new());
        let worker = start(&store, handlers_noop(|_| async { done() }), config(2));
        pass(3).await; // the first claim found nothing
        let id = queued(&store);
        worker
            .wake
            .send(ListenerEvent::Notification(op_db::Notification {
                channel: "op_jobs".into(),
                payload: String::new(),
            }))
            .await
            .unwrap();
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert_eq!(
            store.status(id),
            JobStatus::Completed,
            "claimed and run within 5 ms of the notification"
        );
        let late = queued(&store);
        pass(2).await;
        assert_eq!(
            store.status(late),
            JobStatus::Completed,
            "no notification: the 2 s poll found it"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn after_a_reconnect_the_runner_polls_at_once_because_a_notification_may_have_been_lost()
    {
        let store = Arc::new(FakeStore::new());
        let mut cfg = config(2);
        cfg.poll_interval = Duration::from_secs(3600);
        let worker = start(&store, handlers_noop(|_| async { done() }), cfg);
        pass(1).await;
        let id = queued(&store); // its notification was lost with the connection
        pass(10).await;
        assert_eq!(
            store.status(id),
            JobStatus::Queued,
            "nothing wakes the runner for an hour"
        );
        worker.wake.send(ListenerEvent::Reconnected).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert_eq!(store.status(id), JobStatus::Completed);
    }

    #[tokio::test(start_paused = true)]
    async fn the_lease_is_renewed_while_a_job_outlives_it_and_no_reaper_can_take_it() {
        let store = Arc::new(FakeStore::new());
        let id = store.insert("sleep_noop", json!({}), None).unwrap();
        let handlers = handlers_of("sleep_noop", |_| async {
            tokio::time::sleep(Duration::from_secs(400)).await;
            done()
        });
        let _worker = start(&store, handlers, config(1));
        for _ in 0..45 {
            pass(10).await;
            assert!(
                store.reap_expired().is_empty(),
                "the lease expired while the job was alive"
            );
        }
        assert_eq!(store.status(id), JobStatus::Completed);
        assert!(store.heartbeats.load(Ordering::Relaxed) >= 20);
    }

    #[tokio::test(start_paused = true)]
    async fn a_cancel_is_seen_at_the_next_heartbeat_and_the_job_ends_cancelled() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        let handlers = handlers_noop(|ctx| async move {
            while !ctx.cancelled() {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            done()
        });
        let _worker = start(&store, handlers, config(1));
        pass(5).await;
        assert_eq!(store.cancel(id), JobStatus::Cancelling);
        pass(9).await;
        assert_eq!(
            store.status(id),
            JobStatus::Cancelling,
            "the heartbeat has not fired yet"
        );
        pass(2).await;
        let job = store.job(id);
        assert_eq!(job.status, JobStatus::Cancelled);
        assert_eq!(job.attempts, vec![Some("cancelled".to_owned())]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_killed_worker_loses_its_job_and_its_late_writes_are_refused_by_the_stale_token() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        let hang = || handlers_noop(|_| async { std::future::pending().await });
        let dead = start(&store, hang(), config(1));
        pass(1).await;
        let old = Lease {
            job_id: id,
            project_id: None,
            fencing_token: store.job(id).fencing_token,
            attempt: 1,
        };
        dead.task.abort(); // the process dies: no terminal write, no more heartbeats
        pass(1).await;
        let beats = store.heartbeats.load(Ordering::Relaxed);
        pass(60).await;
        assert_eq!(
            store.heartbeats.load(Ordering::Relaxed),
            beats,
            "nothing renews a dead worker's lease"
        );
        assert!(store.reap_expired().is_empty(), "not expired yet");
        pass(70).await;
        assert_eq!(store.reap_expired(), vec![id]);
        let after = store.job(id);
        assert_eq!(
            (after.status, after.retry_count),
            (JobStatus::Interrupted, 1)
        );
        assert_eq!(after.attempts, vec![Some("lost".to_owned())]);
        store.resume(id).unwrap();
        let mut second = config(1);
        second.owner = "w2".into();
        let _taker = start(&store, hang(), second);
        pass(1).await;
        let taken = store.job(id);
        assert_eq!(
            (taken.status, taken.attempt, taken.fencing_token),
            (JobStatus::Running, 2, 2)
        );
        assert_eq!(taken.owner.as_deref(), Some("w2"));
        // The dead worker's late writes, with its old token:
        let finish = Conclusion::Finish {
            status: JobStatus::Completed,
            outcome: crate::state::AttemptOutcome::Completed,
            result: None,
            error_code: None,
            error_message: None,
        };
        assert_eq!(store.conclude(&old, &finish).await.unwrap(), None);
        assert!(
            store
                .heartbeat(&old, Duration::from_secs(120))
                .await
                .unwrap()
                .is_none()
        );
        assert!(store.progress(&old, &json!({})).await.unwrap().is_none());
        assert!(
            store
                .checkpoint(&old, Some(&json!({"x": 1})))
                .await
                .unwrap()
                .is_none()
        );
        let still = store.job(id);
        assert_eq!(
            (still.status, still.owner.as_deref()),
            (JobStatus::Running, Some("w2"))
        );
        assert!(still.checkpoint.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn a_holder_that_is_taken_over_sees_it_and_its_final_write_is_refused() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        let saw_lost = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let handlers = {
            let saw_lost = Arc::clone(&saw_lost);
            handlers_noop(move |ctx| {
                let saw_lost = Arc::clone(&saw_lost);
                async move {
                    while !ctx.cancelled() {
                        tokio::time::sleep(Duration::from_secs(1)).await;
                    }
                    saw_lost.store(ctx.lost(), Ordering::Relaxed);
                    done()
                }
            })
        };
        let _worker = start(&store, handlers, config(1));
        pass(5).await;
        store.lose_and_requeue(id); // another runner is about to own it
        let taker = store
            .claim(&ClaimRequest {
                kinds: vec!["noop".into()],
                limit: 1,
                owner: "w2".into(),
                lease_ttl: Duration::from_secs(120),
            })
            .await
            .unwrap();
        assert_eq!(taker[0].lease.fencing_token, 2);
        pass(20).await;
        assert!(
            saw_lost.load(Ordering::Relaxed),
            "the heartbeat found the lease lost"
        );
        let job = store.job(id);
        assert_eq!(
            (job.status, job.owner.as_deref()),
            (JobStatus::Running, Some("w2"))
        );
        assert_eq!(
            job.attempts.last().unwrap(),
            &None,
            "the first worker wrote nothing over the new attempt"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_shutdown_stops_claiming_and_a_handler_that_reaches_a_boundary_interrupts_its_job() {
        let store = Arc::new(FakeStore::new());
        let first = queued(&store);
        let handlers = handlers_noop(|ctx| async move {
            while !ctx.shutting_down() {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Outcome::Interrupted
        });
        let worker = start(&store, handlers, config(1));
        pass(3).await;
        let second = queued(&store);
        worker.stop.send(true).unwrap();
        pass(2).await;
        worker.task.await.unwrap();
        let job = store.job(first);
        assert_eq!((job.status, job.retry_count), (JobStatus::Interrupted, 1));
        assert_eq!(job.attempts, vec![Some("interrupted".to_owned())]);
        assert_eq!(
            store.status(second),
            JobStatus::Queued,
            "a draining worker claims nothing"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_handler_that_ignores_the_shutdown_is_aborted_after_the_drain_and_interrupted() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        let worker = start(
            &store,
            handlers_noop(|_| async { std::future::pending().await }),
            config(1),
        );
        pass(2).await;
        worker.stop.send(true).unwrap();
        pass(29).await;
        assert_eq!(
            store.status(id),
            JobStatus::Running,
            "still inside the 30 s drain"
        );
        pass(2).await;
        worker.task.await.unwrap();
        assert_eq!(store.status(id), JobStatus::Interrupted);
    }

    #[tokio::test(start_paused = true)]
    async fn the_soft_time_limit_fails_the_job_and_spends_a_retry() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        let _worker = start(
            &store,
            handlers_noop(|_| async { std::future::pending().await }),
            config(1),
        );
        pass(57).await;
        assert_eq!(
            store.status(id),
            JobStatus::Running,
            "noop's soft limit is 60 s, fired 2 s early"
        );
        pass(3).await;
        let job = store.job(id);
        assert_eq!((job.status, job.retry_count), (JobStatus::Failed, 1));
        assert_eq!(job.error_code.as_deref(), Some("time_limit"));
        assert_eq!(job.attempts, vec![Some("time_limit".to_owned())]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_panic_in_a_handler_fails_that_job_and_the_worker_goes_on() {
        let store = Arc::new(FakeStore::new());
        let bad = store.insert("noop", json!({"bad": true}), None).unwrap();
        let _worker = start(
            &store,
            handlers_noop(|ctx| async move {
                assert!(ctx.params().get("bad").is_none(), "boom");
                done()
            }),
            config(1),
        );
        pass(1).await;
        let job = store.job(bad);
        assert_eq!(job.status, JobStatus::Failed);
        assert_eq!(job.error_code.as_deref(), Some("handler_error"));
        let next = queued(&store);
        pass(3).await;
        assert_eq!(store.status(next), JobStatus::Completed);
    }

    #[tokio::test(start_paused = true)]
    async fn a_failed_claim_does_not_stop_the_worker_and_the_next_poll_recovers() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        store.down.store(true, Ordering::Relaxed);
        let _worker = start(&store, handlers_noop(|_| async { done() }), config(1));
        pass(7).await;
        assert_eq!(store.status(id), JobStatus::Queued);
        store.down.store(false, Ordering::Relaxed);
        pass(2).await;
        assert_eq!(store.status(id), JobStatus::Completed);
    }

    #[tokio::test(start_paused = true)]
    async fn the_terminal_write_waits_out_a_short_database_outage() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        store.conclude_failures.store(3, Ordering::Relaxed);
        let _worker = start(&store, handlers_noop(|_| async { done() }), config(1));
        pass(1).await;
        assert_eq!(
            store.status(id),
            JobStatus::Running,
            "the first writes failed; it is waiting"
        );
        pass(10).await;
        assert_eq!(store.status(id), JobStatus::Completed);
    }

    #[tokio::test(start_paused = true)]
    async fn progress_is_written_once_a_second_except_on_a_stage_change_and_the_last_is_flushed() {
        let store = Arc::new(FakeStore::new());
        let id = queued(&store);
        let handlers = handlers_noop(|ctx| async move {
            for done_n in 0..5 {
                ctx.progress(done_n, Some(5), Some("a"), json!({})).await;
            }
            ctx.progress(5, Some(5), Some("b"), json!({})).await;
            ctx.progress(6, Some(6), Some("b"), json!({})).await;
            done()
        });
        let _worker = start(&store, handlers, config(1));
        pass(2).await;
        assert_eq!(store.status(id), JobStatus::Completed);
        // first call, the stage change, and the flush of the throttled last snapshot
        assert_eq!(store.progress_writes.load(Ordering::Relaxed), 3);
        assert_eq!(store.job(id).progress.unwrap()["done"], 6);
    }

    #[test]
    fn the_settings_are_read_with_the_python_names_and_bad_ones_name_their_variable() {
        let env = Env::from_pairs([("OP_JOB_MAX_RUNNING", "8"), ("OP_JOB_LEASE_TTL_S", "60")]);
        let cfg = RunnerConfig::from_env(&env, "h:1".into()).unwrap();
        assert_eq!(cfg.max_running, 8);
        assert_eq!(cfg.timing.ttl, Duration::from_secs(60));
        assert_eq!(cfg.timing.tick, Duration::from_secs(15));
        for (key, value) in [
            ("OP_JOB_MAX_RUNNING", "0"),
            ("OP_JOB_LEASE_TTL_S", "5"),
            ("OP_JOB_HEARTBEAT_TICK_S", "0"),
            ("OP_JOB_FENCE_MARGIN_S", "120"),
            ("OP_JOB_POLL_INTERVAL_S", "x"),
        ] {
            let err =
                RunnerConfig::from_env(&Env::from_pairs([(key, value)]), "h:1".into()).unwrap_err();
            assert!(err.to_string().contains(key), "{key}: {err}");
        }
        let tick_too_long = Env::from_pairs([("OP_JOB_HEARTBEAT_TICK_S", "90")]);
        assert!(RunnerConfig::from_env(&tick_too_long, "h:1".into()).is_err());
    }
}
