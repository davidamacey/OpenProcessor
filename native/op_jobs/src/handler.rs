//! What a job handler gets ([`JobContext`]), what it may return ([`Outcome`]), and the registry of
//! handlers a process runs ([`Handlers`]).
//!
//! Control is cooperative: [`JobContext::cancelled`], [`JobContext::paused`] and
//! [`JobContext::shutting_down`] are plain reads of what the heartbeat and every fenced write keep
//! current. The handler checks them at a boundary and returns; nothing is interrupted from outside
//! except at the kind's soft time limit, and a handler never blocks the async executor: CPU work
//! goes to `spawn_blocking` or a dedicated pool, as tokio's documentation says.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Map, Value, json};
use tokio::time::Instant;

use crate::kinds::{KindSpec, Runtime, kind};
use crate::retry::through_outage;
use crate::state::PauseReason;
use crate::store::{Control, JobId, JobStore, Lease};
use crate::watch::LeaseWatch;

/// How a handler's run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Done: the result object (`completed`).
    Completed(Map<String, Value>),
    /// Done, with errors recorded in the result (`completed_with_errors`).
    CompletedWithErrors(Map<String, Value>),
    /// Stopped at a boundary after the kind's slice: the job goes back to `queued`, behind other
    /// projects' jobs, and keeps its retry budget.
    Yield,
    /// Stopped at a boundary after seeing [`JobContext::paused`]; the reason is the handler's own,
    /// or none to keep the one requested.
    Paused(Option<PauseReason>),
    /// Stopped at a boundary because the worker is shutting down: the job is `interrupted` and
    /// can be resumed.
    Interrupted,
    /// Stopped because [`JobContext::cancelled`] was true.
    Cancelled,
    /// Failed, with a stable code the client can act on.
    Failed {
        /// The `error_code` the job is served with.
        code: String,
        /// The `error_message`.
        message: String,
    },
}

impl Outcome {
    /// A failure with a code and a message.
    #[must_use]
    pub fn failed(code: &str, message: impl Into<String>) -> Self {
        Self::Failed {
            code: code.to_owned(),
            message: message.into(),
        }
    }
}

struct ProgressState {
    last_at: Option<Instant>,
    last_stage: Option<String>,
    unwritten: Option<Value>,
}

struct Inner<S: JobStore> {
    store: Arc<S>,
    lease: Lease,
    kind: &'static KindSpec,
    params: Value,
    resume_from: Option<Value>,
    watch: LeaseWatch,
    started: Instant,
    progress_min_interval: Duration,
    outage_budget: Duration,
    progress: Mutex<ProgressState>,
}

/// What the wrapper builds a [`JobContext`] from.
pub(crate) struct ContextParts<S: JobStore> {
    pub(crate) store: Arc<S>,
    pub(crate) lease: Lease,
    pub(crate) kind: &'static KindSpec,
    pub(crate) params: Value,
    pub(crate) resume_from: Option<Value>,
    pub(crate) watch: LeaseWatch,
    pub(crate) progress_min_interval: Duration,
    pub(crate) outage_budget: Duration,
}

/// The handler's view of one claimed job. Cloning shares it.
pub struct JobContext<S: JobStore> {
    inner: Arc<Inner<S>>,
}

impl<S: JobStore> Clone for JobContext<S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<S: JobStore> JobContext<S> {
    pub(crate) fn new(parts: ContextParts<S>) -> Self {
        Self {
            inner: Arc::new(Inner {
                store: parts.store,
                lease: parts.lease,
                kind: parts.kind,
                params: parts.params,
                resume_from: parts.resume_from,
                watch: parts.watch,
                started: Instant::now(),
                progress_min_interval: parts.progress_min_interval,
                outage_budget: parts.outage_budget,
                progress: Mutex::new(ProgressState {
                    last_at: None,
                    last_stage: None,
                    unwritten: None,
                }),
            }),
        }
    }

    /// The job's id.
    #[must_use]
    pub fn job_id(&self) -> JobId {
        self.inner.lease.job_id
    }

    /// The kind's record.
    #[must_use]
    pub fn kind(&self) -> &'static KindSpec {
        self.inner.kind
    }

    /// The parameters the job was started with.
    #[must_use]
    pub fn params(&self) -> &Value {
        &self.inner.params
    }

    /// The cursor the previous attempt last wrote with [`JobContext::checkpoint`].
    #[must_use]
    pub fn resume_from(&self) -> Option<&Value> {
        self.inner.resume_from.as_ref()
    }

    /// The attempt number of this run.
    #[must_use]
    pub fn attempt(&self) -> i32 {
        self.inner.lease.attempt
    }

    /// A cancel was requested, or the lease was lost: the job must stop either way.
    #[must_use]
    pub fn cancelled(&self) -> bool {
        self.inner.watch.cancel_requested() || self.inner.watch.lost()
    }

    /// A pause was requested (by the route, or by [`JobContext::request_pause`]).
    #[must_use]
    pub fn paused(&self) -> bool {
        self.inner.watch.pause_requested()
    }

    /// The lease is lost.
    #[must_use]
    pub fn lost(&self) -> bool {
        self.inner.watch.lost()
    }

    /// The worker is draining: stop at a boundary and return [`Outcome::Interrupted`].
    #[must_use]
    pub fn shutting_down(&self) -> bool {
        self.inner.watch.shutting_down()
    }

    /// Whether a write outside the database is still safe (the lease is young enough).
    #[must_use]
    pub fn may_write(&self) -> bool {
        self.inner.watch.may_write()
    }

    /// True once this run lasted the kind's `slice_s`: return [`Outcome::Yield`].
    #[must_use]
    pub fn slice_due(&self) -> bool {
        self.inner
            .kind
            .slice()
            .is_some_and(|slice| self.inner.started.elapsed() >= slice)
    }

    /// Ask for this job's own pause (`audit`, `budget`, ...), then return [`Outcome::Paused`] at
    /// the next boundary.
    pub async fn request_pause(&self, reason: PauseReason) {
        let inner = &self.inner;
        let written = through_outage(inner.outage_budget, || {
            inner.store.request_pause(&inner.lease, reason)
        })
        .await;
        match written {
            Ok(true) => inner.watch.note_pause_requested(),
            Ok(false) => inner.watch.note_lost(),
            Err(error) => self.write_failed("request_pause", &error),
        }
    }

    /// Record a full progress snapshot, at most once per the minimum interval and always on a stage
    /// change; a skipped snapshot is kept and written with the next one or by the wrapper at the
    /// end.
    pub async fn progress(
        &self,
        done: u64,
        total: Option<u64>,
        stage: Option<&str>,
        counters: Value,
    ) {
        let snapshot = json!({"done": done, "total": total, "stage": stage, "counters": counters});
        let due = {
            let mut state = self
                .inner
                .progress
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            state.unwritten = Some(snapshot.clone());
            state
                .last_at
                .is_none_or(|at| at.elapsed() >= self.inner.progress_min_interval)
                || state.last_stage.as_deref() != stage
        };
        if due {
            self.write_progress(&snapshot).await;
        }
    }

    /// Write the last snapshot a throttled call skipped (the wrapper calls this at the end).
    pub(crate) async fn flush_progress(&self) {
        let pending = self
            .inner
            .progress
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .unwritten
            .clone();
        if let Some(snapshot) = pending {
            self.write_progress(&snapshot).await;
        }
    }

    async fn write_progress(&self, snapshot: &Value) {
        let inner = &self.inner;
        let written = through_outage(inner.outage_budget, || {
            inner.store.progress(&inner.lease, snapshot)
        })
        .await;
        self.settle("progress", written);
        let mut state = inner.progress.lock().unwrap_or_else(|p| p.into_inner());
        state.last_at = Some(Instant::now());
        state.last_stage = snapshot
            .get("stage")
            .and_then(Value::as_str)
            .map(str::to_owned);
        state.unwritten = None;
    }

    /// Write the resume cursor, fenced; a resumable kind writes it after each durable chunk.
    pub async fn checkpoint(&self, cursor: Option<&Value>) {
        let inner = &self.inner;
        let written = through_outage(inner.outage_budget, || {
            inner.store.checkpoint(&inner.lease, cursor)
        })
        .await;
        self.settle("checkpoint", written);
    }

    fn settle(&self, what: &str, written: Result<Option<Control>, crate::store::StoreError>) {
        match written {
            Ok(Some(control)) => self.inner.watch.note_good(control),
            Ok(None) => self.inner.watch.note_lost(),
            Err(error) => self.write_failed(what, &error),
        }
    }

    /// A write that could not be confirmed within the outage budget: the lease cannot be trusted
    /// any more, so the job stops writing, as when a heartbeat cannot be confirmed.
    fn write_failed(&self, what: &str, error: &crate::store::StoreError) {
        tracing::warn!(job = %self.inner.lease.job_id, what, %error, "a fenced write failed");
        if error.is_unavailable() {
            self.inner.watch.note_lost();
        }
    }
}

type BoxedRun<S> =
    Arc<dyn Fn(JobContext<S>) -> Pin<Box<dyn Future<Output = Outcome> + Send>> + Send + Sync>;

pub(crate) struct Registered<S: JobStore> {
    pub(crate) spec: &'static KindSpec,
    pub(crate) run: BoxedRun<S>,
}

/// Why a handler could not be registered.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RegisterError {
    /// The name is not in the kind table.
    #[error("job kind {0:?} is not in contracts/json/job_kinds.json")]
    UnknownKind(String),
    /// The kind table says another runtime runs the kind: moving it to Rust is a change of the
    /// generator's `RUST_RUNTIME_KINDS`, made together with the handler.
    #[error("job kind {0:?} is not marked `rust` in contracts/json/job_kinds.json")]
    NotRustKind(String),
    /// A handler is already registered for the kind.
    #[error("job kind {0:?} already has a handler")]
    Twice(String),
}

/// The handlers one process runs. The process claims exactly these kinds.
pub struct Handlers<S: JobStore> {
    by_kind: HashMap<&'static str, Registered<S>>,
}

impl<S: JobStore> Default for Handlers<S> {
    fn default() -> Self {
        Self {
            by_kind: HashMap::new(),
        }
    }
}

impl<S: JobStore> Handlers<S> {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the handler of `kind_name`.
    ///
    /// # Errors
    ///
    /// [`RegisterError`] when the kind does not exist, belongs to the Python runtime, or already
    /// has a handler.
    pub fn register<F, Fut>(&mut self, kind_name: &str, run: F) -> Result<(), RegisterError>
    where
        F: Fn(JobContext<S>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Outcome> + Send + 'static,
    {
        let spec = kind(kind_name).ok_or_else(|| RegisterError::UnknownKind(kind_name.into()))?;
        if spec.runtime != Runtime::Rust {
            return Err(RegisterError::NotRustKind(kind_name.into()));
        }
        if self.by_kind.contains_key(spec.name.as_str()) {
            return Err(RegisterError::Twice(kind_name.into()));
        }
        self.by_kind.insert(
            spec.name.as_str(),
            Registered {
                spec,
                run: Arc::new(move |ctx| Box::pin(run(ctx))),
            },
        );
        Ok(())
    }

    /// The names of the registered kinds, sorted.
    #[must_use]
    pub fn kinds(&self) -> Vec<String> {
        let mut names: Vec<String> = self.by_kind.keys().map(|k| (*k).to_owned()).collect();
        names.sort();
        names
    }

    pub(crate) fn get(&self, kind_name: &str) -> Option<&Registered<S>> {
        self.by_kind.get(kind_name)
    }
}
