//! What the runner needs from the job tables, as a trait: [`PgJobStore`](crate::PgJobStore) is the
//! real one; the runner's tests use an in-memory one with a paused clock.
//!
//! Every write that belongs to a running job names the job's fencing token. A `None` answer from
//! such a write means the token no longer holds the row (`Lost`): another runner took the job over,
//! the lease expired, or the row was finished. The holder then writes nothing more.

use std::fmt;
use std::future::Future;
use std::time::Duration;

use op_db::{DbError, ProjectId};
use serde_json::Value;
use uuid::Uuid;

use crate::state::{AttemptOutcome, JobStatus, PauseReason};

/// A job's id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub Uuid);

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.hyphenated().fmt(f)
    }
}

/// The claim a runner holds on a job: the row, and the fencing token that proves it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lease {
    /// The job.
    pub job_id: JobId,
    /// Its project, or `None` for a deployment-wide job.
    pub project_id: Option<ProjectId>,
    /// The token the claim wrote. Every later write names it.
    pub fencing_token: i64,
    /// The attempt number the claim wrote.
    pub attempt: i32,
}

/// A job a runner has claimed.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaimedJob {
    /// The claim.
    pub lease: Lease,
    /// The job's kind.
    pub kind: String,
    /// The job's queue.
    pub queue: String,
    /// The parameters the job was started with.
    pub params: Value,
    /// The cursor the previous attempt last wrote, if any.
    pub checkpoint: Option<Value>,
}

/// What a runner asks a store to claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimRequest {
    /// The kinds this process has a handler for.
    pub kinds: Vec<String>,
    /// The most jobs to claim now.
    pub limit: usize,
    /// Who is claiming (`host:pid`), stored in `jobs.owner`.
    pub owner: String,
    /// How long the new lease lasts without a renewal.
    pub lease_ttl: Duration,
}

/// What a fenced write learns about the row it wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Control {
    /// The row's status now.
    pub status: JobStatus,
    /// A cancel was requested.
    pub cancel_requested: bool,
    /// A pause was requested.
    pub pause_requested: bool,
    /// The reason of the pause request.
    pub pause_reason: Option<PauseReason>,
}

/// How a run ended, as the one fenced write that ends it.
#[derive(Debug, Clone, PartialEq)]
pub enum Conclusion {
    /// A terminal status. A row that is `cancelling` can only end `cancelled` (the transition
    /// table has no other exit), so the store writes `cancelled` there and keeps the result.
    Finish {
        /// The terminal status asked for.
        status: JobStatus,
        /// How the attempt ended.
        outcome: AttemptOutcome,
        /// The result object, if any.
        result: Option<serde_json::Map<String, Value>>,
        /// A stable code for a failure.
        error_code: Option<String>,
        /// A message for a failure.
        error_message: Option<String>,
    },
    /// The worker stopped: `running` to `interrupted`, `cancelling` to `cancelled`; spends a retry.
    Interrupt,
    /// The handler stopped at a boundary after a pause request: `running` to `paused`.
    Pause {
        /// The reason, when the handler gave its own; else the one already requested, else `user`.
        reason: Option<PauseReason>,
    },
    /// A sliced job gives way: `running` to `queued`, no retry spent.
    Yield,
}

/// Why a store call failed. Never a verdict on a job: the caller retries or lets the lease run out.
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    /// The database call failed.
    #[error(transparent)]
    Db(#[from] DbError),
    /// The store refused for another reason.
    #[error("{0}")]
    Other(String),
}

impl StoreError {
    /// Whether the database could not be reached, so trying again later is reasonable.
    #[must_use]
    pub fn is_unavailable(&self) -> bool {
        match self {
            Self::Db(e) => e.is_unavailable(),
            Self::Other(_) => true,
        }
    }
}

/// The job tables as the runner uses them.
pub trait JobStore: Send + Sync + 'static {
    /// Claim up to `request.limit` queued jobs of the requested kinds: `queued` to `running`, the
    /// attempt and the fencing token up by one, an owner and a lease, one `job_attempts` row. Jobs
    /// that can never run (their project is gone, their retry budget is spent) end `failed`
    /// instead.
    fn claim(
        &self,
        request: &ClaimRequest,
    ) -> impl Future<Output = Result<Vec<ClaimedJob>, StoreError>> + Send;

    /// Extend the lease by `ttl` if `lease` still holds the row.
    fn heartbeat(
        &self,
        lease: &Lease,
        ttl: Duration,
    ) -> impl Future<Output = Result<Option<Control>, StoreError>> + Send;

    /// Store a full progress snapshot if `lease` still holds the row.
    fn progress(
        &self,
        lease: &Lease,
        snapshot: &Value,
    ) -> impl Future<Output = Result<Option<Control>, StoreError>> + Send;

    /// Store the resume cursor if `lease` still holds the row.
    fn checkpoint(
        &self,
        lease: &Lease,
        cursor: Option<&Value>,
    ) -> impl Future<Output = Result<Option<Control>, StoreError>> + Send;

    /// The handler asks for its own pause (`audit`, `budget`, ...); `false` means `Lost`.
    fn request_pause(
        &self,
        lease: &Lease,
        reason: PauseReason,
    ) -> impl Future<Output = Result<bool, StoreError>> + Send;

    /// End the run with one fenced write. The status the row has afterwards, or `None` for `Lost`
    /// (nothing was written).
    fn conclude(
        &self,
        lease: &Lease,
        conclusion: &Conclusion,
    ) -> impl Future<Output = Result<Option<JobStatus>, StoreError>> + Send;
}
