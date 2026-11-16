//! The lease: renewals, progress and checkpoint writes that name the fencing token, and the
//! reaper's first rule (a lease that expired).
//!
//! A running job's `fencing_token` moves up by one at every claim, and every write the holder
//! makes names that token and the live statuses in its `WHERE`. A holder whose lease was taken
//! over matches no row and gets `None` back (`Lost`). READ COMMITTED
//! (<https://www.postgresql.org/docs/18/transaction-iso.html#XACT-READ-COMMITTED>): an `UPDATE`
//! that finds its row changed by a concurrent commit re-evaluates its `WHERE` on the new version,
//! which is what makes a single statement a safe compare-and-set. A heartbeat touches only
//! `heartbeat_at` and `lease_expires_at`, which are in no index.

use std::time::Duration;

use op_db::{DbError, DeploymentTx, Scope, Statement, Tx};
use serde_json::Value;

use crate::events::{StateRow, emit_state, insert, progress_payload, state_columns};
use crate::state::PauseReason;
use crate::store::{Control, Lease};

pub(crate) const HEARTBEAT: Statement = Statement::new(
    "jobs.heartbeat",
    concat!(
        "UPDATE jobs j SET heartbeat_at = now(), \
         lease_expires_at = now() + make_interval(secs => $3::double precision) \
         WHERE j.id = $1 AND j.fencing_token = $2 AND j.status IN ('running', 'cancelling') \
         RETURNING ",
        state_columns!()
    ),
);

pub(crate) const PROGRESS: Statement = Statement::new(
    "jobs.progress",
    concat!(
        "UPDATE jobs j SET progress = $3, progress_updated_at = now() \
         WHERE j.id = $1 AND j.fencing_token = $2 AND j.status IN ('running', 'cancelling') \
         RETURNING ",
        state_columns!()
    ),
);

pub(crate) const CHECKPOINT: Statement = Statement::new(
    "jobs.checkpoint",
    concat!(
        "UPDATE jobs j SET checkpoint = $3 \
         WHERE j.id = $1 AND j.fencing_token = $2 AND j.status IN ('running', 'cancelling') \
         RETURNING ",
        state_columns!()
    ),
);

pub(crate) const REQUEST_PAUSE: Statement = Statement::new(
    "jobs.request_pause",
    "UPDATE jobs j SET pause_requested_at = coalesce(j.pause_requested_at, now()), \
     pause_reason = coalesce(j.pause_reason, $3::text) \
     WHERE j.id = $1 AND j.fencing_token = $2 AND j.status = 'running' RETURNING 1",
);

/// Reaper rule a: a lease that expired. `cancelling` goes `cancelled`, `running` goes
/// `interrupted` and spends a retry; the batch is `FOR UPDATE SKIP LOCKED`, so two reapers never
/// act on one row.
pub(crate) const EXPIRE_LEASES: Statement = Statement::new(
    "jobs.expire_leases",
    concat!(
        "WITH expired AS (SELECT id FROM jobs WHERE status IN ('running', 'cancelling') \
           AND lease_expires_at < now() ORDER BY lease_expires_at LIMIT $1 \
           FOR UPDATE SKIP LOCKED) \
         UPDATE jobs j SET status = CASE WHEN j.status = 'cancelling' THEN 'cancelled' \
           ELSE 'interrupted' END, \
           finished_at = CASE WHEN j.status = 'cancelling' THEN now() END, \
           owner = NULL, lease_expires_at = NULL, updated_at = now(), \
           retry_count = j.retry_count + CASE WHEN j.status = 'running' THEN 1 ELSE 0 END \
         FROM expired WHERE j.id = expired.id RETURNING ",
        state_columns!()
    ),
);

pub(crate) const END_LOST_ATTEMPT: Statement = Statement::new(
    "jobs.end_lost_attempt",
    "UPDATE job_attempts SET ended_at = now(), outcome = 'lost' \
     WHERE job_id = $1 AND attempt = $2 AND ended_at IS NULL",
);

/// Every statement of this module.
pub(crate) const ALL: &[Statement] = &[
    HEARTBEAT,
    PROGRESS,
    CHECKPOINT,
    REQUEST_PAUSE,
    EXPIRE_LEASES,
    END_LOST_ATTEMPT,
];

/// Renew the lease by `ttl`. `None`: the token no longer holds the row.
pub(crate) async fn heartbeat<S: Scope>(
    tx: &mut Tx<S>,
    lease: &Lease,
    ttl: Duration,
) -> Result<Option<Control>, DbError> {
    let row = tx
        .query_opt(
            &HEARTBEAT,
            &[&lease.job_id.0, &lease.fencing_token, &ttl.as_secs_f64()],
        )
        .await?;
    row.map(|r| StateRow::read(&r).map(|s| s.control()))
        .transpose()
}

/// Store a progress snapshot and its `job.progress` event.
pub(crate) async fn progress<S: Scope>(
    tx: &mut Tx<S>,
    lease: &Lease,
    snapshot: &Value,
) -> Result<Option<Control>, DbError> {
    let Some(row) = tx
        .query_opt(
            &PROGRESS,
            &[&lease.job_id.0, &lease.fencing_token, snapshot],
        )
        .await?
    else {
        return Ok(None);
    };
    let state = StateRow::read(&row)?;
    let payload = progress_payload(state.id, &state.kind, snapshot);
    insert(tx, state.project_id, state.id, "job.progress", &payload).await?;
    Ok(Some(state.control()))
}

/// Store the resume cursor.
pub(crate) async fn checkpoint<S: Scope>(
    tx: &mut Tx<S>,
    lease: &Lease,
    cursor: Option<&Value>,
) -> Result<Option<Control>, DbError> {
    let row = tx
        .query_opt(
            &CHECKPOINT,
            &[&lease.job_id.0, &lease.fencing_token, &cursor],
        )
        .await?;
    row.map(|r| StateRow::read(&r).map(|s| s.control()))
        .transpose()
}

/// The handler asks for its own pause. `false`: the token no longer holds a running row.
pub(crate) async fn request_pause<S: Scope>(
    tx: &mut Tx<S>,
    lease: &Lease,
    reason: PauseReason,
) -> Result<bool, DbError> {
    let reason = reason.as_str();
    Ok(tx
        .query_opt(
            &REQUEST_PAUSE,
            &[&lease.job_id.0, &lease.fencing_token, &reason],
        )
        .await?
        .is_some())
}

/// Reaper rule a: end the leases that expired, in deployment scope. Returns the rows as they are
/// now; each one's attempt ends `lost` and a `job.state` event is written.
///
/// # Errors
///
/// A [`DbError`] when a statement fails; nothing is committed here.
pub async fn expire_leases(
    tx: &mut DeploymentTx,
    batch: usize,
) -> Result<Vec<ExpiredJob>, DbError> {
    let batch = i64::try_from(batch).unwrap_or(i64::MAX);
    let rows = tx.query(&EXPIRE_LEASES, &[&batch]).await?;
    let mut out = Vec::with_capacity(rows.len());
    for row in &rows {
        let state = StateRow::read(row)?;
        tx.execute(&END_LOST_ATTEMPT, &[&state.id, &state.attempt])
            .await?;
        emit_state(tx, &state).await?;
        out.push(ExpiredJob {
            job_id: crate::store::JobId(state.id),
            status: state.status,
        });
    }
    Ok(out)
}

/// A job whose lease the reaper ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpiredJob {
    /// The job.
    pub job_id: crate::store::JobId,
    /// `interrupted`, or `cancelled` for a job that was being cancelled.
    pub status: crate::state::JobStatus,
}
