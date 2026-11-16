//! The claim: `queued` to `running`, for the kinds a process has a handler for.
//!
//! One statement takes up to `limit` rows with `FOR UPDATE SKIP LOCKED`, which "can be used to avoid
//! lock contention with multiple consumers accessing a queue-like table"
//! (<https://www.postgresql.org/docs/18/sql-select.html#SQL-FOR-UPDATE-SHARE>), so two runners
//! racing for the same rows each get different ones. It bumps the attempt and the fencing token,
//! writes the owner and the lease, and inserts the `job_attempts` row, all in the one statement and
//! the one transaction that also writes the `job.state` events.
//!
//! A job that can never run is failed instead of claimed: its project was archived or deleted while
//! it waited (`project_unavailable`), or its retry budget is spent (`attempts_exhausted`). A row with
//! a cancel request is never claimed.

use op_db::{DbError, DeploymentTx, Statement};

use crate::events::{StateRow, emit_state, state_columns};
use crate::kinds::contract;
use crate::state::JobStatus;
use crate::store::{ClaimRequest, ClaimedJob, JobId, Lease};

pub(crate) const FAIL_UNAVAILABLE_PROJECT: Statement = Statement::new(
    "jobs.fail_unavailable_project",
    concat!(
        "UPDATE jobs j SET status = 'failed', error_code = 'project_unavailable', \
         error_message = 'project is ' || p.status, finished_at = now(), updated_at = now() \
         FROM projects p WHERE p.id = j.project_id AND j.status = 'queued' \
         AND j.kind = ANY($1::text[]) AND p.status = ANY($2::text[]) RETURNING ",
        state_columns!()
    ),
);

pub(crate) const FAIL_EXHAUSTED: Statement = Statement::new(
    "jobs.fail_exhausted",
    concat!(
        "UPDATE jobs j SET status = 'failed', error_code = 'attempts_exhausted', \
         error_message = 'every attempt ended in a failure', finished_at = now(), updated_at = now() \
         WHERE j.status = 'queued' AND j.kind = ANY($1::text[]) AND j.retry_count >= j.max_attempts \
         RETURNING ",
        state_columns!()
    ),
);

pub(crate) const CLAIM: Statement = Statement::new(
    "jobs.claim",
    "WITH picked AS (SELECT id FROM jobs WHERE status = 'queued' AND kind = ANY($1::text[]) \
           AND cancel_requested_at IS NULL AND retry_count < max_attempts \
           ORDER BY queued_at NULLS LAST, id LIMIT $2 FOR UPDATE SKIP LOCKED), \
         claimed AS (UPDATE jobs j SET status = 'running', attempt = j.attempt + 1, \
           fencing_token = j.fencing_token + 1, owner = $3, heartbeat_at = now(), \
           lease_expires_at = now() + make_interval(secs => $4::double precision), \
           started_at = coalesce(j.started_at, now()), updated_at = now() \
           FROM picked WHERE j.id = picked.id \
           RETURNING j.id, j.project_id, j.kind, j.queue, j.status, j.attempt, j.retry_count, \
             j.pause_reason, j.pause_requested_at, j.cancel_requested_at, j.error_code, \
             j.error_message, j.fencing_token, j.params, j.checkpoint), \
         attempts AS (INSERT INTO job_attempts (job_id, attempt, fencing_token, owner) \
           SELECT id, attempt, fencing_token, $3 FROM claimed) \
         SELECT j.id, j.project_id, j.kind, j.queue, j.status, j.attempt, j.retry_count, \
           j.pause_reason, j.pause_requested_at IS NOT NULL AS pause_requested, \
           j.cancel_requested_at IS NOT NULL AS cancel_requested, j.error_code, \
           left(j.error_message, 500) AS error_message, j.fencing_token, j.params, j.checkpoint \
         FROM claimed j",
);

/// Every statement of this module.
pub(crate) const ALL: &[Statement] = &[FAIL_UNAVAILABLE_PROJECT, FAIL_EXHAUSTED, CLAIM];

/// Fail what can never run, then claim up to `request.limit` jobs, in `tx` (the caller commits).
pub(crate) async fn claim_batch(
    tx: &mut DeploymentTx,
    request: &ClaimRequest,
) -> Result<Vec<ClaimedJob>, DbError> {
    let unavailable = &contract().wrapper.unavailable_project_statuses;
    let dead = tx
        .query(&FAIL_UNAVAILABLE_PROJECT, &[&request.kinds, unavailable])
        .await?;
    let spent = tx.query(&FAIL_EXHAUSTED, &[&request.kinds]).await?;
    for row in dead.iter().chain(&spent) {
        emit_state(tx, &StateRow::read(row)?).await?;
    }
    let limit = i64::try_from(request.limit).unwrap_or(i64::MAX);
    let rows = tx
        .query(
            &CLAIM,
            &[
                &request.kinds,
                &limit,
                &request.owner,
                &request.lease_ttl.as_secs_f64(),
            ],
        )
        .await?;
    let mut claimed = Vec::with_capacity(rows.len());
    for row in &rows {
        let state = StateRow::read(row)?;
        emit_state(tx, &state).await?;
        let token: i64 = row
            .try_get("fencing_token")
            .map_err(|e| DbError::call("jobs.claim", e))?;
        claimed.push(ClaimedJob {
            lease: Lease {
                job_id: JobId(state.id),
                project_id: state.project_id.map(op_db::ProjectId::new),
                fencing_token: token,
                attempt: state.attempt,
            },
            kind: state.kind,
            queue: state.queue,
            params: row
                .try_get("params")
                .map_err(|e| DbError::call("jobs.claim", e))?,
            checkpoint: row
                .try_get("checkpoint")
                .map_err(|e| DbError::call("jobs.claim", e))?,
        });
        debug_assert_eq!(state.status, JobStatus::Running);
    }
    Ok(claimed)
}
