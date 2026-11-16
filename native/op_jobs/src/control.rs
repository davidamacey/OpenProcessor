//! The controls a person has over a job, and what a running job observes of them.
//!
//! Cancel and pause are database writes only: there is no revoke and no message to recall (plan 02b
//! section 3.3). The row's status is the truth. A `queued`, `paused` or `interrupted` job is
//! cancelled at once; a `running` one becomes `cancelling`, which the handler sees at its next
//! heartbeat or progress write and stops, and the wrapper then writes `cancelled`. A pause of a
//! `queued` job is at once; of a `running` one it is a request, which the handler honours at a
//! boundary. Resume moves `paused` or `interrupted` back to `queued` in a savepoint, because the
//! exclusive-group index may refuse the move when another job of the group became active meanwhile.

use op_db::{DbError, SavepointName, Scope, Statement, Tx};

use crate::events::{StateRow, emit_state, state_columns};
use crate::exclusive::{Conflict, classify};
use crate::state::PauseReason;
use crate::store::{Control, JobId};

/// The channel runners listen on. A notification is a wake-up only: the poll is the backstop.
pub const JOBS_CHANNEL: &str = "op_jobs";

pub(crate) const NOTIFY_JOBS: Statement =
    Statement::new("jobs.notify", "SELECT pg_notify('op_jobs', $1)");

pub(crate) const CANCEL: Statement = Statement::new(
    "jobs.cancel",
    concat!(
        "UPDATE jobs j SET cancel_requested_at = coalesce(j.cancel_requested_at, now()), \
           cancel_requested_by = coalesce(j.cancel_requested_by, $2), \
           status = CASE WHEN j.status IN ('queued', 'paused', 'interrupted') THEN 'cancelled' \
             ELSE 'cancelling' END, \
           finished_at = CASE WHEN j.status IN ('queued', 'paused', 'interrupted') \
             THEN now() END, \
           updated_at = CASE WHEN j.status IN ('queued', 'paused', 'interrupted') \
             THEN now() ELSE j.updated_at END \
         WHERE j.id = $1 AND j.status IN ('queued', 'running', 'paused', 'interrupted', \
           'cancelling') RETURNING ",
        state_columns!()
    ),
);

pub(crate) const PAUSE: Statement = Statement::new(
    "jobs.pause",
    concat!(
        "UPDATE jobs j SET status = CASE WHEN j.status = 'queued' THEN 'paused' ELSE j.status END, \
           pause_requested_at = coalesce(j.pause_requested_at, now()), \
           pause_reason = coalesce(j.pause_reason, $2::text), \
           updated_at = CASE WHEN j.status = 'queued' THEN now() ELSE j.updated_at END \
         WHERE j.id = $1 AND j.status IN ('queued', 'running', 'paused') RETURNING ",
        state_columns!()
    ),
);

pub(crate) const RESUME: Statement = Statement::new(
    "jobs.resume",
    concat!(
        "UPDATE jobs j SET status = 'queued', pause_requested_at = NULL, pause_reason = NULL, \
           queued_at = now(), last_dispatched_at = NULL, updated_at = now() \
         WHERE j.id = $1 AND j.status IN ('paused', 'interrupted') RETURNING ",
        state_columns!()
    ),
);

/// Every statement of this module.
pub(crate) const ALL: &[Statement] = &[NOTIFY_JOBS, CANCEL, PAUSE, RESUME];

const RESUME_SAVEPOINT: SavepointName = SavepointName::new("resume_job");

/// Wake the runners that listen for `queue`: `NOTIFY op_jobs` is delivered when the transaction
/// commits.
pub(crate) async fn notify_queued<S: Scope>(tx: &mut Tx<S>, queue: &str) -> Result<(), DbError> {
    tx.execute(&NOTIFY_JOBS, &[&queue]).await.map(drop)
}

/// Cancel in one statement. `None` when the job is missing or already terminal.
///
/// # Errors
///
/// A [`DbError`] when the statement fails; nothing is committed here.
pub async fn cancel<S: Scope>(
    tx: &mut Tx<S>,
    job: JobId,
    requested_by: &str,
) -> Result<Option<Control>, DbError> {
    let row = tx.query_opt(&CANCEL, &[&job.0, &requested_by]).await?;
    changed(tx, row).await
}

/// Ask a job to pause: a `queued` job pauses at once, a `running` one is asked to; a `paused` one
/// is returned as it is. `None` when the job cannot be paused from where it is (missing, terminal,
/// `interrupted`, `cancelling`) or when its kind is not pausable: the caller checks the kind.
///
/// # Errors
///
/// A [`DbError`] when the statement fails; nothing is committed here.
pub async fn pause<S: Scope>(
    tx: &mut Tx<S>,
    job: JobId,
    reason: PauseReason,
) -> Result<Option<Control>, DbError> {
    let reason = reason.as_str();
    let row = tx.query_opt(&PAUSE, &[&job.0, &reason]).await?;
    changed(tx, row).await
}

/// What a resume did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resumed {
    /// The job is `queued` again.
    Queued(Control),
    /// The job is not `paused` or `interrupted`.
    NotResumable,
    /// Another job of the same exclusive group became active meanwhile; this one stays as it was.
    Busy,
}

/// Move a `paused` or `interrupted` job back to `queued` and wake the runners.
///
/// # Errors
///
/// A [`DbError`] for any failure other than the exclusive-group refusal, which is
/// [`Resumed::Busy`].
pub async fn resume<S: Scope>(tx: &mut Tx<S>, job: JobId) -> Result<Resumed, DbError> {
    tx.savepoint(RESUME_SAVEPOINT).await?;
    match tx.query_opt(&RESUME, &[&job.0]).await {
        Ok(None) => {
            tx.release(RESUME_SAVEPOINT).await?;
            Ok(Resumed::NotResumable)
        }
        Ok(Some(row)) => {
            tx.release(RESUME_SAVEPOINT).await?;
            let state = StateRow::read(&row)?;
            emit_state(tx, &state).await?;
            notify_queued(tx, &state.queue).await?;
            Ok(Resumed::Queued(state.control()))
        }
        Err(error) if classify(error.unique_violation()) == Conflict::ActiveGroup => {
            tx.rollback_to(RESUME_SAVEPOINT).await?;
            tx.release(RESUME_SAVEPOINT).await?;
            Ok(Resumed::Busy)
        }
        Err(error) => Err(error),
    }
}

async fn changed<S: Scope>(
    tx: &mut Tx<S>,
    row: Option<op_db::Row>,
) -> Result<Option<Control>, DbError> {
    let Some(row) = row else {
        return Ok(None);
    };
    let state = StateRow::read(&row)?;
    emit_state(tx, &state).await?;
    Ok(Some(state.control()))
}
