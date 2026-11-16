//! Starting a job, and the exclusive groups.
//!
//! The partial unique index `uq_jobs_one_active` (project, group, scope; only for active
//! statuses) is what keeps two workers from ever running the same exclusive group for one project:
//! a second active job of the group is refused at insert, and a resume that would make a second one
//! active is refused at the move. The index decides; the runner adds nothing on top.
//!
//! A refusal aborts the statement, and PostgreSQL then refuses every further statement of the
//! transaction, so every insert or move that the index may refuse runs inside a savepoint
//! (<https://www.postgresql.org/docs/18/sql-savepoint.html>); after rolling back to it the same
//! transaction can read the active job's id. The reaper's rule b (an interrupted job that resumes
//! by itself) puts each row in its own savepoint, so a race skips the row instead of rolling back
//! the batch.

use std::time::Duration;

use op_db::{DbError, DeploymentTx, ProjectId, SavepointName, Scope, Statement, Tx};
use serde_json::Value;
use uuid::Uuid;

use crate::control::notify_queued;
use crate::events::{StateRow, emit_state, state_columns};
use crate::kinds::{KindSpec, ProjectScope, kind};
use crate::state::JobStatus;
use crate::store::JobId;

/// The partial unique index of active jobs per project, group and scope.
pub const UQ_ACTIVE: &str = "uq_jobs_one_active";
/// The unique index of idempotency keys.
pub const UQ_IDEMPOTENCY: &str = "uq_jobs_idempotency";

/// What a unique violation was about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Conflict {
    /// Another job of the same exclusive group and scope is active.
    ActiveGroup,
    /// The idempotency key was used before.
    Idempotency,
    /// Something else: not ours to interpret.
    Other,
}

/// Read the constraint name of a unique violation.
pub(crate) fn classify(constraint: Option<&str>) -> Conflict {
    match constraint {
        Some(UQ_ACTIVE) => Conflict::ActiveGroup,
        Some(UQ_IDEMPOTENCY) => Conflict::Idempotency,
        _ => Conflict::Other,
    }
}

/// A job to insert as `queued`.
#[derive(Debug, Clone, PartialEq)]
pub struct NewJob {
    /// The kind.
    pub kind: String,
    /// The project, or `None` for a deployment-wide job.
    pub project: Option<ProjectId>,
    /// The queue.
    pub queue: String,
    /// The parameters.
    pub params: Value,
    /// The exclusive group, if the kind has one.
    pub exclusive_group: Option<String>,
    /// The scope value that narrows the group; empty for the whole project.
    pub scope: String,
    /// Whether the reaper resumes it by itself after an interruption.
    pub auto_resume: bool,
    /// The retry budget.
    pub max_attempts: i32,
    /// How long it may wait queued.
    pub max_queue_wait: Duration,
    /// An idempotency key, 8 to 200 characters; a repeat returns the first job.
    pub idempotency_key: Option<String>,
    /// The job that started this one.
    pub parent: Option<JobId>,
    /// Who asked (`system`, a user name).
    pub requested_by: String,
}

/// Why a [`NewJob`] cannot be made from a kind.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NewJobError {
    /// The kind is not in the table.
    #[error("no job kind named {0:?}")]
    UnknownKind(String),
    /// The kind needs a project and none was given, or is deployment-wide and one was.
    #[error("job kind {0} {1}")]
    Project(String, &'static str),
    /// The kind's exclusive scope names a `params` key that is missing.
    #[error("job kind {kind} needs params[{key:?}] as its scope")]
    Scope {
        /// The kind.
        kind: String,
        /// The `params` key.
        key: String,
    },
}

impl NewJob {
    /// A job of `kind_name` with the queue, limits, exclusive group and scope the kind table says.
    /// The scope value is `scope` when given, else the text of the `params` key the kind names.
    ///
    /// # Errors
    ///
    /// [`NewJobError`] for an unknown kind, a project that does not fit the kind, or a missing
    /// scope key (`_scope_value` and `_check_project` of `src/services/jobs/service.py`).
    pub fn for_kind(
        kind_name: &str,
        project: Option<ProjectId>,
        params: Value,
        scope: Option<String>,
        requested_by: &str,
    ) -> Result<Self, NewJobError> {
        let spec = kind(kind_name).ok_or_else(|| NewJobError::UnknownKind(kind_name.into()))?;
        check_project(spec, project)?;
        let scope = match scope {
            Some(value) => value,
            None if spec.exclusive_scope.is_empty() => String::new(),
            None => match params.get(&spec.exclusive_scope) {
                Some(Value::String(text)) => text.clone(),
                Some(other) => other.to_string(),
                None => {
                    return Err(NewJobError::Scope {
                        kind: spec.name.clone(),
                        key: spec.exclusive_scope.clone(),
                    });
                }
            },
        };
        Ok(Self {
            kind: spec.name.clone(),
            project,
            queue: spec.queue.clone(),
            params,
            exclusive_group: spec.exclusive_group.clone(),
            scope,
            auto_resume: spec.auto_resume,
            max_attempts: spec.max_attempts,
            max_queue_wait: Duration::from_secs(spec.max_queue_wait_s),
            idempotency_key: None,
            parent: None,
            requested_by: requested_by.to_owned(),
        })
    }
}

fn check_project(spec: &KindSpec, project: Option<ProjectId>) -> Result<(), NewJobError> {
    match (spec.project_scope, project) {
        (ProjectScope::Project | ProjectScope::Target, None) => {
            Err(NewJobError::Project(spec.name.clone(), "needs a project"))
        }
        (ProjectScope::Deployment, Some(_)) => Err(NewJobError::Project(
            spec.name.clone(),
            "is deployment-wide and takes no project",
        )),
        _ => Ok(()),
    }
}

pub(crate) const INSERT_QUEUED: Statement = Statement::new(
    "jobs.insert_queued",
    concat!(
        "INSERT INTO jobs AS j (project_id, kind, scope, exclusive_group, queue, status, params, \
           auto_resume, max_attempts, max_queue_wait, idempotency_key, parent_job_id, \
           requested_by, queued_at) \
         VALUES ($1, $2, $3, $4, $5, 'queued', $6, $7, $8, \
           make_interval(secs => $9::double precision), $10, $11, $12, now()) RETURNING ",
        state_columns!()
    ),
);

pub(crate) const BY_IDEMPOTENCY: Statement = Statement::new(
    "jobs.by_idempotency",
    concat!(
        "SELECT ",
        state_columns!(),
        ", j.params AS params FROM jobs j WHERE j.project_id IS NOT DISTINCT FROM $1 \
         AND j.kind = $2 AND j.idempotency_key = $3"
    ),
);

pub(crate) const ACTIVE_OF_GROUP: Statement = Statement::new(
    "jobs.active_of_group",
    "SELECT id FROM jobs WHERE project_id IS NOT DISTINCT FROM $1 AND exclusive_group = $2 \
     AND scope = $3 AND status = ANY($4::text[]) LIMIT 1",
);

/// Rule b: interrupted jobs that resume by themselves, or whose retry budget is spent.
pub(crate) const SELECT_RESUMABLE: Statement = Statement::new(
    "jobs.select_resumable",
    "SELECT j.id, j.retry_count >= j.max_attempts AS exhausted FROM jobs j \
     WHERE j.status = 'interrupted' AND j.auto_resume \
       AND j.updated_at < now() - make_interval(secs => $1::double precision) \
       AND (j.retry_count >= j.max_attempts OR NOT EXISTS (SELECT 1 FROM jobs a \
         WHERE a.project_id IS NOT DISTINCT FROM j.project_id \
           AND a.exclusive_group = j.exclusive_group AND a.scope = j.scope \
           AND a.status = ANY($2::text[]))) \
     ORDER BY j.updated_at LIMIT $3 FOR UPDATE OF j SKIP LOCKED",
);

pub(crate) const FAIL_INTERRUPTED_EXHAUSTED: Statement = Statement::new(
    "jobs.fail_interrupted_exhausted",
    concat!(
        "UPDATE jobs j SET status = 'failed', error_code = 'attempts_exhausted', \
           error_message = 'every attempt ended in a failure', finished_at = now(), \
           updated_at = now() WHERE j.id = $1 AND j.status = 'interrupted' RETURNING ",
        state_columns!()
    ),
);

pub(crate) const REQUEUE_INTERRUPTED: Statement = Statement::new(
    "jobs.requeue_interrupted",
    concat!(
        "UPDATE jobs j SET status = 'queued', pause_requested_at = NULL, pause_reason = NULL, \
           queued_at = now(), last_dispatched_at = NULL, updated_at = now() \
         WHERE j.id = $1 AND j.status = 'interrupted' RETURNING ",
        state_columns!()
    ),
);

/// Every statement of this module.
pub(crate) const ALL: &[Statement] = &[
    INSERT_QUEUED,
    BY_IDEMPOTENCY,
    ACTIVE_OF_GROUP,
    SELECT_RESUMABLE,
    FAIL_INTERRUPTED_EXHAUSTED,
    REQUEUE_INTERRUPTED,
];

const START_SAVEPOINT: SavepointName = SavepointName::new("start_job");
const REQUEUE_SAVEPOINT: SavepointName = SavepointName::new("requeue_row");

/// What an insert did.
#[derive(Debug, Clone, PartialEq)]
pub enum Inserted {
    /// The job is `queued`; runners were woken.
    Created(JobId),
    /// The idempotency key matched an earlier job: that one.
    Existing {
        /// The first job with this key.
        job: JobId,
        /// Whether its parameters equal the ones given (`false` is a reuse of the key for another
        /// request, which the caller refuses).
        same_params: bool,
    },
    /// Another job of the exclusive group is active.
    Busy {
        /// The active job, if it is still there.
        active: Option<JobId>,
    },
}

/// Insert `new` as `queued` in a savepoint, write its `job.state` event and `NOTIFY op_jobs`.
///
/// # Errors
///
/// A [`DbError`] for any failure other than the two unique indexes' refusals, which are
/// [`Inserted::Busy`] and [`Inserted::Existing`].
pub async fn insert_queued<S: Scope>(tx: &mut Tx<S>, new: &NewJob) -> Result<Inserted, DbError> {
    let project = new.project.map(ProjectId::uuid);
    let parent = new.parent.map(|p| p.0);
    tx.savepoint(START_SAVEPOINT).await?;
    let inserted = tx
        .query_opt(
            &INSERT_QUEUED,
            &[
                &project,
                &new.kind,
                &new.scope,
                &new.exclusive_group,
                &new.queue,
                &new.params,
                &new.auto_resume,
                &new.max_attempts,
                &new.max_queue_wait.as_secs_f64(),
                &new.idempotency_key,
                &parent,
                &new.requested_by,
            ],
        )
        .await;
    match inserted {
        Ok(row) => {
            tx.release(START_SAVEPOINT).await?;
            let state = StateRow::read(&row.ok_or(DbError::Data {
                what: "jobs.insert_queued",
                detail: "an insert returned no row".into(),
            })?)?;
            emit_state(tx, &state).await?;
            notify_queued(tx, &state.queue).await?;
            Ok(Inserted::Created(JobId(state.id)))
        }
        Err(error) => match classify(error.unique_violation()) {
            Conflict::Other => Err(error),
            conflict => {
                tx.rollback_to(START_SAVEPOINT).await?;
                tx.release(START_SAVEPOINT).await?;
                after_refusal(tx, new, conflict).await
            }
        },
    }
}

async fn after_refusal<S: Scope>(
    tx: &mut Tx<S>,
    new: &NewJob,
    conflict: Conflict,
) -> Result<Inserted, DbError> {
    let project = new.project.map(ProjectId::uuid);
    if conflict == Conflict::Idempotency {
        let row = tx
            .query_opt(
                &BY_IDEMPOTENCY,
                &[&project, &new.kind, &new.idempotency_key],
            )
            .await?;
        if let Some(row) = row {
            let state = StateRow::read(&row)?;
            let stored: Value = row
                .try_get("params")
                .map_err(|e| DbError::call("jobs.by_idempotency", e))?;
            return Ok(Inserted::Existing {
                job: JobId(state.id),
                same_params: stored == new.params,
            });
        }
    }
    let group = new.exclusive_group.clone().unwrap_or_default();
    let active: Vec<String> = JobStatus::ALL
        .into_iter()
        .filter(|s| s.is_active())
        .map(|s| s.as_str().to_owned())
        .collect();
    let found = tx
        .query_opt(&ACTIVE_OF_GROUP, &[&project, &group, &new.scope, &active])
        .await?;
    let active = found
        .map(|row| row.try_get::<_, Uuid>("id").map(JobId))
        .transpose()
        .map_err(|e| DbError::call("jobs.active_of_group", e))?;
    Ok(Inserted::Busy { active })
}

/// What reaper rule b did with a batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResumeReport {
    /// Jobs put back to `queued`.
    pub requeued: usize,
    /// Jobs whose retry budget was spent: `failed`, `attempts_exhausted`.
    pub failed: usize,
    /// Jobs left `interrupted` because another job of their group became active in the race.
    pub skipped: usize,
}

/// Reaper rule b: put the `interrupted` jobs of `auto_resume` kinds back to `queued` once they
/// settled (`settle`), when no job of their group is active; fail the ones whose retry budget is
/// spent. Each row runs in its own savepoint.
///
/// # Errors
///
/// A [`DbError`] for any failure other than the exclusive-group refusal, which skips the row.
pub async fn resume_interrupted(
    tx: &mut DeploymentTx,
    settle: Duration,
    batch: usize,
) -> Result<ResumeReport, DbError> {
    let active: Vec<String> = JobStatus::ALL
        .into_iter()
        .filter(|s| s.is_active())
        .map(|s| s.as_str().to_owned())
        .collect();
    let batch = i64::try_from(batch).unwrap_or(i64::MAX);
    let candidates = tx
        .query(&SELECT_RESUMABLE, &[&settle.as_secs_f64(), &active, &batch])
        .await?;
    let mut report = ResumeReport::default();
    for candidate in &candidates {
        let id: Uuid = candidate
            .try_get("id")
            .map_err(|e| DbError::call("jobs.select_resumable", e))?;
        let exhausted: bool = candidate
            .try_get("exhausted")
            .map_err(|e| DbError::call("jobs.select_resumable", e))?;
        match resume_one(tx, id, exhausted).await? {
            Moved::Requeued => report.requeued += 1,
            Moved::Failed => report.failed += 1,
            Moved::Skipped => report.skipped += 1,
            Moved::Gone => {}
        }
    }
    Ok(report)
}

/// What happened to one candidate of rule b.
enum Moved {
    Requeued,
    Failed,
    /// Another job of the group became active in the race: the row stays interrupted.
    Skipped,
    /// The row moved on before this transaction got to it.
    Gone,
}

async fn resume_one(tx: &mut DeploymentTx, id: Uuid, exhausted: bool) -> Result<Moved, DbError> {
    if exhausted {
        return Ok(
            match tx.query_opt(&FAIL_INTERRUPTED_EXHAUSTED, &[&id]).await? {
                Some(row) => {
                    emit_state(tx, &StateRow::read(&row)?).await?;
                    Moved::Failed
                }
                None => Moved::Gone,
            },
        );
    }
    tx.savepoint(REQUEUE_SAVEPOINT).await?;
    match tx.query_opt(&REQUEUE_INTERRUPTED, &[&id]).await {
        Ok(row) => {
            tx.release(REQUEUE_SAVEPOINT).await?;
            let Some(row) = row else {
                return Ok(Moved::Gone);
            };
            let state = StateRow::read(&row)?;
            emit_state(tx, &state).await?;
            notify_queued(tx, &state.queue).await?;
            Ok(Moved::Requeued)
        }
        Err(error) if classify(error.unique_violation()) == Conflict::ActiveGroup => {
            tx.rollback_to(REQUEUE_SAVEPOINT).await?;
            tx.release(REQUEUE_SAVEPOINT).await?;
            tracing::info!(job = %id, "a job of the group became active; the interrupted one stays");
            Ok(Moved::Skipped)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn only_the_two_unique_indexes_are_interpreted() {
        assert_eq!(classify(Some("uq_jobs_one_active")), Conflict::ActiveGroup);
        assert_eq!(classify(Some("uq_jobs_idempotency")), Conflict::Idempotency);
        assert_eq!(classify(Some("jobs_pkey")), Conflict::Other);
        assert_eq!(classify(None), Conflict::Other);
    }

    #[test]
    fn a_new_job_takes_its_queue_group_and_limits_from_the_kind_table() {
        let job = NewJob::for_kind(
            "label_op",
            Some(ProjectId::new(Uuid::nil())),
            json!({"op_id": "abc"}),
            None,
            "tester",
        )
        .unwrap();
        assert_eq!(job.queue, "interactive");
        assert_eq!(job.exclusive_group.as_deref(), Some("label_op"));
        assert_eq!(job.scope, "abc");
        assert_eq!(job.max_attempts, 3);
        assert_eq!(job.max_queue_wait, Duration::from_secs(3600));
        let project_wide = NewJob::for_kind(
            "reprocess",
            Some(ProjectId::new(Uuid::nil())),
            json!({}),
            None,
            "tester",
        )
        .unwrap();
        assert_eq!(project_wide.scope, "");
        assert!(project_wide.auto_resume);
    }

    #[test]
    fn a_kind_with_a_scope_key_refuses_params_without_it() {
        let err = NewJob::for_kind(
            "label_op",
            Some(ProjectId::new(Uuid::nil())),
            json!({}),
            None,
            "t",
        )
        .unwrap_err();
        assert_eq!(
            err,
            NewJobError::Scope {
                kind: "label_op".into(),
                key: "op_id".into()
            }
        );
        let explicit = NewJob::for_kind(
            "label_op",
            Some(ProjectId::new(Uuid::nil())),
            json!({}),
            Some("given".into()),
            "t",
        )
        .unwrap();
        assert_eq!(explicit.scope, "given");
    }

    #[test]
    fn a_project_kind_needs_a_project_and_a_deployment_kind_takes_none() {
        assert_eq!(
            NewJob::for_kind("reprocess", None, json!({}), None, "t").unwrap_err(),
            NewJobError::Project("reprocess".into(), "needs a project")
        );
        assert_eq!(
            NewJob::for_kind(
                "noop",
                Some(ProjectId::new(Uuid::nil())),
                json!({}),
                None,
                "t"
            )
            .unwrap_err(),
            NewJobError::Project("noop".into(), "is deployment-wide and takes no project")
        );
        assert!(matches!(
            NewJob::for_kind("no_such", None, json!({}), None, "t"),
            Err(NewJobError::UnknownKind(_))
        ));
        assert!(NewJob::for_kind("noop", None, json!({}), None, "t").is_ok());
    }
}
