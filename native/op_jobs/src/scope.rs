//! The scope a job's own writes run in: its project, or the whole deployment for a job that
//! belongs to no project (`src/tasks/runtime/scoped.py`).
//!
//! A job's rows are read first under deployment scope (the project is not known until the row is
//! read); every later write of the wrapper runs in the job's project scope, so a bug cannot touch
//! another project's rows. The repository functions are generic over [`op_db::Scope`], so the same
//! code serves both; [`JobTx`] only picks the scope for a [`Lease`](crate::Lease).

use op_db::{DbError, DeploymentTx, Pool, ProjectId, ProjectTx, deployment_tx, project_tx};

/// A transaction in the scope of one job.
pub(crate) enum JobTx {
    Project(ProjectTx),
    Deployment(DeploymentTx),
}

/// Run `$body` with `$tx` bound to the transaction inside a [`JobTx`], whichever scope it is.
macro_rules! scoped {
    ($job_tx:expr, $tx:ident => $body:expr) => {
        match $job_tx {
            $crate::scope::JobTx::Project($tx) => $body,
            $crate::scope::JobTx::Deployment($tx) => $body,
        }
    };
}
pub(crate) use scoped;

impl JobTx {
    /// Begin in the job's project scope, or the deployment scope for a job with no project.
    pub(crate) async fn open(pool: &Pool, project: Option<ProjectId>) -> Result<Self, DbError> {
        match project {
            Some(id) => Ok(Self::Project(project_tx(pool, id).await?)),
            None => Ok(Self::Deployment(
                deployment_tx(pool, "job runtime write of a deployment-wide job").await?,
            )),
        }
    }

    /// Commit on success; roll back and pass the error on otherwise.
    pub(crate) async fn finish<T>(self, outcome: Result<T, DbError>) -> Result<T, DbError> {
        match outcome {
            Ok(value) => {
                scoped!(self, tx => tx.commit().await)?;
                Ok(value)
            }
            Err(error) => {
                // The error is the one to report; a failed rollback drops the connection anyway.
                drop(scoped!(self, tx => tx.rollback().await));
                Err(error)
            }
        }
    }
}
