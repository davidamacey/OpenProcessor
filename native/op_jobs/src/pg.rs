//! The real [`JobStore`]: the statements of this crate, each in the transaction scope of its job.

use std::time::Duration;

use op_db::{DbError, Pool, deployment_tx};
use serde_json::Value;

use crate::claim::claim_batch;
use crate::scope::{JobTx, scoped};
use crate::state::{JobStatus, PauseReason};
use crate::store::{ClaimRequest, ClaimedJob, Conclusion, Control, JobStore, Lease, StoreError};
use crate::{lease, terminal};

/// The job tables behind a pool.
#[derive(Debug, Clone)]
pub struct PgJobStore {
    pool: Pool,
}

impl PgJobStore {
    /// A store over `pool`. The pool should be a [`op_db::Profile::Job`] pool with room for every
    /// running job's writes and the heartbeats.
    #[must_use]
    pub fn new(pool: Pool) -> Self {
        Self { pool }
    }

    async fn scoped_write<T, F>(&self, lease: &Lease, run: F) -> Result<T, StoreError>
    where
        F: AsyncFnOnce(&mut JobTx) -> Result<T, DbError>,
    {
        let mut tx = JobTx::open(&self.pool, lease.project_id).await?;
        let out = run(&mut tx).await;
        Ok(tx.finish(out).await?)
    }
}

impl JobStore for PgJobStore {
    async fn claim(&self, request: &ClaimRequest) -> Result<Vec<ClaimedJob>, StoreError> {
        let mut tx = deployment_tx(&self.pool, "job claim across projects").await?;
        match claim_batch(&mut tx, request).await {
            Ok(claimed) => {
                tx.commit().await?;
                Ok(claimed)
            }
            Err(error) => {
                drop(tx.rollback().await);
                Err(error.into())
            }
        }
    }

    async fn heartbeat(&self, lease: &Lease, ttl: Duration) -> Result<Option<Control>, StoreError> {
        self.scoped_write(
            lease,
            async |tx| scoped!(tx, t => lease::heartbeat(t, lease, ttl).await),
        )
        .await
    }

    async fn progress(
        &self,
        lease: &Lease,
        snapshot: &Value,
    ) -> Result<Option<Control>, StoreError> {
        self.scoped_write(
            lease,
            async |tx| scoped!(tx, t => lease::progress(t, lease, snapshot).await),
        )
        .await
    }

    async fn checkpoint(
        &self,
        lease: &Lease,
        cursor: Option<&Value>,
    ) -> Result<Option<Control>, StoreError> {
        self.scoped_write(
            lease,
            async |tx| scoped!(tx, t => lease::checkpoint(t, lease, cursor).await),
        )
        .await
    }

    async fn request_pause(&self, lease: &Lease, reason: PauseReason) -> Result<bool, StoreError> {
        self.scoped_write(
            lease,
            async |tx| scoped!(tx, t => lease::request_pause(t, lease, reason).await),
        )
        .await
    }

    async fn conclude(
        &self,
        lease: &Lease,
        conclusion: &Conclusion,
    ) -> Result<Option<JobStatus>, StoreError> {
        self.scoped_write(
            lease,
            async |tx| scoped!(tx, t => terminal::conclude(t, lease, conclusion).await),
        )
        .await
    }
}
