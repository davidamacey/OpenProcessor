//! A session-level advisory lock on a dedicated direct connection: the single-ticker lock of the
//! periodic role (`src/tasks/beat_lock.py`).
//!
//! The lock lives as long as the connection. A session lock needs a direct connection (a
//! transaction pooler would hand the session to another client), and the connection carries TCP
//! keepalives and a user timeout (set by [`DbSettings::pg_config`]) so a half-open connection fails
//! within about 30 s and the server then frees the lock
//! (<https://www.postgresql.org/docs/18/explicit-locking.html#ADVISORY-LOCKS>).

use std::time::Duration;

use tokio::task::JoinHandle;
use tokio_postgres::NoTls;

use crate::error::DbError;
use crate::settings::DbSettings;
use crate::statement::TRY_ADVISORY_LOCK;

const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// A held session advisory lock.
#[derive(Debug)]
pub struct SessionLock {
    client: tokio_postgres::Client,
    driver: JoinHandle<()>,
}

impl Drop for SessionLock {
    fn drop(&mut self) {
        self.driver.abort();
    }
}

impl SessionLock {
    /// Try to take `pg_try_advisory_lock(class, 0)` on a new direct connection that is kept open.
    /// `Ok(None)` when another session holds it.
    ///
    /// # Errors
    ///
    /// When the connection or the statement fails.
    pub async fn try_acquire(
        settings: &DbSettings,
        application_name: &str,
        class: i32,
    ) -> Result<Option<Self>, DbError> {
        let config = settings.pg_config(application_name, true, false)?;
        let (client, connection) = config
            .connect(NoTls)
            .await
            .map_err(|e| DbError::call("lock.connect", e))?;
        let driver = tokio::spawn(async move {
            if let Err(error) = connection.await {
                tracing::warn!(%error, "the advisory-lock connection failed");
            }
        });
        let lock = Self { client, driver };
        let taken = tokio::time::timeout(
            PROBE_TIMEOUT,
            lock.client
                .query_one(TRY_ADVISORY_LOCK.sql, &[&class, &0_i32]),
        )
        .await
        .map_err(|_| DbError::CallTimeout {
            what: TRY_ADVISORY_LOCK.name,
            seconds: PROBE_TIMEOUT.as_secs(),
        })?
        .map_err(|e| DbError::call(TRY_ADVISORY_LOCK.name, e))?;
        Ok(taken.get::<_, bool>(0).then_some(lock))
    }

    /// Check that the lock's connection still answers. An error means the lock is gone (the
    /// server frees it when the session ends) and the holder must stop acting as the ticker.
    ///
    /// # Errors
    ///
    /// When the connection does not answer within ten seconds or has failed.
    pub async fn probe(&self) -> Result<(), DbError> {
        tokio::time::timeout(PROBE_TIMEOUT, self.client.simple_query("SELECT 1"))
            .await
            .map_err(|_| DbError::CallTimeout {
                what: "lock.probe",
                seconds: PROBE_TIMEOUT.as_secs(),
            })?
            .map_err(|e| DbError::call("lock.probe", e))?;
        Ok(())
    }
}
