//! The scoped transaction: the only way application code reaches a row-level-secured table.
//!
//! Every project table has row-level security keyed on two transaction-local settings,
//! `op.scope` and `op.project_id` (`src/db/session.py` sets them the same way). A [`Tx`] begins a
//! transaction on a pooled connection and sets them in its first statement, so a statement of
//! this transaction sees only its scope's rows, and `set_config(..., true)` is transaction-local,
//! so the next user of the connection inherits nothing.
//!
//! * [`ProjectTx`] sees one project's rows; [`project_tx`] makes one, and anybody may.
//! * [`DeploymentTx`] sees every project's rows. [`deployment_tx`] is for the code that must act
//!   across projects (the job runner claims a job before it knows the project, the reaper scans
//!   all of them). `tests/deployment_scope.rs` fails when another crate calls it, as
//!   `src/db/scope_allowlist.py` does for the Python side.
//!
//! A repository function takes `&mut ProjectTx` (or `&mut Tx<S>` when it is valid in every scope).
//! There is no way to get a `Tx` except through these two functions, because the pool is private.
//!
//! A repository function takes the scope it needs, and the control compiles:
//!
//! ```
//! async fn read(tx: &mut op_db::ProjectTx, st: &op_db::Statement) {
//!     let _ = tx.query(st, &[]).await;
//! }
//! ```
//!
//! ```compile_fail,E0599
//! // The same call with a bare pool does not compile: a `Pool` is not a scope.
//! async fn read(pool: &mut op_db::Pool, st: &op_db::Statement) {
//!     let _ = pool.query(st, &[]).await;
//! }
//! ```
//!
//! ```compile_fail,E0624
//! // The pool's checkout is crate-private: no handler can open a bare transaction.
//! async fn bare(pool: &op_db::Pool) {
//!     let _conn = pool.checkout().await;
//! }
//! ```
//!
//! ```compile_fail,E0308
//! // A function that takes a project transaction does not accept a deployment one.
//! fn needs_project(_: &mut op_db::ProjectTx) {}
//! fn wrong(tx: &mut op_db::DeploymentTx) {
//!     needs_project(tx);
//! }
//! ```
//!
//! Dropping a `Tx` that was not committed or rolled back drops its connection (the server rolls
//! the transaction back), so a cancelled future never returns a half-finished transaction to the
//! pool. Statements are prepared with `prepare_cached` (one cache per pooled connection) unless a
//! transaction pooler is in front, where each statement is prepared for its own use.

use std::marker::PhantomData;
use std::time::Duration;

use tokio_postgres::Row;
use tokio_postgres::types::ToSql;

use crate::error::DbError;
use crate::ids::ProjectId;
use crate::pool::Pool;
use crate::settings::Pooler;
use crate::statement::{APPLY_SCOPE, Statement};

mod sealed {
    pub trait Sealed {}
}

/// What a transaction is allowed to see. Sealed: only the two scopes below exist.
pub trait Scope: sealed::Sealed {}

/// The scope of one project's rows.
#[derive(Debug)]
pub struct Project;
/// The scope of every project's rows.
#[derive(Debug)]
pub struct Deployment;

impl sealed::Sealed for Project {}
impl sealed::Sealed for Deployment {}
impl Scope for Project {}
impl Scope for Deployment {}

/// A transaction that sees one project's rows.
pub type ProjectTx = Tx<Project>;
/// A transaction that sees every project's rows.
pub type DeploymentTx = Tx<Deployment>;

/// A savepoint name. A `const` so a bad name fails to compile: it is spliced into SQL text.
#[derive(Debug, Clone, Copy)]
pub struct SavepointName(&'static str);

impl SavepointName {
    /// A name of ASCII lowercase letters and underscores.
    ///
    /// # Panics
    ///
    /// At compile time, when used in a `const`, if the name is empty or has any other character.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        let bytes = name.as_bytes();
        assert!(!bytes.is_empty(), "a savepoint name is not empty");
        let mut i = 0;
        while i < bytes.len() {
            assert!(
                bytes[i].is_ascii_lowercase() || bytes[i] == b'_',
                "a savepoint name is lowercase ASCII letters and underscores"
            );
            i += 1;
        }
        Self(name)
    }
}

/// An open scoped transaction on one pooled connection.
pub struct Tx<S: Scope> {
    conn: Option<deadpool_postgres::Client>,
    cached: bool,
    call_timeout: Duration,
    _scope: PhantomData<S>,
}

impl<S: Scope> std::fmt::Debug for Tx<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tx")
            .field("open", &self.conn.is_some())
            .finish_non_exhaustive()
    }
}

/// Begin a transaction that sees only `project`'s rows.
///
/// # Errors
///
/// [`DbError::Pool`] when no connection is free within the pool's wait timeout, or any error of
/// beginning the transaction and setting the scope.
pub async fn project_tx(pool: &Pool, project: ProjectId) -> Result<ProjectTx, DbError> {
    Tx::begin(pool, "project", &project.to_string()).await
}

/// Begin a transaction that sees every project's rows. `reason` says why, and is logged.
///
/// Only the crates listed in `tests/deployment_scope.rs` may call this.
///
/// # Errors
///
/// As [`project_tx`].
pub async fn deployment_tx(pool: &Pool, reason: &'static str) -> Result<DeploymentTx, DbError> {
    tracing::debug!(reason, "deployment_scope");
    Tx::begin(pool, "deployment", "").await
}

impl<S: Scope> Tx<S> {
    async fn begin(pool: &Pool, scope: &'static str, project: &str) -> Result<Self, DbError> {
        let conn = pool.checkout().await?;
        let mut tx = Self {
            conn: Some(conn),
            cached: pool.pooler() == Pooler::None,
            call_timeout: pool.profile().call_timeout(),
            _scope: PhantomData,
        };
        tx.batch("db.begin", "BEGIN").await?;
        tx.execute(&APPLY_SCOPE, &[&scope, &project]).await?;
        Ok(tx)
    }

    /// Run a statement and return every row.
    ///
    /// # Errors
    ///
    /// [`DbError::Call`] when the statement fails, [`DbError::CallTimeout`] when it does not answer
    /// within the pool profile's bound (the connection is then dropped), [`DbError::Finished`]
    /// after a commit or a rollback.
    pub async fn query(
        &mut self,
        statement: &Statement,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<Vec<Row>, DbError> {
        let conn = self.conn.as_ref().ok_or(DbError::Finished {
            what: statement.name,
        })?;
        let cached = self.cached;
        let call = async {
            let prepared = if cached {
                conn.prepare_cached(statement.sql).await?
            } else {
                conn.prepare(statement.sql).await?
            };
            conn.query(&prepared, params).await
        };
        let out = tokio::time::timeout(self.call_timeout, call).await;
        self.settle(statement.name, out)
    }

    /// Run a statement that returns at most one row.
    ///
    /// # Errors
    ///
    /// As [`Tx::query`]; more than one row is an error of the statement, not of this call, and the
    /// first row is returned.
    pub async fn query_opt(
        &mut self,
        statement: &Statement,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<Option<Row>, DbError> {
        Ok(self.query(statement, params).await?.into_iter().next())
    }

    /// Run a statement and return the number of rows it changed.
    ///
    /// # Errors
    ///
    /// As [`Tx::query`].
    pub async fn execute(
        &mut self,
        statement: &Statement,
        params: &[&(dyn ToSql + Sync)],
    ) -> Result<u64, DbError> {
        let conn = self.conn.as_ref().ok_or(DbError::Finished {
            what: statement.name,
        })?;
        let cached = self.cached;
        let call = async {
            let prepared = if cached {
                conn.prepare_cached(statement.sql).await?
            } else {
                conn.prepare(statement.sql).await?
            };
            conn.execute(&prepared, params).await
        };
        let out = tokio::time::timeout(self.call_timeout, call).await;
        self.settle(statement.name, out)
    }

    /// Set a savepoint: a statement that fails after this can be undone with
    /// [`Tx::rollback_to`] without aborting the whole transaction.
    ///
    /// # Errors
    ///
    /// As [`Tx::query`].
    pub async fn savepoint(&mut self, name: SavepointName) -> Result<(), DbError> {
        self.batch("db.savepoint", &format!("SAVEPOINT {}", name.0))
            .await
    }

    /// Undo everything since the savepoint and keep the savepoint.
    ///
    /// # Errors
    ///
    /// As [`Tx::query`].
    pub async fn rollback_to(&mut self, name: SavepointName) -> Result<(), DbError> {
        self.batch(
            "db.rollback_to",
            &format!("ROLLBACK TO SAVEPOINT {}", name.0),
        )
        .await
    }

    /// Forget the savepoint and keep its changes.
    ///
    /// # Errors
    ///
    /// As [`Tx::query`].
    pub async fn release(&mut self, name: SavepointName) -> Result<(), DbError> {
        self.batch("db.release", &format!("RELEASE SAVEPOINT {}", name.0))
            .await
    }

    /// Commit and give the connection back to the pool.
    ///
    /// # Errors
    ///
    /// As [`Tx::query`]. After an error the connection is dropped, not reused.
    pub async fn commit(mut self) -> Result<(), DbError> {
        self.batch("db.commit", "COMMIT").await?;
        self.conn = None;
        Ok(())
    }

    /// Roll back and give the connection back to the pool.
    ///
    /// # Errors
    ///
    /// As [`Tx::query`]. After an error the connection is dropped, not reused.
    pub async fn rollback(mut self) -> Result<(), DbError> {
        self.batch("db.rollback", "ROLLBACK").await?;
        self.conn = None;
        Ok(())
    }

    async fn batch(&mut self, what: &'static str, sql: &str) -> Result<(), DbError> {
        let conn = self.conn.as_ref().ok_or(DbError::Finished { what })?;
        let out = tokio::time::timeout(self.call_timeout, conn.batch_execute(sql)).await;
        self.settle(what, out)
    }

    /// A call that ran out of time or failed on the wire leaves the connection in an unknown
    /// state: drop it. A statement the server refused leaves the transaction open for the caller
    /// to roll back.
    fn settle<T>(
        &mut self,
        what: &'static str,
        out: Result<Result<T, tokio_postgres::Error>, tokio::time::error::Elapsed>,
    ) -> Result<T, DbError> {
        match out {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(source)) => {
                if source.as_db_error().is_none() {
                    self.detach();
                }
                Err(DbError::call(what, source))
            }
            Err(_) => {
                self.detach();
                Err(DbError::CallTimeout {
                    what,
                    seconds: self.call_timeout.as_secs(),
                })
            }
        }
    }

    fn detach(&mut self) {
        if let Some(conn) = self.conn.take() {
            // Takes the connection out of the pool for good; dropping it closes the socket and
            // the server rolls the transaction back (deadpool `Object::take`).
            drop(deadpool_postgres::Object::take(conn));
        }
    }
}

impl<S: Scope> Drop for Tx<S> {
    fn drop(&mut self) {
        self.detach();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_savepoint_name_is_checked_when_it_is_a_const() {
        const OK: SavepointName = SavepointName::new("requeue_row");
        assert_eq!(OK.0, "requeue_row");
    }

    #[test]
    #[should_panic(expected = "lowercase ASCII")]
    fn a_savepoint_name_with_other_characters_is_refused() {
        let _ = SavepointName::new("x; DROP TABLE jobs");
    }
}
