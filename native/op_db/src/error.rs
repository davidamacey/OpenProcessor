//! The one error type of the database layer.

use crate::settings::SettingsError;

/// Why a database call failed. A caller that must tell a refused write from a broken connection
/// asks [`DbError::unique_violation`] and [`DbError::is_unavailable`].
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    /// A database setting or its password file could not be used.
    #[error(transparent)]
    Settings(#[from] SettingsError),
    /// No connection could be taken from the pool: the wait, create or recycle timeout ran out,
    /// the pool is closed, or the database refused the new connection.
    #[error("no database connection available: {0}")]
    Pool(#[from] deadpool_postgres::PoolError),
    /// The pool could not be built.
    #[error("the connection pool could not be built: {0}")]
    Build(#[from] deadpool_postgres::BuildError),
    /// A statement or a connection step failed. `what` names the statement of the registry, or the
    /// step, so a log line says which call it was.
    #[error("database call `{what}` failed: {source}")]
    Call {
        /// The registered statement name, or the connection step.
        what: &'static str,
        /// The driver's error.
        #[source]
        source: tokio_postgres::Error,
    },
    /// A call did not answer within its bound. The connection was dropped, because what it was
    /// doing is unknown.
    #[error("database call `{what}` did not answer within {seconds} s; the connection was dropped")]
    CallTimeout {
        /// The registered statement name, or the connection step.
        what: &'static str,
        /// The bound that ran out.
        seconds: u64,
    },
    /// The database returned a value this build does not know how to read (an unknown status, a
    /// column of another type).
    #[error("unexpected data from `{what}`: {detail}")]
    Data {
        /// Where it was read.
        what: &'static str,
        /// What is wrong.
        detail: String,
    },
    /// The transaction was already committed, rolled back or dropped after an error.
    #[error("the transaction is finished: `{what}` was called on it")]
    Finished {
        /// The call that found the transaction finished.
        what: &'static str,
    },
}

impl DbError {
    /// The name of the constraint when a unique index refused the write (`23505`), as the driver
    /// reports it.
    #[must_use]
    pub fn unique_violation(&self) -> Option<&str> {
        let Self::Call { source, .. } = self else {
            return None;
        };
        let db = source.as_db_error()?;
        if *db.code() == tokio_postgres::error::SqlState::UNIQUE_VIOLATION {
            db.constraint()
        } else {
            None
        }
    }

    /// Whether the database could not be reached or stopped answering, as opposed to a statement
    /// it refused. A runner treats these as "try again later", never as a verdict on a job.
    #[must_use]
    pub fn is_unavailable(&self) -> bool {
        match self {
            Self::Pool(_) | Self::Build(_) | Self::CallTimeout { .. } => true,
            Self::Call { source, .. } => source.is_closed() || source.as_db_error().is_none(),
            Self::Settings(_) | Self::Data { .. } | Self::Finished { .. } => false,
        }
    }

    /// A driver error of the call named `what` (a statement of the registry, or a step).
    #[must_use]
    pub fn call(what: &'static str, source: tokio_postgres::Error) -> Self {
        Self::Call { what, source }
    }
}
