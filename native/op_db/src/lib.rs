//! The database layer of the server: the pool (private), the scoped transaction, the statement
//! registry, the `LISTEN` connection, a session advisory lock, and the test harness that loads the
//! schema dump.
//!
//! The pool is private to this crate. Code reaches the database through [`project_tx`] (one
//! project's rows) or, for the few callers that act across projects, [`deployment_tx`]; see
//! [`tx`] for why a repository call without a scope does not compile.
#![forbid(unsafe_code)]

mod error;
mod ids;
mod listener;
mod lock;
mod pool;
mod settings;
mod statement;
#[cfg(feature = "db")]
pub mod testing;
pub mod tx;

pub use error::DbError;
pub use ids::ProjectId;
pub use listener::{
    Backoff, ListenConnector, ListenEvents, ListenSession, Listener, ListenerEvent, Notification,
    PgConnector, PgSession,
};
pub use lock::SessionLock;
pub use pool::{Pool, PoolSpec, Profile};
pub use settings::{DbSettings, Pooler, SettingsError, SslMode};
pub use statement::{ALL as STATEMENTS, APPLY_SCOPE, Statement};
pub use tokio_postgres::Row;
pub use tokio_postgres::types::{FromSql, ToSql};
pub use tx::{
    Deployment, DeploymentTx, Project, ProjectTx, SavepointName, Scope, Tx, deployment_tx,
    project_tx,
};
