//! The connection pool. It is private to this crate: the rest of the server reaches the database
//! only through [`crate::project_tx`] and [`crate::deployment_tx`], so no code can open a
//! transaction without a scope.
//!
//! deadpool-postgres 0.14.2 (<https://docs.rs/deadpool-postgres/0.14.2/deadpool_postgres/>) is
//! used as documented: `Manager::from_config` with a [`ManagerConfig`], the builder's `max_size`
//! and its three timeouts (`wait`, `create`, `recycle`, which need the `Runtime::Tokio1` runtime
//! set on the builder), and `prepare_cached` for the per-connection statement cache.
//! [`RecyclingMethod::Verified`] runs a test query when a connection is taken again, which is what
//! the Python engines' `pool_pre_ping` did: a half-open connection is found at checkout, not by
//! the first job statement.

use std::time::Duration;

use deadpool_postgres::{Manager, ManagerConfig, RecyclingMethod, Runtime};

use crate::error::DbError;
use crate::settings::{CONNECT_TIMEOUT, DbSettings, Pooler};

/// What the connections of a pool are for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Request handlers: short server-side limits from the role, a short wait for a free slot (a
    /// full pool answers 503 quickly instead of queueing requests).
    Api,
    /// A worker or a periodic tick: long statement limits, a longer wait for a slot.
    Job,
}

impl Profile {
    fn wait_timeout(self) -> Duration {
        match self {
            Self::Api => Duration::from_secs(5),
            Self::Job => Duration::from_secs(30),
        }
    }

    /// The bound on one statement as seen by the client: the server's own limit plus a margin, so
    /// the server's error is normally what the caller sees and this is the backstop.
    pub(crate) fn call_timeout(self) -> Duration {
        match self {
            Self::Api => Duration::from_secs(35),
            Self::Job => Duration::from_secs(30 * 60 + 30),
        }
    }
}

/// How one pool is made.
#[derive(Debug, Clone)]
pub struct PoolSpec {
    /// The `application_name` the connections carry (visible in `pg_stat_activity`).
    pub application_name: String,
    /// The most connections the pool opens.
    pub max_size: usize,
    /// What the connections are for.
    pub profile: Profile,
}

/// A pool of connections. Cloning shares the pool.
#[derive(Clone)]
pub struct Pool {
    inner: deadpool_postgres::Pool,
    pooler: Pooler,
    profile: Profile,
}

impl std::fmt::Debug for Pool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pool")
            .field("profile", &self.profile)
            .field("pooler", &self.pooler)
            .finish_non_exhaustive()
    }
}

impl Pool {
    /// Build the pool and take one connection from it, so a wrong host or password fails at
    /// startup (deadpool does not connect when the pool is built).
    ///
    /// # Errors
    ///
    /// [`DbError::Settings`] for an unreadable password file, [`DbError::Build`] when the pool
    /// cannot be built, [`DbError::Pool`] when the first connection cannot be made.
    pub async fn connect(settings: &DbSettings, spec: PoolSpec) -> Result<Self, DbError> {
        let config =
            settings.pg_config(&spec.application_name, false, spec.profile == Profile::Job)?;
        let manager = Manager::from_config(
            config,
            settings.tls_connector()?,
            ManagerConfig {
                recycling_method: RecyclingMethod::Verified,
            },
        );
        let inner = deadpool_postgres::Pool::builder(manager)
            .max_size(spec.max_size)
            .wait_timeout(Some(spec.profile.wait_timeout()))
            .create_timeout(Some(CONNECT_TIMEOUT))
            .recycle_timeout(Some(Duration::from_secs(5)))
            .runtime(Runtime::Tokio1)
            .build()?;
        drop(inner.get().await?);
        Ok(Self {
            inner,
            pooler: settings.pooler,
            profile: spec.profile,
        })
    }

    pub(crate) async fn checkout(&self) -> Result<deadpool_postgres::Client, DbError> {
        Ok(self.inner.get().await?)
    }

    pub(crate) fn pooler(&self) -> Pooler {
        self.pooler
    }

    pub(crate) fn profile(&self) -> Profile {
        self.profile
    }

    /// Close the pool: waiting callers fail, idle connections close, borrowed ones close when
    /// they come back.
    pub fn close(&self) {
        self.inner.close();
    }
}
