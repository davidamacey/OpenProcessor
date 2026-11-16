//! The job runner: claim, lease, heartbeat, fencing, terminal states, exclusive groups, and the
//! periodic framework, on the existing `jobs` and `job_attempts` tables. It replaces Celery: jobs
//! are rows in PostgreSQL, claimed with `FOR UPDATE SKIP LOCKED`, woken by `LISTEN`/`NOTIFY`, and
//! kept alive by leases with fencing tokens (plan 18, section 2.5; semantics: plan 02b).
//!
//! The roles that run this are wired into `op-server` by a later slice.
#![forbid(unsafe_code)]

mod claim;
pub mod control;
pub mod exclusive;
pub mod handlers;
pub mod kinds;
pub mod lease;
pub mod periodic;
pub mod runner;
pub mod state;
pub mod terminal;

mod events;
mod handler;
mod pg;
mod retry;
mod scope;
mod store;
mod watch;

#[cfg(test)]
mod conformance;
#[cfg(test)]
#[cfg(feature = "db")]
mod db_proofs;
#[cfg(test)]
#[cfg(feature = "db")]
mod db_tier;
#[cfg(test)]
mod fake;

pub use handler::{Handlers, JobContext, Outcome, RegisterError};
pub use pg::PgJobStore;
pub use store::{
    ClaimRequest, ClaimedJob, Conclusion, Control, JobId, JobStore, Lease, StoreError,
};
pub use watch::{LeaseTiming, LeaseWatch};

/// Every statement of the registry, for the database-tier prepare test.
#[must_use]
pub fn statements() -> Vec<op_db::Statement> {
    [
        claim::ALL,
        lease::ALL,
        terminal::ALL,
        control::ALL,
        exclusive::ALL,
        &[events::EVENT_INSERT],
    ]
    .concat()
}
