//! The statement registry: every SQL text of the server is a named constant.
//!
//! A statement is a `const` of this type in the crate that owns it, listed in that crate's `ALL`
//! slice. The database-tier test prepares every statement of every `ALL` against the schema head
//! (`testing::TestDatabase::prepare_all`), so a column that was renamed or a typo fails a test and
//! not a job at night. No SQL text is built from input anywhere: values travel as parameters.

/// One named SQL text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Statement {
    /// A stable name, `<area>.<what>`, used in errors and logs.
    pub name: &'static str,
    /// The SQL text, with `$1`-style parameters.
    pub sql: &'static str,
}

impl Statement {
    /// A statement of the registry.
    #[must_use]
    pub const fn new(name: &'static str, sql: &'static str) -> Self {
        Self { name, sql }
    }
}

/// `set_config(..., true)` is transaction-local, so a pooled connection never carries a scope to
/// its next user (as `src/db/session.py` sets it).
pub const APPLY_SCOPE: Statement = Statement::new(
    "db.apply_scope",
    "SELECT set_config('op.scope', $1, true), set_config('op.project_id', $2, true)",
);

/// A session-level advisory lock: held until the connection ends.
pub const TRY_ADVISORY_LOCK: Statement = Statement::new(
    "db.try_advisory_lock",
    "SELECT pg_try_advisory_lock($1, $2)",
);

/// Every statement of this crate, for the database-tier prepare test.
pub const ALL: &[Statement] = &[APPLY_SCOPE, TRY_ADVISORY_LOCK];
