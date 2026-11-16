//! The test harness of the database tier (cargo feature `db`; Docker is needed).
//!
//! [`TestDatabase::start`] starts PostgreSQL with testcontainers-modules 0.15.0
//! (<https://docs.rs/testcontainers-modules/0.15.0/testcontainers_modules/postgres/>), creates the
//! two roles of `src/db/roles.py` (the owner with `BYPASSRLS`, the application role without), loads
//! `native/op_db/schema/head.sql`, applies every `src/db/migrations/sql/00NN_*.sql` newer than the
//! revision the dump records, and re-applies the application role's grants. The dump is produced
//! by `scripts/codegen/dump_schema.py` in the sweep; this crate commits no hand-written copy.
//!
//! The application role is a different role from the table owner and has no `BYPASSRLS`, so a test
//! through [`TestDatabase::pool`] meets row-level security exactly as the server does.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use op_config::Env;
use testcontainers_modules::postgres::Postgres;
use testcontainers_modules::testcontainers::runners::AsyncRunner;
use testcontainers_modules::testcontainers::{ContainerAsync, ImageExt};
use tokio_postgres::NoTls;

/// A raw connection, as the table owner or the superuser: for seeding and read-back in tests.
pub use tokio_postgres::Client as RawClient;

use crate::pool::{Pool, PoolSpec, Profile};
use crate::settings::DbSettings;
use crate::statement::Statement;

/// The server version the stack runs (`docker-compose.yml`).
const POSTGRES_TAG: &str = "18.6-trixie";
const SUPER_PASSWORD: &str = "postgres";
const ROLE_PASSWORD: &str = "op-test-password";

/// What the dump's first line must say, so a loader knows which migrations are newer.
const REVISION_PREFIX: &str = "-- alembic revision: ";

/// A PostgreSQL container with the schema loaded.
pub struct TestDatabase {
    _container: ContainerAsync<Postgres>,
    settings: DbSettings,
    port: u16,
    password_file: PathBuf,
}

impl Drop for TestDatabase {
    fn drop(&mut self) {
        drop(fs::remove_file(&self.password_file));
    }
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// The dump without the psql meta-commands (`\restrict`, `\unrestrict`) pg_dump 18 writes, which
/// are not SQL, and without `CREATE SCHEMA public` (the container already has it).
fn sql_of_dump(dump: &str) -> String {
    dump.lines()
        .filter(|line| !line.starts_with('\\') && *line != "CREATE SCHEMA public;")
        .collect::<Vec<_>>()
        .join("\n")
}

/// The numeric prefix of a revision or file name (`0009_x` gives 9).
fn number_of(name: &str) -> Option<u32> {
    name.split('_').next()?.parse().ok()
}

/// `(file name, text)` of every migration SQL file newer than the dump's revision, in order.
fn newer_migrations(revision: &str) -> Vec<(String, String)> {
    let head = number_of(revision)
        .unwrap_or_else(|| panic!("the dump's revision {revision:?} has no numeric prefix"));
    let dir = manifest_dir().join("../../src/db/migrations/sql");
    let Ok(entries) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<(u32, String, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let n = number_of(&name)?;
            (name.ends_with(".sql") && n > head).then(|| (n, name, e.path()))
        })
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|(_, name, path)| {
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
            (name, text)
        })
        .collect()
}

/// What the server said, not the bare `db error` that `tokio_postgres::Error` displays.
fn describe(e: &tokio_postgres::Error) -> String {
    e.as_db_error().map_or_else(
        || e.to_string(),
        |d| {
            format!(
                "{}: {} (detail: {:?}, where: {:?})",
                d.code().code(),
                d.message(),
                d.detail(),
                d.where_()
            )
        },
    )
}

async fn connect(port: u16, user: &str, password: &str) -> RawClient {
    let mut config = tokio_postgres::Config::new();
    config
        .host("127.0.0.1")
        .port(port)
        .user(user)
        .password(password)
        .dbname("postgres");
    let (client, connection) = config
        .connect(NoTls)
        .await
        .unwrap_or_else(|e| panic!("cannot connect to the test database as {user}: {e}"));
    tokio::spawn(connection);
    client
}

impl TestDatabase {
    /// Start the container and load the schema.
    ///
    /// # Panics
    ///
    /// With a message that says what to do when `native/op_db/schema/head.sql` is missing (the
    /// coordinator generates it with `scripts/codegen/dump_schema.py`), when Docker is not
    /// reachable, or when the dump or a newer migration does not load.
    pub async fn start() -> Self {
        let head = manifest_dir().join("schema/head.sql");
        let dump = fs::read_to_string(&head).unwrap_or_else(|_| {
            panic!(
                "{} is missing: the database tier loads the schema dump that \
                 `python3 scripts/codegen/dump_schema.py` writes from a migrated database; run it \
                 (the sweep does) before these tests",
                head.display()
            )
        });
        let revision = dump
            .lines()
            .next()
            .and_then(|l| l.strip_prefix(REVISION_PREFIX))
            .unwrap_or_else(|| {
                panic!(
                    "{} must start with `{REVISION_PREFIX}<revision>`",
                    head.display()
                )
            })
            .trim()
            .to_owned();

        let container = Postgres::default()
            .with_host_auth()
            .with_tag(POSTGRES_TAG)
            .start()
            .await
            .unwrap_or_else(|e| {
                panic!("cannot start the PostgreSQL container (is Docker up?): {e}")
            });
        let port = container
            .get_host_port_ipv4(5432)
            .await
            .unwrap_or_else(|e| panic!("the container has no port 5432: {e}"));

        let superuser = connect(port, "postgres", SUPER_PASSWORD).await;
        superuser
            .batch_execute(&roles_sql())
            .await
            .unwrap_or_else(|e| panic!("cannot create the roles: {}", describe(&e)));
        let owner = connect(port, "op_owner", ROLE_PASSWORD).await;
        owner
            .batch_execute(&sql_of_dump(&dump))
            .await
            .unwrap_or_else(|e| panic!("{} does not load: {}", head.display(), describe(&e)));
        // pg_dump ends its preamble with `set_config('search_path', '', false)`, a session setting.
        owner
            .batch_execute("RESET search_path")
            .await
            .unwrap_or_else(|e| panic!("cannot reset the search path: {}", describe(&e)));
        for (name, text) in newer_migrations(&revision) {
            owner
                .batch_execute(&text)
                .await
                .unwrap_or_else(|e| panic!("migration {name} does not load: {}", describe(&e)));
        }
        owner.batch_execute(GRANTS_SQL).await.unwrap_or_else(|e| {
            panic!(
                "cannot grant the application role its rights: {}",
                describe(&e)
            )
        });

        static NEXT: AtomicU32 = AtomicU32::new(0);
        let password_file = std::env::temp_dir().join(format!(
            "op_db_testing_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::write(&password_file, ROLE_PASSWORD)
            .unwrap_or_else(|e| panic!("cannot write the password file: {e}"));
        let env = Env::from_pairs([
            ("OP_DB_HOST", "127.0.0.1".to_owned()),
            ("OP_DB_PORT", port.to_string()),
            ("OP_DB_NAME", "postgres".to_owned()),
            ("OP_DB_SSLMODE", "disable".to_owned()),
            (
                "OP_DB_APP_PASSWORD_FILE",
                password_file.display().to_string(),
            ),
        ]);
        let settings = DbSettings::from_env(&env)
            .unwrap_or_else(|e| panic!("the test settings are not valid: {e}"));
        Self {
            _container: container,
            settings,
            port,
            password_file,
        }
    }

    /// The settings of the application role on this database (for a `Pool` or a `Listener`).
    #[must_use]
    pub fn settings(&self) -> &DbSettings {
        &self.settings
    }

    /// A pool of the application role: row-level security applies.
    ///
    /// # Panics
    ///
    /// When the pool cannot connect.
    pub async fn pool(&self, profile: Profile, max_size: usize) -> Pool {
        Pool::connect(
            &self.settings,
            PoolSpec {
                application_name: "op-db-test".to_owned(),
                max_size,
                profile,
            },
        )
        .await
        .unwrap_or_else(|e| panic!("the test pool cannot connect: {e}"))
    }

    /// A connection as the table owner, which bypasses row-level security: for seeding rows and
    /// for reading them back past a scope.
    pub async fn owner(&self) -> RawClient {
        connect(self.port, "op_owner", ROLE_PASSWORD).await
    }

    /// Prepare every statement against the loaded schema, as the application role.
    ///
    /// # Panics
    ///
    /// Naming the first statement the database refuses to prepare.
    pub async fn prepare_all(&self, statements: &[Statement]) {
        let client = connect(self.port, "op_app", ROLE_PASSWORD).await;
        for statement in statements {
            if let Err(e) = client.prepare(statement.sql).await {
                panic!(
                    "statement `{}` does not prepare: {}",
                    statement.name,
                    describe(&e)
                );
            }
        }
    }
}

fn roles_sql() -> String {
    format!(
        "CREATE ROLE op_owner LOGIN PASSWORD '{ROLE_PASSWORD}' NOSUPERUSER NOCREATEDB NOCREATEROLE BYPASSRLS;\n\
         CREATE ROLE op_app LOGIN PASSWORD '{ROLE_PASSWORD}' NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;\n\
         ALTER ROLE op_app SET idle_in_transaction_session_timeout = '60s';\n\
         ALTER ROLE op_app SET statement_timeout = '30s';\n\
         ALTER ROLE op_app SET lock_timeout = '5s';\n\
         ALTER ROLE op_app SET timezone = 'UTC';\n\
         ALTER SCHEMA public OWNER TO op_owner;\n\
         GRANT CONNECT ON DATABASE postgres TO op_app;\n"
    )
}

/// `apply_grants` of `src/db/roles.py`.
const GRANTS_SQL: &str = "\
GRANT USAGE ON SCHEMA public TO op_app;\n\
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO op_app;\n\
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO op_app;\n\
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO op_app;\n\
REVOKE INSERT, UPDATE, DELETE ON alembic_version FROM op_app;\n";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dump_loses_psql_meta_commands_and_the_public_schema_creation() {
        let dump = "-- alembic revision: 0009_x\n\\restrict abc\nSET x = 1;\nCREATE SCHEMA public;\nCREATE TABLE t ();\n\\unrestrict abc\n";
        assert_eq!(
            sql_of_dump(dump),
            "-- alembic revision: 0009_x\nSET x = 1;\nCREATE TABLE t ();"
        );
    }

    #[test]
    fn a_revision_and_a_file_name_order_by_their_number() {
        assert_eq!(number_of("0009_ingest_retry_at"), Some(9));
        assert_eq!(number_of("0010_core.sql"), Some(10));
        assert_eq!(number_of("readme.md"), None);
    }
}
