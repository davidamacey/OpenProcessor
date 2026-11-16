//! The database settings, read from the same `OP_DB_*` variables as `src/config/database.py`.
//!
//! `OP_DB_SSLMODE` and `OP_DB_SSLROOTCERT` choose TLS and its trust roots; `tls.rs` holds the
//! connector.
//!
//! The password is never an environment value: it is read from the file `OP_DB_APP_PASSWORD_FILE`
//! names, only when a connection is built, and never stored in [`DbSettings`] or printed.

use std::path::PathBuf;
use std::time::Duration;

use op_config::{Env, EnvError};
use tokio_postgres_rustls::MakeRustlsConnect;

use crate::tls;

/// Seconds a new connection may take to be established (`CONNECT_TIMEOUT_S` of the Python side).
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// A statement running longer than this ends on the server (a job connection; the API role keeps
/// its own shorter limit from `ALTER ROLE`).
const JOB_STATEMENT_TIMEOUT: &str = "30min";
/// A statement waiting for a lock longer than this ends on the server.
const JOB_LOCK_TIMEOUT: &str = "30s";

/// Why a setting cannot be used. Every message names the variable.
#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    /// The variable is unset where required, or does not parse.
    #[error(transparent)]
    Env(#[from] EnvError),
    /// The variable parsed but its value is not allowed.
    #[error("{key}: {reason}")]
    Invalid {
        /// The environment variable name.
        key: &'static str,
        /// What is wrong with the value.
        reason: String,
    },
    /// The password file cannot be read or is empty.
    #[error("{key}: the password file {path} {reason}")]
    Password {
        /// The environment variable that names the file.
        key: &'static str,
        /// The path it names.
        path: String,
        /// What is wrong with the file.
        reason: &'static str,
    },
}

/// How the client wants TLS, with libpq's meaning
/// (<https://www.postgresql.org/docs/18/libpq-ssl.html>). What each mode verifies is in
/// `tls.rs`. `allow` is [`SslMode::Prefer`]: libpq tries plain text first and TLS only when
/// the server refuses it, the driver cannot retry that way, and offering TLS first protects at
/// least as much.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SslMode {
    /// `disable`: plain text, never an `SSLRequest`.
    Disable,
    /// `prefer` and `allow`: TLS when the server offers it, plain text when it does not; the
    /// certificate is not verified. A handshake that starts and then fails fails the connection
    /// (libpq would retry in plain text; refusing is the stricter answer).
    Prefer,
    /// `require`: TLS or no connection. The certificate chain is verified when
    /// `OP_DB_SSLROOTCERT` is set (libpq does when a root file exists), the host name never.
    Require,
    /// `verify-ca`: TLS, the certificate chain is verified, the host name is not.
    VerifyCa,
    /// `verify-full`: TLS, the chain and the host name are verified.
    VerifyFull,
}

/// What stands between the process and PostgreSQL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooler {
    /// A direct connection: prepared statements are cached per connection.
    None,
    /// PgBouncer in transaction mode: a server-side prepared statement does not survive the
    /// transaction, so statements are prepared per use.
    Pgbouncer,
}

/// The `OP_DB_*` settings of one process.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbSettings {
    /// `OP_DB_HOST`: the host the pool connects to (the pooler when there is one).
    pub host: String,
    /// `OP_DB_DIRECT_HOST`: the host session advisory locks and `LISTEN` use; never a pooler.
    pub direct_host: String,
    /// `OP_DB_PORT`.
    pub port: u16,
    /// `OP_DB_NAME`.
    pub name: String,
    /// `OP_DB_APP_USER`: the application role (no `BYPASSRLS`).
    pub app_user: String,
    /// `OP_DB_APP_PASSWORD_FILE`.
    pub app_password_file: PathBuf,
    /// `OP_DB_SSLMODE`.
    pub ssl_mode: SslMode,
    /// `OP_DB_SSLROOTCERT`: a PEM file of CA certificates that is then the only trust root of
    /// `require`, `verify-ca` and `verify-full`; unset, the platform's roots are trusted.
    pub ssl_root_cert: Option<PathBuf>,
    /// `OP_DB_POOLER`.
    pub pooler: Pooler,
    /// `OP_DB_API_CONNECTIONS_TOTAL`: the connections the API may hold across all replicas.
    pub api_connections_total: usize,
    /// `OP_API_REPLICAS`.
    pub api_replicas: usize,
}

fn identifier(key: &'static str, value: String) -> Result<String, SettingsError> {
    let mut chars = value.chars();
    let first_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_');
    let rest_ok = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if first_ok && rest_ok && value.len() <= 63 {
        Ok(value)
    } else {
        Err(SettingsError::Invalid {
            key,
            reason: format!("not a valid PostgreSQL identifier (got {value:?})"),
        })
    }
}

fn host(key: &'static str, value: String) -> Result<String, SettingsError> {
    let ok = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if ok {
        Ok(value)
    } else {
        Err(SettingsError::Invalid {
            key,
            reason: format!(
                "must be a bare host name or address, not a URL or path (got {value:?})"
            ),
        })
    }
}

fn at_least_one(key: &'static str, value: usize) -> Result<usize, SettingsError> {
    if value >= 1 {
        Ok(value)
    } else {
        Err(SettingsError::Invalid {
            key,
            reason: "must be at least 1".into(),
        })
    }
}

fn tls_settings(env: &Env) -> Result<(SslMode, Option<PathBuf>), SettingsError> {
    let text = env
        .get("OP_DB_SSLMODE", "prefer".to_owned())?
        .to_lowercase();
    let ssl_mode = match text.as_str() {
        "disable" => SslMode::Disable,
        "allow" | "prefer" => SslMode::Prefer,
        "require" => SslMode::Require,
        "verify-ca" => SslMode::VerifyCa,
        "verify-full" => SslMode::VerifyFull,
        other => {
            return Err(SettingsError::Invalid {
                key: "OP_DB_SSLMODE",
                reason: format!(
                    "unknown value {other:?}; use one of disable, allow, prefer, require, \
                     verify-ca, verify-full"
                ),
            });
        }
    };
    let root: String = env.get(tls::ROOT_CERT_KEY, String::new())?;
    let ssl_root_cert = (!root.is_empty()).then(|| PathBuf::from(root));
    if ssl_root_cert.is_some() && matches!(ssl_mode, SslMode::Disable | SslMode::Prefer) {
        return Err(SettingsError::Invalid {
            key: tls::ROOT_CERT_KEY,
            reason: format!(
                "is used only with OP_DB_SSLMODE require, verify-ca or verify-full, not {text:?}"
            ),
        });
    }
    // Build the connector once so that an unreadable roots file is refused at startup.
    tls::connector(ssl_mode, ssl_root_cert.as_deref())?;
    Ok((ssl_mode, ssl_root_cert))
}

impl DbSettings {
    /// Read the settings; every default is the Python side's.
    ///
    /// # Errors
    ///
    /// A [`SettingsError`] naming the variable when a value does not parse or is not allowed.
    /// `OP_DB_SSLMODE` is one of libpq's six values; `OP_DB_SSLROOTCERT` is refused with `disable`
    /// and `prefer`/`allow` (it would be silently unused, and a connection nobody verifies must
    /// not look verified), when the file cannot be read, or when it holds no usable certificate.
    pub fn from_env(env: &Env) -> Result<Self, SettingsError> {
        let host_value = host("OP_DB_HOST", env.get("OP_DB_HOST", "postgres".to_owned())?)?;
        let direct_host = host(
            "OP_DB_DIRECT_HOST",
            env.get("OP_DB_DIRECT_HOST", host_value.clone())?,
        )?;
        let (ssl_mode, ssl_root_cert) = tls_settings(env)?;
        let pooler = match env
            .get("OP_DB_POOLER", "none".to_owned())?
            .to_lowercase()
            .as_str()
        {
            "none" => Pooler::None,
            "pgbouncer" => Pooler::Pgbouncer,
            other => {
                return Err(SettingsError::Invalid {
                    key: "OP_DB_POOLER",
                    reason: format!("unknown value {other:?}; use one of none, pgbouncer"),
                });
            }
        };
        let port: u16 = env.get("OP_DB_PORT", 5432)?;
        if port == 0 {
            return Err(SettingsError::Invalid {
                key: "OP_DB_PORT",
                reason: "must be 1 to 65535".into(),
            });
        }
        Ok(Self {
            host: host_value,
            direct_host,
            port,
            name: identifier(
                "OP_DB_NAME",
                env.get("OP_DB_NAME", "openprocessor".to_owned())?,
            )?,
            app_user: identifier(
                "OP_DB_APP_USER",
                env.get("OP_DB_APP_USER", "op_app".to_owned())?,
            )?,
            app_password_file: PathBuf::from(env.get(
                "OP_DB_APP_PASSWORD_FILE",
                "/run/secrets/op_pg/app_password".to_owned(),
            )?),
            ssl_mode,
            ssl_root_cert,
            pooler,
            api_connections_total: at_least_one(
                "OP_DB_API_CONNECTIONS_TOTAL",
                env.get("OP_DB_API_CONNECTIONS_TOTAL", 128)?,
            )?,
            api_replicas: at_least_one("OP_API_REPLICAS", env.get("OP_API_REPLICAS", 1)?)?,
        })
    }

    /// The connections one API process may hold: the deployment total shared by the replicas, at
    /// least one (`db_pool_size` of the Python side, with one process per replica).
    #[must_use]
    pub fn api_pool_size(&self) -> usize {
        (self.api_connections_total / self.api_replicas).max(1)
    }

    fn password(&self) -> Result<String, SettingsError> {
        let path = self.app_password_file.display().to_string();
        let text = std::fs::read_to_string(&self.app_password_file).map_err(|_| {
            SettingsError::Password {
                key: "OP_DB_APP_PASSWORD_FILE",
                path: path.clone(),
                reason: "cannot be read",
            }
        })?;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Err(SettingsError::Password {
                key: "OP_DB_APP_PASSWORD_FILE",
                path,
                reason: "is empty",
            });
        }
        Ok(trimmed.to_owned())
    }

    /// The TLS connector for [`Self::pg_config`]'s connections: every connection this crate makes
    /// is made with it, whatever the mode (`disable` never uses it).
    ///
    /// # Errors
    ///
    /// [`SettingsError::Invalid`] when the roots file or the platform's roots cannot be loaded.
    pub fn tls_connector(&self) -> Result<MakeRustlsConnect, SettingsError> {
        tls::connector(self.ssl_mode, self.ssl_root_cert.as_deref())
    }

    /// The driver configuration of one connection.
    ///
    /// `direct` connects to `OP_DB_DIRECT_HOST` (advisory locks and `LISTEN` never meet a pooler).
    /// `job_limits` raises the statement and lock timeouts for a long-running worker, as the
    /// Python job engines do; PgBouncer refuses startup options, so they are left out there and
    /// the role's own limits apply. TCP keepalives and a user timeout make a half-open connection
    /// fail within about 30 s instead of the kernel's default.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Password`] when the password file cannot be read or is empty.
    pub fn pg_config(
        &self,
        application_name: &str,
        direct: bool,
        job_limits: bool,
    ) -> Result<tokio_postgres::Config, SettingsError> {
        let mut config = tokio_postgres::Config::new();
        config
            .host(if direct {
                &self.direct_host
            } else {
                &self.host
            })
            .port(self.port)
            .dbname(&self.name)
            .user(&self.app_user)
            .password(self.password()?)
            .application_name(application_name)
            .ssl_mode(match self.ssl_mode {
                SslMode::Disable => tokio_postgres::config::SslMode::Disable,
                SslMode::Prefer => tokio_postgres::config::SslMode::Prefer,
                SslMode::Require | SslMode::VerifyCa | SslMode::VerifyFull => {
                    tokio_postgres::config::SslMode::Require
                }
            })
            .connect_timeout(CONNECT_TIMEOUT)
            .keepalives(true)
            .keepalives_idle(Duration::from_secs(10))
            .keepalives_interval(Duration::from_secs(5))
            .keepalives_retries(3)
            .tcp_user_timeout(Duration::from_secs(30));
        if job_limits && (direct || self.pooler == Pooler::None) {
            config.options(format!(
                "-c statement_timeout={JOB_STATEMENT_TIMEOUT} -c lock_timeout={JOB_LOCK_TIMEOUT}"
            ));
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> Env {
        Env::from_pairs(pairs.iter().map(|(k, v)| (*k, *v)))
    }

    #[test]
    fn the_defaults_are_the_python_defaults() {
        let s = DbSettings::from_env(&env(&[])).unwrap();
        assert_eq!(s.host, "postgres");
        assert_eq!(s.direct_host, "postgres");
        assert_eq!((s.port, s.name.as_str()), (5432, "openprocessor"));
        assert_eq!(s.app_user, "op_app");
        assert_eq!(s.ssl_mode, SslMode::Prefer);
        assert_eq!(s.pooler, Pooler::None);
        assert_eq!(s.api_pool_size(), 128);
    }

    #[test]
    fn the_direct_host_follows_the_host_unless_set() {
        let s = DbSettings::from_env(&env(&[("OP_DB_HOST", "pgb")])).unwrap();
        assert_eq!(s.direct_host, "pgb");
        let s = DbSettings::from_env(&env(&[("OP_DB_HOST", "pgb"), ("OP_DB_DIRECT_HOST", "pg")]))
            .unwrap();
        assert_eq!((s.host.as_str(), s.direct_host.as_str()), ("pgb", "pg"));
    }

    #[test]
    fn a_bad_value_names_its_variable() {
        for (key, value) in [
            ("OP_DB_HOST", "http://db/"),
            ("OP_DB_PORT", "0"),
            ("OP_DB_PORT", "x"),
            ("OP_DB_NAME", "Open Processor"),
            ("OP_DB_APP_USER", "1op"),
            ("OP_DB_POOLER", "pgcat"),
            ("OP_DB_SSLMODE", "sometimes"),
            ("OP_DB_SSLMODE", "verify_full"),
            ("OP_API_REPLICAS", "0"),
            ("OP_DB_API_CONNECTIONS_TOTAL", "0"),
        ] {
            let err = DbSettings::from_env(&env(&[(key, value)])).unwrap_err();
            assert!(err.to_string().contains(key), "{key}: {err}");
        }
    }

    #[test]
    fn every_ssl_mode_maps_to_what_libpq_means_by_it() {
        for (text, mode) in [
            ("disable", SslMode::Disable),
            ("allow", SslMode::Prefer),
            ("prefer", SslMode::Prefer),
            ("PREFER", SslMode::Prefer),
            ("require", SslMode::Require),
            ("verify-ca", SslMode::VerifyCa),
            ("verify-full", SslMode::VerifyFull),
        ] {
            let s = DbSettings::from_env(&env(&[("OP_DB_SSLMODE", text)])).unwrap();
            assert_eq!(s.ssl_mode, mode, "{text}");
            assert_eq!(s.ssl_root_cert, None);
        }
    }

    #[test]
    fn the_driver_mode_is_plain_prefer_or_require() {
        use tokio_postgres::config::SslMode as Driver;
        let path = password_file("sslmap", Some("pw"));
        let file = path.to_str().unwrap();
        for (text, driver) in [
            ("disable", Driver::Disable),
            ("allow", Driver::Prefer),
            ("prefer", Driver::Prefer),
            ("require", Driver::Require),
            ("verify-ca", Driver::Require),
            ("verify-full", Driver::Require),
        ] {
            let s = DbSettings::from_env(&env(&[
                ("OP_DB_SSLMODE", text),
                ("OP_DB_APP_PASSWORD_FILE", file),
            ]))
            .unwrap();
            let config = s.pg_config("a", false, false).unwrap();
            assert_eq!(config.get_ssl_mode(), driver, "{text}");
        }
    }

    fn ca_file(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("op_db_ca_{}_{name}", std::process::id()));
        let ca = rcgen::generate_simple_self_signed(vec!["ca.test".to_owned()]).unwrap();
        std::fs::write(&path, ca.cert.pem()).unwrap();
        path
    }

    #[test]
    fn a_roots_file_is_read_only_by_the_modes_that_verify() {
        let ca = ca_file("modes");
        let file = ca.to_str().unwrap();
        for mode in ["require", "verify-ca", "verify-full"] {
            let s = DbSettings::from_env(&env(&[
                ("OP_DB_SSLMODE", mode),
                ("OP_DB_SSLROOTCERT", file),
            ]))
            .unwrap();
            assert_eq!(s.ssl_root_cert.as_deref(), Some(ca.as_path()), "{mode}");
        }
        for mode in ["disable", "allow", "prefer"] {
            let err = DbSettings::from_env(&env(&[
                ("OP_DB_SSLMODE", mode),
                ("OP_DB_SSLROOTCERT", file),
            ]))
            .unwrap_err()
            .to_string();
            assert!(
                err.contains("OP_DB_SSLROOTCERT") && err.contains(mode),
                "{mode}: {err}"
            );
        }
    }

    #[test]
    fn a_roots_file_that_is_missing_or_holds_no_certificate_is_refused_naming_the_variable() {
        let missing = password_file("rootgone", None);
        let empty = password_file("rootempty", Some("no certificate here\n"));
        let garbled = password_file(
            "rootgarbled",
            Some("-----BEGIN CERTIFICATE-----\n!!!\n-----END CERTIFICATE-----\n"),
        );
        for (path, why) in [
            (&missing, "cannot be read"),
            (&empty, "no usable CA certificate"),
            (&garbled, "PEM"),
        ] {
            let err = DbSettings::from_env(&env(&[
                ("OP_DB_SSLMODE", "verify-full"),
                ("OP_DB_SSLROOTCERT", path.to_str().unwrap()),
            ]))
            .unwrap_err()
            .to_string();
            assert!(
                err.contains("OP_DB_SSLROOTCERT") && err.contains(why),
                "{why}: {err}"
            );
        }
    }

    #[test]
    fn the_pool_share_divides_by_the_replicas_and_is_at_least_one() {
        let s = DbSettings::from_env(&env(&[
            ("OP_DB_API_CONNECTIONS_TOTAL", "10"),
            ("OP_API_REPLICAS", "4"),
        ]))
        .unwrap();
        assert_eq!(s.api_pool_size(), 2);
        let s = DbSettings::from_env(&env(&[
            ("OP_DB_API_CONNECTIONS_TOTAL", "3"),
            ("OP_API_REPLICAS", "8"),
        ]))
        .unwrap();
        assert_eq!(s.api_pool_size(), 1);
    }

    fn password_file(name: &str, content: Option<&str>) -> PathBuf {
        let path = std::env::temp_dir().join(format!("op_db_test_{}_{name}", std::process::id()));
        match content {
            Some(c) => std::fs::write(&path, c).unwrap(),
            None => drop(std::fs::remove_file(&path)),
        }
        path
    }

    #[test]
    fn the_password_comes_from_the_file_and_a_missing_or_empty_file_names_the_variable() {
        let path = password_file("ok", Some("s3cret\n"));
        let s = DbSettings::from_env(&env(&[("OP_DB_APP_PASSWORD_FILE", path.to_str().unwrap())]))
            .unwrap();
        let config = s.pg_config("op-test", false, true).unwrap();
        assert_eq!(config.get_password(), Some(&b"s3cret"[..]));
        assert_eq!(config.get_user(), Some("op_app"));
        assert_eq!(config.get_application_name(), Some("op-test"));

        for (name, content, why) in [
            ("gone", None, "cannot be read"),
            ("empty", Some(" \n"), "empty"),
        ] {
            let path = password_file(name, content);
            let s =
                DbSettings::from_env(&env(&[("OP_DB_APP_PASSWORD_FILE", path.to_str().unwrap())]))
                    .unwrap();
            let msg = s
                .pg_config("op-test", false, false)
                .unwrap_err()
                .to_string();
            assert!(
                msg.contains("OP_DB_APP_PASSWORD_FILE") && msg.contains(why),
                "{msg}"
            );
            assert!(!msg.contains("s3cret"));
        }
    }

    #[test]
    fn job_connections_raise_the_timeouts_except_behind_a_pooler() {
        let path = password_file("limits", Some("pw"));
        let file = path.to_str().unwrap();
        let direct = DbSettings::from_env(&env(&[("OP_DB_APP_PASSWORD_FILE", file)])).unwrap();
        let options = direct.pg_config("a", false, true).unwrap();
        assert!(
            options
                .get_options()
                .unwrap()
                .contains("statement_timeout=30min")
        );
        assert!(
            direct
                .pg_config("a", false, false)
                .unwrap()
                .get_options()
                .is_none()
        );

        let pooled = DbSettings::from_env(&env(&[
            ("OP_DB_APP_PASSWORD_FILE", file),
            ("OP_DB_POOLER", "pgbouncer"),
        ]))
        .unwrap();
        assert!(
            pooled
                .pg_config("a", false, true)
                .unwrap()
                .get_options()
                .is_none()
        );
        // a direct connection (LISTEN, advisory lock) never meets the pooler
        assert!(
            pooled
                .pg_config("a", true, true)
                .unwrap()
                .get_options()
                .is_some()
        );
    }
}
