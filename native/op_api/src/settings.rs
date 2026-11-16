//! The front's settings, read once at start from the environment snapshot ([`Env`]).
//!
//! Every value that bounds work has a default and a refusal for zero, so a typo cannot switch a
//! bound off. The keep-alive and the shutdown grace equal what uvicorn ran with in
//! `docker-compose.yml` (`--timeout-keep-alive=75`, `--timeout-graceful-shutdown=30`). The
//! concurrency limit does not: uvicorn's `--limit-concurrency=512` is per worker process and the
//! product ran many, so one limit of 512 for the whole front would shed load the product took.
//! The front's default is its connection bound; each Python worker keeps its own limit and its
//! 503 passes through the front.

use std::error::Error;
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use http::Uri;
use http::uri::{Authority, Scheme};
use op_config::{Env, EnvError};

/// Where the Python API listens, as `http://host:port`. Required: a front without an upstream has
/// nothing to forward to, so it refuses to start rather than guess.
pub const PY_API_URL: &str = "OP_PY_API_URL";
/// The address the front binds (`host:port`).
pub const LISTEN: &str = "OP_API_LISTEN";
/// Requests in flight at once; one more is answered 503 at once. `/live` and the front's metrics
/// path are not counted.
pub const CONCURRENCY_LIMIT: &str = "OP_API_CONCURRENCY_LIMIT";
/// Open client connections; past it the front stops accepting until one closes.
pub const MAX_CONNECTIONS: &str = "OP_API_MAX_CONNECTIONS";
/// Seconds an idle client connection is kept, and the most a client may take to send request
/// headers.
pub const KEEP_ALIVE_S: &str = "OP_API_KEEP_ALIVE_S";
/// Seconds in-flight requests get to finish after the stop signal.
pub const SHUTDOWN_GRACE_S: &str = "OP_API_SHUTDOWN_GRACE_S";
/// Seconds to connect to the Python API before the request is answered 503.
pub const UPSTREAM_CONNECT_TIMEOUT_S: &str = "OP_API_UPSTREAM_CONNECT_TIMEOUT_S";

/// Why the settings are refused. Every variant names the key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SettingsError {
    /// A key is missing or does not parse as its type.
    Env(EnvError),
    /// A key parses but its value is not allowed.
    Invalid {
        /// The environment variable name.
        key: &'static str,
        /// What is wrong with the value.
        reason: String,
    },
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Env(e) => e.fmt(f),
            Self::Invalid { key, reason } => write!(f, "{key}: {reason}"),
        }
    }
}

impl Error for SettingsError {}

impl From<EnvError> for SettingsError {
    fn from(e: EnvError) -> Self {
        Self::Env(e)
    }
}

/// The Python API's address: the scheme and authority every forwarded request is sent to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Upstream {
    /// Always `http`: the Python API is reached inside the compose network.
    pub scheme: Scheme,
    /// Host and port.
    pub authority: Authority,
}

/// Everything the `api` role reads from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiSettings {
    /// The Python API requests are forwarded to.
    pub upstream: Upstream,
    /// The address the front binds.
    pub listen: SocketAddr,
    /// Requests in flight at once before the 503 (the probes are exempt).
    pub concurrency_limit: usize,
    /// Open client connections at most.
    pub max_connections: usize,
    /// Idle connection and request-header deadline.
    pub keep_alive: Duration,
    /// Time in-flight requests get to finish at shutdown.
    pub shutdown_grace: Duration,
    /// Time to connect to the Python API.
    pub upstream_connect_timeout: Duration,
}

impl ApiSettings {
    /// The address the front binds when [`LISTEN`] is not set: every interface, port 8000 (the
    /// container's port that compose publishes as the API port).
    pub const DEFAULT_LISTEN: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8000);

    /// The default of both the connection bound and the concurrency limit: with the same number, a
    /// connection that is open can always be answered, and the Python workers' own limits are what
    /// shed.
    pub const DEFAULT_MAX_CONNECTIONS: u64 = 4096;

    /// Read the settings.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Env`] when [`PY_API_URL`] is not set or a value does not parse;
    /// [`SettingsError::Invalid`] when the URL is not `http://host:port` without a path, or a
    /// bound is zero.
    pub fn from_env(env: &Env) -> Result<Self, SettingsError> {
        let upstream = parse_upstream(&env.require::<String>(PY_API_URL)?)?;
        let listen = env.get(LISTEN, Self::DEFAULT_LISTEN)?;
        Ok(Self {
            upstream,
            listen,
            concurrency_limit: count(env, CONCURRENCY_LIMIT, Self::DEFAULT_MAX_CONNECTIONS)?,
            max_connections: count(env, MAX_CONNECTIONS, Self::DEFAULT_MAX_CONNECTIONS)?,
            keep_alive: seconds(env, KEEP_ALIVE_S, 75)?,
            shutdown_grace: seconds(env, SHUTDOWN_GRACE_S, 30)?,
            upstream_connect_timeout: seconds(env, UPSTREAM_CONNECT_TIMEOUT_S, 5)?,
        })
    }
}

fn bound(env: &Env, key: &'static str, default: u64) -> Result<u64, SettingsError> {
    match env.get(key, default)? {
        0 => Err(SettingsError::Invalid {
            key,
            reason: "must be at least 1".into(),
        }),
        n => Ok(n),
    }
}

fn count(env: &Env, key: &'static str, default: u64) -> Result<usize, SettingsError> {
    usize::try_from(bound(env, key, default)?).map_err(|_| SettingsError::Invalid {
        key,
        reason: "too large".into(),
    })
}

fn seconds(env: &Env, key: &'static str, default: u64) -> Result<Duration, SettingsError> {
    bound(env, key, default).map(Duration::from_secs)
}

fn parse_upstream(text: &str) -> Result<Upstream, SettingsError> {
    let invalid = |reason: &str| SettingsError::Invalid {
        key: PY_API_URL,
        reason: format!("{reason} (got {text:?})"),
    };
    let uri: Uri = text
        .parse()
        .map_err(|_| invalid("not a URL like http://host:port"))?;
    if uri.scheme() != Some(&Scheme::HTTP) {
        return Err(invalid("the scheme must be http"));
    }
    if !matches!(uri.path(), "" | "/") || uri.query().is_some() {
        return Err(invalid("must have no path or query"));
    }
    let authority = uri.authority().cloned().ok_or_else(|| invalid("no host"))?;
    if authority.port_u16().is_none() {
        return Err(invalid("no port"));
    }
    Ok(Upstream {
        scheme: Scheme::HTTP,
        authority,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(extra: &[(&str, &str)]) -> Env {
        let mut pairs = vec![(PY_API_URL, "http://api-py:4613")];
        pairs.extend_from_slice(extra);
        Env::from_pairs(pairs)
    }

    #[test]
    fn the_defaults_are_what_the_product_ran_with() {
        let s = ApiSettings::from_env(&env(&[])).unwrap();
        assert_eq!(s.upstream.authority.as_str(), "api-py:4613");
        assert_eq!(s.listen, ApiSettings::DEFAULT_LISTEN);
        assert_eq!(s.concurrency_limit, 4096);
        assert_eq!(s.keep_alive, Duration::from_secs(75));
        assert_eq!(s.shutdown_grace, Duration::from_secs(30));
    }

    #[test]
    fn a_missing_upstream_names_the_key() {
        let err = ApiSettings::from_env(&Env::from_pairs([("OP_X", "1")])).unwrap_err();
        assert!(err.to_string().contains(PY_API_URL), "{err}");
    }

    #[test]
    fn a_bad_upstream_is_refused_with_the_key_and_the_value() {
        for bad in [
            "api-py:4613",
            "https://api-py:4613",
            "http://api-py",
            "http://api-py:4613/v1",
            "",
        ] {
            let err = ApiSettings::from_env(&Env::from_pairs([(PY_API_URL, bad)])).unwrap_err();
            let text = err.to_string();
            assert!(
                text.contains(PY_API_URL) && text.contains(bad),
                "{bad}: {text}"
            );
        }
    }

    #[test]
    fn a_zero_bound_is_refused_not_taken_as_unlimited() {
        for key in [
            CONCURRENCY_LIMIT,
            MAX_CONNECTIONS,
            KEEP_ALIVE_S,
            SHUTDOWN_GRACE_S,
            UPSTREAM_CONNECT_TIMEOUT_S,
        ] {
            let err = ApiSettings::from_env(&env(&[(key, "0")])).unwrap_err();
            assert!(err.to_string().contains(key), "{key}: {err}");
        }
    }

    #[test]
    fn a_value_that_does_not_parse_names_the_key() {
        let err = ApiSettings::from_env(&env(&[(LISTEN, "everywhere")])).unwrap_err();
        assert!(err.to_string().contains(LISTEN), "{err}");
        let err = ApiSettings::from_env(&env(&[(CONCURRENCY_LIMIT, "many")])).unwrap_err();
        assert!(err.to_string().contains(CONCURRENCY_LIMIT), "{err}");
    }
}
