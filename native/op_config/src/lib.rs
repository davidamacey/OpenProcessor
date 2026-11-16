//! Reading `OP_*` settings from the environment into typed structs through one helper, and the
//! refusal of retired variables. Only the helper ([`Env`]) exists so far; the crate also holds the
//! one function that installs the process's TLS crypto provider ([`install_tls_provider`]),
//! because it is the one dependency every crate that speaks TLS already has.
#![forbid(unsafe_code)]

mod tls;

pub use tls::install_tls_provider;

use std::collections::HashMap;
use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::str::FromStr;

/// Why a setting could not be read. Every variant names the key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvError {
    /// A required setting is not set.
    Missing {
        /// The environment variable name.
        key: String,
    },
    /// The value is set but does not parse as the wanted type.
    Invalid {
        /// The environment variable name.
        key: String,
        /// The value as found.
        value: String,
        /// What the parser said.
        reason: String,
    },
    /// The value is not valid Unicode, so it cannot be shown or parsed.
    NotUnicode {
        /// The environment variable name.
        key: String,
    },
}

impl fmt::Display for EnvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { key } => write!(f, "{key} is required and not set"),
            Self::Invalid { key, value, reason } => {
                write!(f, "{key} has the invalid value {value:?}: {reason}")
            }
            Self::NotUnicode { key } => write!(f, "{key} is not valid Unicode"),
        }
    }
}

impl Error for EnvError {}

/// A snapshot of the process environment. Settings are read from the snapshot, so a value that
/// changes after startup is never seen half-way and tests build one without touching the process.
#[derive(Debug, Clone, Default)]
pub struct Env {
    vars: HashMap<String, OsString>,
}

impl Env {
    /// Snapshot the environment of this process. Variables whose name is not valid Unicode are
    /// left out: no setting has such a name.
    #[must_use]
    pub fn from_process() -> Self {
        let vars = std::env::vars_os()
            .filter_map(|(k, v)| k.into_string().ok().map(|k| (k, v)))
            .collect();
        Self { vars }
    }

    /// Build a snapshot from key and value pairs (tests, and callers that load a file).
    #[must_use]
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<OsString>,
    {
        let vars = pairs
            .into_iter()
            .map(|(k, v)| (k.into(), v.into()))
            .collect();
        Self { vars }
    }

    /// Read `key` as a `T`, or return `default` when it is not set.
    ///
    /// # Errors
    ///
    /// [`EnvError::Invalid`] when the key is set to a value `T` does not parse (an empty value
    /// counts as set), and [`EnvError::NotUnicode`] when the value is not valid Unicode.
    pub fn get<T>(&self, key: &str, default: T) -> Result<T, EnvError>
    where
        T: FromStr,
        T::Err: fmt::Display,
    {
        match self.raw(key)? {
            Some(raw) => parse(key, raw),
            None => Ok(default),
        }
    }

    /// Read `key` as a `T`; the key must be set.
    ///
    /// # Errors
    ///
    /// [`EnvError::Missing`] when the key is not set, otherwise the errors of [`Env::get`].
    pub fn require<T>(&self, key: &str) -> Result<T, EnvError>
    where
        T: FromStr,
        T::Err: fmt::Display,
    {
        match self.raw(key)? {
            Some(raw) => parse(key, raw),
            None => Err(EnvError::Missing { key: key.into() }),
        }
    }

    fn raw(&self, key: &str) -> Result<Option<&str>, EnvError> {
        self.vars.get(key).map_or(Ok(None), |v| {
            v.to_str()
                .map(Some)
                .ok_or_else(|| EnvError::NotUnicode { key: key.into() })
        })
    }
}

fn parse<T>(key: &str, raw: &str) -> Result<T, EnvError>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    raw.parse().map_err(|e: T::Err| EnvError::Invalid {
        key: key.into(),
        value: raw.into(),
        reason: e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env {
        Env::from_pairs([("OP_PORT", "4603"), ("OP_BAD", "x1"), ("OP_NAME", "api")])
    }

    #[test]
    fn get_returns_the_default_when_unset_and_the_value_when_set() {
        assert_eq!(env().get("OP_PORT", 1_u16).unwrap(), 4603);
        assert_eq!(env().get("OP_OTHER", 7_u16).unwrap(), 7);
        assert_eq!(env().get("OP_NAME", String::new()).unwrap(), "api");
    }

    #[test]
    fn a_missing_required_key_names_itself() {
        let err = env().require::<u16>("OP_ABSENT").unwrap_err();
        assert_eq!(
            err,
            EnvError::Missing {
                key: "OP_ABSENT".into()
            }
        );
        assert!(err.to_string().contains("OP_ABSENT"));
    }

    #[test]
    fn a_bad_value_names_the_key_and_the_value() {
        let msg = env().get("OP_BAD", 1_u16).unwrap_err().to_string();
        assert!(msg.contains("OP_BAD") && msg.contains("x1"), "{msg}");
        let msg = env().require::<u16>("OP_BAD").unwrap_err().to_string();
        assert!(msg.contains("OP_BAD") && msg.contains("x1"), "{msg}");
    }

    #[test]
    fn an_empty_value_is_set_and_not_the_default() {
        let env = Env::from_pairs([("OP_PORT", "")]);
        assert!(matches!(
            env.get("OP_PORT", 1_u16),
            Err(EnvError::Invalid { .. })
        ));
    }

    #[test]
    fn a_non_unicode_value_is_refused() {
        use std::os::unix::ffi::OsStringExt;
        let env = Env::from_pairs([("OP_NAME", OsString::from_vec(vec![0xff, 0xfe]))]);
        assert_eq!(
            env.require::<String>("OP_NAME").unwrap_err(),
            EnvError::NotUnicode {
                key: "OP_NAME".into()
            }
        );
    }

    #[test]
    fn from_process_sees_the_real_environment() {
        let path = Env::from_process().require::<String>("PATH").unwrap();
        assert!(!path.is_empty());
    }
}
