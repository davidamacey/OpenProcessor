//! The error types of the storage layer. Every message names the key or path it is about, and
//! none carries a credential.

use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

use op_config::EnvError;

use crate::keys::StorageKey;

pub use op_local_file::KeyError;

/// Why a storage operation failed.
#[derive(Debug)]
pub enum StorageError {
    /// A file operation failed.
    Io {
        /// What was being done.
        op: &'static str,
        /// The path it was done to.
        path: PathBuf,
        /// The operating system's answer.
        source: io::Error,
    },
    /// A path handed in is not inside the landing root it must be in.
    OutsideRoot {
        /// The path as given.
        path: PathBuf,
    },
    /// The object store refused or failed an operation. The message names the store, never a
    /// credential.
    Store {
        /// What was being done.
        op: &'static str,
        /// The key it was done to.
        key: String,
        /// The store's answer.
        source: object_store::Error,
    },
    /// The object under a key has a different size than the bytes it must hold. Nothing is
    /// overwritten.
    SizeMismatch {
        /// The key.
        key: String,
        /// The size the bytes have.
        expected: u64,
        /// The size found.
        found: u64,
    },
    /// No object has this key.
    NotFound {
        /// The key.
        key: String,
    },
    /// A setting is missing or does not parse.
    Setting(EnvError),
    /// A setting is set to a value that cannot be used.
    Config {
        /// The setting.
        key: &'static str,
        /// What is wrong with it.
        reason: String,
    },
    /// A key was refused.
    Key(KeyError),
    /// The operation is not allowed on this key.
    Refused {
        /// The key.
        key: String,
        /// Why not.
        reason: &'static str,
    },
}

impl StorageError {
    pub(crate) fn store(op: &'static str, key: &StorageKey, source: object_store::Error) -> Self {
        Self::Store {
            op,
            key: key.to_string(),
            source,
        }
    }

    pub(crate) fn io(op: &'static str, path: &std::path::Path, source: io::Error) -> Self {
        Self::Io {
            op,
            path: path.to_path_buf(),
            source,
        }
    }
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { op, path, source } => {
                write!(f, "{op} of {} failed: {source}", path.display())
            }
            Self::OutsideRoot { path } => {
                write!(f, "{} is not inside the landing root", path.display())
            }
            Self::Store { op, key, source } => write!(f, "{op} of {key:?} failed: {source}"),
            Self::SizeMismatch {
                key,
                expected,
                found,
            } => write!(
                f,
                "object {key:?} holds {found} bytes, not the {expected} that were to be stored"
            ),
            Self::NotFound { key } => write!(f, "no object {key:?}"),
            Self::Setting(e) => e.fmt(f),
            Self::Config { key, reason } => write!(f, "{key} cannot be used: {reason}"),
            Self::Key(e) => e.fmt(f),
            Self::Refused { key, reason } => write!(f, "{key:?}: {reason}"),
        }
    }
}

impl Error for StorageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Store { source, .. } => Some(source),
            Self::Setting(e) => Some(e),
            Self::Key(e) => Some(e),
            Self::OutsideRoot { .. }
            | Self::SizeMismatch { .. }
            | Self::NotFound { .. }
            | Self::Config { .. }
            | Self::Refused { .. } => None,
        }
    }
}
