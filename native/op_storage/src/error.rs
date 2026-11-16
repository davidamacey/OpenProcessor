//! The error types of the storage layer. Every message names the key or path it is about, and
//! none carries a credential.

use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;

/// Why a storage key was refused or could not be turned into a local path.
#[derive(Debug)]
pub enum KeyError {
    /// The key is not a valid key: empty, a leading or trailing `/`, an empty segment, `.` or
    /// `..`, a control character or a backslash, or a root with nothing under it.
    Invalid {
        /// The key as given.
        key: String,
        /// What is wrong with it.
        reason: String,
    },
    /// The first segment is not one of the roots of the key layout.
    UnknownRoot {
        /// The key as given.
        key: String,
    },
    /// The key's root has no local directory, so it cannot be turned into a path.
    NotLocal {
        /// The key as given.
        key: String,
    },
    /// The key resolves to a place outside its root (for example through a symlink).
    OutsideRoot {
        /// The key as given.
        key: String,
    },
    /// The key's path could not be resolved on disk (it does not exist, or cannot be read).
    Unresolvable {
        /// The key as given.
        key: String,
        /// The operating system's answer.
        source: io::Error,
    },
}

impl fmt::Display for KeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { key, reason } => write!(f, "storage key {key:?} is invalid: {reason}"),
            Self::UnknownRoot { key } => write!(f, "storage key {key:?} has an unknown root"),
            Self::NotLocal { key } => {
                write!(f, "storage key {key:?} is not under a local root")
            }
            Self::OutsideRoot { key } => {
                write!(f, "storage key {key:?} resolves outside its root")
            }
            Self::Unresolvable { key, source } => {
                write!(f, "storage key {key:?} cannot be resolved: {source}")
            }
        }
    }
}

impl Error for KeyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Unresolvable { source, .. } => Some(source),
            _ => None,
        }
    }
}

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
}

impl StorageError {
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
        }
    }
}

impl Error for StorageError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::OutsideRoot { .. } => None,
        }
    }
}
