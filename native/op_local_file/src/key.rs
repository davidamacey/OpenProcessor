//! The rule a storage key must satisfy (plan 18 section 4.3) and the errors of a key.
//!
//! A key is `/`-separated, relative, and its first segment is a [`Root`]. The rule is the one
//! `object_store::path::Path::parse` applies (no empty segment, no `.` or `..`, no control
//! character) with three more refusals so that what a record stores and what the object store
//! sees are the same string: no leading or trailing `/`, no backslash, and a root with nothing
//! under it is no object. `op_storage` relies on this being at least as strict as `Path::parse`
//! and checks it by a property test.

use std::error::Error;
use std::fmt;
use std::io;

/// The first segment of a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Root {
    /// The landing zone on local fast disk (never in the object store).
    Landing,
    /// Source roots read in place; never written or erased.
    Source,
    /// Everything a project owns in the object store.
    Projects,
    /// Models.
    Models,
    /// Bake-off results.
    BakeoffResults,
    /// Caches.
    Cache,
}

impl Root {
    /// Every root, in the order of the key layout.
    pub const ALL: [Self; 6] = [
        Self::Landing,
        Self::Source,
        Self::Projects,
        Self::Models,
        Self::BakeoffResults,
        Self::Cache,
    ];

    /// The key segment of this root.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Landing => "landing",
            Self::Source => "source",
            Self::Projects => "projects",
            Self::Models => "models",
            Self::BakeoffResults => "bakeoff_results",
            Self::Cache => "cache",
        }
    }

    /// Whether keys under this root name a file on a local disk (the landing zone and the source
    /// roots); every other root lives in the object store.
    #[must_use]
    pub fn is_local(self) -> bool {
        matches!(self, Self::Landing | Self::Source)
    }

    fn from_segment(segment: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.as_str() == segment)
    }
}

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

/// Check `key` against the rule and name its root.
///
/// # Errors
///
/// [`KeyError::UnknownRoot`] when the first segment is not a root, [`KeyError::Invalid`] for
/// every other refusal.
pub fn validate_key(key: &str) -> Result<Root, KeyError> {
    let invalid = |reason: &str| KeyError::Invalid {
        key: key.to_owned(),
        reason: reason.to_owned(),
    };
    if key.is_empty() {
        return Err(invalid("a key must not be empty"));
    }
    if key.contains('\\') {
        return Err(invalid("a backslash is not allowed"));
    }
    if key.starts_with('/') || key.ends_with('/') {
        return Err(invalid("a leading or trailing '/' is not allowed"));
    }
    let (first, rest) = key.split_once('/').unwrap_or((key, ""));
    for segment in key.split('/') {
        if segment.is_empty() {
            return Err(invalid("an empty segment is not allowed"));
        }
        if segment == "." || segment == ".." {
            return Err(invalid("'.' and '..' are not allowed"));
        }
        if segment.chars().any(|c| c.is_ascii_control()) {
            return Err(invalid("a control character is not allowed"));
        }
    }
    let root = Root::from_segment(first).ok_or_else(|| KeyError::UnknownRoot {
        key: key.to_owned(),
    })?;
    if rest.is_empty() {
        return Err(invalid("a root with nothing under it is not an object"));
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_with_a_known_root_and_something_under_it_is_valid() {
        for root in Root::ALL {
            assert_eq!(validate_key(&format!("{}/x", root.as_str())).unwrap(), root);
        }
        assert_eq!(validate_key("source/trip/IMG 1.jpg").unwrap(), Root::Source);
    }

    #[test]
    fn what_could_leave_a_root_or_is_not_in_canonical_form_is_refused() {
        for bad in [
            "",
            "source",
            "source/",
            "/source/a",
            "source/a/",
            "source//a",
            "source/../a",
            "source/./a",
            "source/a/..",
            "source/a\\b",
            "source/a\0b",
            "source/a\nb",
            "source/a\x7fb",
            "projects/1/originals/../../x",
        ] {
            assert!(
                matches!(validate_key(bad), Err(KeyError::Invalid { .. })),
                "{bad:?} must be refused"
            );
        }
    }

    #[test]
    fn an_unknown_or_miscased_root_is_named_as_such() {
        for bad in ["upgrade/a", "Source/a", "sources/a"] {
            assert!(
                matches!(validate_key(bad), Err(KeyError::UnknownRoot { .. })),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn only_the_landing_zone_and_the_source_roots_are_local() {
        let local: Vec<_> = Root::ALL.into_iter().filter(|r| r.is_local()).collect();
        assert_eq!(local, [Root::Landing, Root::Source]);
    }
}
