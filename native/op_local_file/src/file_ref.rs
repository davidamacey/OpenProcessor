//! A reference to a photo file the engine reads itself: a key under the landing zone or a source
//! root, and the claim the server made for the file.

use crate::claim::FileClaim;
use crate::key::{KeyError, Root, validate_key};

/// A validated local key with the claim for its file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileRef {
    key: String,
    root: Root,
    claim: FileClaim,
}

impl FileRef {
    /// `key` must satisfy [`validate_key`] and name a local root (the landing zone or a source
    /// root).
    ///
    /// # Errors
    ///
    /// [`KeyError::Invalid`] or [`KeyError::UnknownRoot`] from the key rule, and
    /// [`KeyError::NotLocal`] for a root that lives in the object store.
    pub fn new(key: &str, claim: FileClaim) -> Result<Self, KeyError> {
        let root = validate_key(key)?;
        if !root.is_local() {
            return Err(KeyError::NotLocal {
                key: key.to_owned(),
            });
        }
        Ok(Self {
            key: key.to_owned(),
            root,
            claim,
        })
    }

    /// The key, `landing/...` or `source/...`.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// The key's root: [`Root::Landing`] or [`Root::Source`].
    #[must_use]
    pub fn root(&self) -> Root {
        self.root
    }

    /// What the server knows about the file's bytes.
    #[must_use]
    pub fn claim(&self) -> &FileClaim {
        &self.claim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim() -> FileClaim {
        FileClaim::of(b"abc")
    }

    #[test]
    fn a_landing_or_source_key_is_a_reference() {
        for (key, root) in [
            ("landing/7/x.0", Root::Landing),
            ("source/a.jpg", Root::Source),
        ] {
            let file = FileRef::new(key, claim()).unwrap();
            assert_eq!(
                (file.key(), file.root(), *file.claim()),
                (key, root, claim())
            );
        }
    }

    #[test]
    fn an_object_store_key_a_bad_key_and_an_unknown_root_are_refused_each_its_way() {
        assert!(matches!(
            FileRef::new("projects/1/originals/ab/cd", claim()),
            Err(KeyError::NotLocal { .. })
        ));
        assert!(matches!(
            FileRef::new("source/../a", claim()),
            Err(KeyError::Invalid { .. })
        ));
        assert!(matches!(
            FileRef::new("upload/a", claim()),
            Err(KeyError::UnknownRoot { .. })
        ));
    }
}
