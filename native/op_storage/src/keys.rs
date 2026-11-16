//! The key layout of plan 18 section 4.3: by project and content.
//!
//! A [`StorageKey`] is a `/`-separated relative key whose first segment is a root. It is
//! validated with [`object_store::path::Path::parse`] (no `.`, `..`, empty segments or control
//! characters) and is refused when it is not in its canonical form, so what is stored in the
//! record and what the object store sees are the same string.

use std::fmt;

use object_store::path::Path;

use crate::error::KeyError;

/// The id of a project (not its slug, which can be renamed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProjectId(pub i64);

impl fmt::Display for ProjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A SHA-256 digest. Displays as 64 lowercase hex characters.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentHash(pub [u8; 32]);

impl ContentHash {
    /// The first two hex characters, the directory level of the layout.
    #[must_use]
    pub fn prefix(&self) -> String {
        format!("{:02x}", self.0[0])
    }
}

impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl fmt::Debug for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentHash({self})")
    }
}

/// The kinds of derived file (plan 18 section 4.3). Derived files are deleted by prefix and
/// crops and thumbnails are a cache that can be made again from the original.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DerivedKind {
    /// Crops of detected objects.
    Crops,
    /// Thumbnails of photos.
    Thumbs,
    /// Exports.
    Exports,
    /// Training copies.
    Training,
    /// Holdout snapshots.
    Snapshots,
}

impl DerivedKind {
    /// The key segment of this kind.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Crops => "crops",
            Self::Thumbs => "thumbs",
            Self::Exports => "exports",
            Self::Training => "training",
            Self::Snapshots => "snapshots",
        }
    }
}

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
    const ALL: [Self; 6] = [
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

    fn from_segment(segment: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.as_str() == segment)
    }
}

/// A validated storage key, or a prefix of keys (for delete by prefix).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StorageKey {
    path: Path,
    root: Root,
}

impl StorageKey {
    /// Validate `key`: a known root, then at least one more segment, every segment a legal path
    /// part, no leading or trailing `/`, no backslash.
    ///
    /// # Errors
    ///
    /// [`KeyError::UnknownRoot`] when the first segment is not a root, [`KeyError::Invalid`] for
    /// every other refusal.
    pub fn parse(key: &str) -> Result<Self, KeyError> {
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
        let path = Path::parse(key).map_err(|e| invalid(&e.to_string()))?;
        if path.as_ref() != key {
            return Err(invalid("a leading or trailing '/' is not allowed"));
        }
        let (first, rest) = key.split_once('/').unwrap_or((key, ""));
        let root = Root::from_segment(first).ok_or_else(|| KeyError::UnknownRoot {
            key: key.to_owned(),
        })?;
        if rest.is_empty() {
            return Err(invalid("a root with nothing under it is not an object"));
        }
        Ok(Self { path, root })
    }

    /// The original of an upload: `projects/<project>/originals/<sha256[0:2]>/<sha256>`.
    #[must_use]
    pub fn original(project: ProjectId, hash: &ContentHash) -> Self {
        Self::from_safe_parts(
            Root::Projects,
            [
                project.to_string(),
                "originals".to_owned(),
                hash.prefix(),
                hash.to_string(),
            ],
        )
    }

    /// A derived file: `projects/<project>/derived/<kind>/<photo_key>/<name>`. `name` may hold
    /// `/` to make folders.
    ///
    /// # Errors
    ///
    /// [`KeyError::Invalid`] when `name` is empty or has an illegal segment.
    pub fn derived(
        project: ProjectId,
        kind: DerivedKind,
        photo_key: i64,
        name: &str,
    ) -> Result<Self, KeyError> {
        Self::parse(&format!(
            "projects/{project}/derived/{}/{photo_key}/{name}",
            kind.as_str()
        ))
    }

    /// A file under a source root, read in place: `source/<rel>`.
    ///
    /// # Errors
    ///
    /// [`KeyError::Invalid`] when `rel` is empty or has an illegal segment.
    pub fn source(rel: &str) -> Result<Self, KeyError> {
        Self::parse(&format!("source/{rel}"))
    }

    /// A photo in the landing zone, named by its content: `landing/<project>/<sha256>`.
    #[must_use]
    pub fn landing(project: ProjectId, hash: &ContentHash) -> Self {
        Self::from_safe_parts(Root::Landing, [project.to_string(), hash.to_string()])
    }

    fn from_safe_parts(root: Root, parts: impl IntoIterator<Item = String>) -> Self {
        let path = std::iter::once(root.as_str().to_owned())
            .chain(parts)
            .collect();
        Self { path, root }
    }

    /// The root this key is under.
    #[must_use]
    pub fn root(&self) -> Root {
        self.root
    }

    /// The key as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.path.as_ref()
    }

    /// The key as an object store path.
    #[must_use]
    pub fn as_path(&self) -> &Path {
        &self.path
    }

    /// The segments under the root.
    pub fn segments(&self) -> impl Iterator<Item = &str> {
        self.as_str().split('/').skip(1)
    }
}

impl fmt::Display for StorageKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash() -> ContentHash {
        let mut b = [0u8; 32];
        b[0] = 0xab;
        b[31] = 0x01;
        ContentHash(b)
    }

    #[test]
    fn the_original_key_is_by_project_id_and_content() {
        let key = StorageKey::original(ProjectId(7), &hash());
        let h = hash().to_string();
        assert_eq!(key.as_str(), format!("projects/7/originals/ab/{h}"));
        assert_eq!(key.root(), Root::Projects);
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn the_landing_key_is_by_project_id_and_content() {
        let key = StorageKey::landing(ProjectId(7), &hash());
        assert_eq!(key.as_str(), format!("landing/7/{}", hash()));
        assert_eq!(key.root(), Root::Landing);
    }

    #[test]
    fn derived_and_source_keys_follow_the_layout() {
        let d = StorageKey::derived(ProjectId(3), DerivedKind::Thumbs, 42, "256.jpg").unwrap();
        assert_eq!(d.as_str(), "projects/3/derived/thumbs/42/256.jpg");
        let d = StorageKey::derived(ProjectId(3), DerivedKind::Crops, 42, "a/b.jpg").unwrap();
        assert_eq!(d.as_str(), "projects/3/derived/crops/42/a/b.jpg");
        let s = StorageKey::source("trip/IMG_1.jpg").unwrap();
        assert_eq!(s.as_str(), "source/trip/IMG_1.jpg");
        assert_eq!(s.root(), Root::Source);
        assert_eq!(s.segments().collect::<Vec<_>>(), ["trip", "IMG_1.jpg"]);
    }

    #[test]
    fn every_derived_kind_has_its_own_segment() {
        let names: Vec<_> = [
            DerivedKind::Crops,
            DerivedKind::Thumbs,
            DerivedKind::Exports,
            DerivedKind::Training,
            DerivedKind::Snapshots,
        ]
        .map(DerivedKind::as_str)
        .into();
        assert_eq!(
            names,
            ["crops", "thumbs", "exports", "training", "snapshots"]
        );
    }

    #[test]
    fn parse_refuses_what_could_leave_a_root() {
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
            "projects/1/originals/../../x",
        ] {
            assert!(
                matches!(StorageKey::parse(bad), Err(KeyError::Invalid { .. })),
                "{bad:?} must be refused"
            );
        }
    }

    #[test]
    fn parse_refuses_an_unknown_root_and_accepts_every_known_one() {
        assert!(matches!(
            StorageKey::parse("upgrade/a"),
            Err(KeyError::UnknownRoot { .. })
        ));
        assert!(matches!(
            StorageKey::parse("Source/a"),
            Err(KeyError::UnknownRoot { .. })
        ));
        for root in Root::ALL {
            let key = StorageKey::parse(&format!("{}/x", root.as_str())).unwrap();
            assert_eq!(key.root(), root);
        }
    }

    #[test]
    fn constructors_refuse_a_name_that_leaves_its_folder() {
        assert!(StorageKey::source("../etc/passwd").is_err());
        assert!(StorageKey::source("").is_err());
        assert!(StorageKey::source("a//b").is_err());
        assert!(StorageKey::derived(ProjectId(1), DerivedKind::Crops, 1, "../x").is_err());
        assert!(StorageKey::derived(ProjectId(1), DerivedKind::Crops, 1, "").is_err());
    }

    #[test]
    fn a_parsed_key_prints_as_it_was_given() {
        let text = "projects/1/derived/crops/5/x.jpg";
        assert_eq!(StorageKey::parse(text).unwrap().to_string(), text);
    }
}
