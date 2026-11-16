//! Local read paths (plan 18 section 4.4).
//!
//! The engine and ExifTool need a file path. Two kinds of key have one: landing keys and `source/`
//! keys. A key is validated as a [`StorageKey`], joined to its configured root, canonicalised
//! (which follows every symlink), and refused unless the result is inside the canonical root.
//! `object_store`'s `LocalFileSystem` follows symlinks that leave its root, which is why this
//! check lives here and the key layout never creates a symlink.

use std::error::Error;
use std::fmt;
use std::io;
use std::path::{Path, PathBuf};

use op_config::{Env, EnvError};

use crate::error::KeyError;
use crate::keys::{Root, StorageKey};

/// The setting naming the landing directory.
pub const LANDING_ROOT_KEY: &str = "OP_LANDING_ROOT";
/// The existing setting naming the directory photos are read from in place.
pub const SOURCE_ROOT_KEY: &str = "OP_SOURCE_ROOT";

/// Why the local roots could not be set up.
#[derive(Debug)]
pub enum RootsError {
    /// A setting is missing or does not parse.
    Setting(EnvError),
    /// A root is not an existing directory.
    Root {
        /// The root's role, `landing` or `source`.
        role: &'static str,
        /// The path as configured.
        path: PathBuf,
        /// The operating system's answer, or `None` when the path exists but is no directory.
        source: Option<io::Error>,
    },
}

impl fmt::Display for RootsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Setting(e) => e.fmt(f),
            Self::Root {
                role,
                path,
                source: Some(e),
            } => write!(f, "the {role} root {} is not usable: {e}", path.display()),
            Self::Root { role, path, .. } => {
                write!(f, "the {role} root {} is not a directory", path.display())
            }
        }
    }
}

impl Error for RootsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Setting(e) => Some(e),
            Self::Root { source, .. } => source.as_ref().map(|e| e as &(dyn Error + 'static)),
        }
    }
}

/// The two local roots, both canonical existing directories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalRoots {
    /// Where uploads land before they are made durable.
    pub landing: PathBuf,
    /// Where source photos are read in place, never written.
    pub sources: PathBuf,
}

impl LocalRoots {
    /// Canonicalise both roots; each must be an existing directory.
    ///
    /// # Errors
    ///
    /// [`RootsError::Root`] naming the role and path of the first root that is not a directory.
    pub fn new(landing: &Path, sources: &Path) -> Result<Self, RootsError> {
        Ok(Self {
            landing: canonical_dir("landing", landing)?,
            sources: canonical_dir("source", sources)?,
        })
    }

    /// Read `OP_LANDING_ROOT` and `OP_SOURCE_ROOT`; neither has a default.
    ///
    /// # Errors
    ///
    /// [`RootsError::Setting`] naming the missing key, [`RootsError::Root`] when a directory is
    /// not usable.
    pub fn from_env(env: &Env) -> Result<Self, RootsError> {
        let landing: String = env.require(LANDING_ROOT_KEY).map_err(RootsError::Setting)?;
        let sources: String = env.require(SOURCE_ROOT_KEY).map_err(RootsError::Setting)?;
        Self::new(Path::new(&landing), Path::new(&sources))
    }

    /// The file a landing or source key names, canonical and inside its root.
    ///
    /// This asks the file system (`canonicalize`), so an async caller runs it on the blocking
    /// pool. The answer is a path for a read; a symlink swapped in afterwards is not seen, which
    /// is why no code of ours creates one under a root.
    ///
    /// # Errors
    ///
    /// [`KeyError::NotLocal`] for a root with no local directory, [`KeyError::Unresolvable`] when
    /// the file does not exist or cannot be read, [`KeyError::OutsideRoot`] when it resolves
    /// outside the root (a symlink that leaves it).
    pub fn resolve(&self, key: &StorageKey) -> Result<PathBuf, KeyError> {
        let root = match key.root() {
            Root::Landing => &self.landing,
            Root::Source => &self.sources,
            _ => {
                return Err(KeyError::NotLocal {
                    key: key.to_string(),
                });
            }
        };
        let joined = key.segments().fold(root.clone(), |p, seg| p.join(seg));
        let resolved = std::fs::canonicalize(&joined).map_err(|source| KeyError::Unresolvable {
            key: key.to_string(),
            source,
        })?;
        if resolved.starts_with(root) {
            Ok(resolved)
        } else {
            Err(KeyError::OutsideRoot {
                key: key.to_string(),
            })
        }
    }
}

fn canonical_dir(role: &'static str, path: &Path) -> Result<PathBuf, RootsError> {
    let fail = |source| RootsError::Root {
        role,
        path: path.to_path_buf(),
        source,
    };
    let canonical = std::fs::canonicalize(path).map_err(|e| fail(Some(e)))?;
    if canonical.is_dir() {
        Ok(canonical)
    } else {
        Err(fail(None))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::symlink;

    use tempfile::TempDir;

    use super::*;
    use crate::keys::{ContentHash, ProjectId};

    struct Fixture {
        _dir: TempDir,
        outside: PathBuf,
        roots: LocalRoots,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let base = fs::canonicalize(dir.path()).unwrap();
        for d in ["landing", "sources", "outside"] {
            fs::create_dir(base.join(d)).unwrap();
        }
        fs::write(base.join("outside/secret"), b"secret").unwrap();
        let roots = LocalRoots::new(&base.join("landing"), &base.join("sources")).unwrap();
        Fixture {
            outside: base.join("outside"),
            roots,
            _dir: dir,
        }
    }

    #[test]
    fn a_source_key_resolves_to_its_canonical_file() {
        let f = fixture();
        fs::create_dir(f.roots.sources.join("trip")).unwrap();
        fs::write(f.roots.sources.join("trip/a.jpg"), b"x").unwrap();
        let got = f
            .roots
            .resolve(&StorageKey::source("trip/a.jpg").unwrap())
            .unwrap();
        assert_eq!(got, f.roots.sources.join("trip/a.jpg"));
    }

    #[test]
    fn a_landing_key_resolves_to_its_file() {
        let f = fixture();
        let upload = uuid::Uuid::from_u128(5);
        let key = StorageKey::landing(ProjectId(2), &ContentHash([9; 32]), upload, 1);
        let dir = f.roots.landing.join("2");
        fs::create_dir(&dir).unwrap();
        fs::write(
            dir.join(format!("{}.{upload}.1", ContentHash([9; 32]))),
            b"x",
        )
        .unwrap();
        assert!(f.roots.resolve(&key).unwrap().starts_with(&f.roots.landing));
    }

    #[test]
    fn a_symlinked_file_that_leaves_the_root_is_refused() {
        let f = fixture();
        symlink(f.outside.join("secret"), f.roots.sources.join("link")).unwrap();
        let err = f
            .roots
            .resolve(&StorageKey::source("link").unwrap())
            .unwrap_err();
        assert!(matches!(err, KeyError::OutsideRoot { .. }), "{err}");
    }

    #[test]
    fn a_symlinked_folder_that_leaves_the_root_is_refused() {
        let f = fixture();
        symlink(&f.outside, f.roots.landing.join("dir")).unwrap();
        let err = f
            .roots
            .resolve(&StorageKey::parse("landing/dir/secret").unwrap())
            .unwrap_err();
        assert!(matches!(err, KeyError::OutsideRoot { .. }), "{err}");
    }

    #[test]
    fn a_symlink_that_stays_inside_the_root_is_followed() {
        let f = fixture();
        fs::write(f.roots.sources.join("real.jpg"), b"x").unwrap();
        symlink("real.jpg", f.roots.sources.join("alias.jpg")).unwrap();
        let got = f
            .roots
            .resolve(&StorageKey::source("alias.jpg").unwrap())
            .unwrap();
        assert_eq!(got, f.roots.sources.join("real.jpg"));
    }

    #[test]
    fn a_missing_file_and_a_root_with_no_directory_are_refused() {
        let f = fixture();
        let missing = f.roots.resolve(&StorageKey::source("nope.jpg").unwrap());
        assert!(matches!(missing, Err(KeyError::Unresolvable { .. })));
        let projects = StorageKey::parse("projects/1/originals/ab/cd").unwrap();
        assert!(matches!(
            f.roots.resolve(&projects),
            Err(KeyError::NotLocal { .. })
        ));
    }

    #[test]
    fn a_sibling_directory_that_shares_the_root_name_as_prefix_is_outside() {
        let f = fixture();
        let sibling = f.roots.sources.with_file_name("sources_evil");
        fs::create_dir(&sibling).unwrap();
        fs::write(sibling.join("x"), b"x").unwrap();
        symlink(&sibling, f.roots.sources.join("evil")).unwrap();
        let err = f
            .roots
            .resolve(&StorageKey::parse("source/evil/x").unwrap())
            .unwrap_err();
        assert!(matches!(err, KeyError::OutsideRoot { .. }), "{err}");
    }

    #[test]
    fn roots_come_from_the_settings_and_a_missing_one_names_its_key() {
        let f = fixture();
        let env = Env::from_pairs([
            (LANDING_ROOT_KEY, f.roots.landing.as_os_str()),
            (SOURCE_ROOT_KEY, f.roots.sources.as_os_str()),
        ]);
        assert_eq!(LocalRoots::from_env(&env).unwrap(), f.roots);

        let only_landing = Env::from_pairs([(LANDING_ROOT_KEY, f.roots.landing.as_os_str())]);
        let msg = LocalRoots::from_env(&only_landing).unwrap_err().to_string();
        assert!(msg.contains("OP_SOURCE_ROOT"), "{msg}");
        let msg = LocalRoots::from_env(&Env::default())
            .unwrap_err()
            .to_string();
        assert!(msg.contains("OP_LANDING_ROOT"), "{msg}");
    }

    #[test]
    fn a_root_that_is_not_a_directory_is_refused_by_role() {
        let f = fixture();
        let file = f.roots.sources.join("a_file");
        fs::write(&file, b"x").unwrap();
        let err = LocalRoots::new(&f.roots.landing, &file).unwrap_err();
        assert!(err.to_string().contains("source root"), "{err}");
        let err = LocalRoots::new(&f.roots.landing.join("absent"), &f.roots.sources).unwrap_err();
        assert!(err.to_string().contains("landing root"), "{err}");
    }
}
