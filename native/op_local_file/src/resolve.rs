//! The canonical-prefix rule of plan 18 section 4.4: a key is joined to its root's directory,
//! canonicalised (which follows every symlink), and refused unless the result is inside the
//! canonical directory. `object_store`'s `LocalFileSystem` follows symlinks that leave its root,
//! which is why the check is here and the key layout never creates a symlink.

use std::io;
use std::path::{Path, PathBuf};

use crate::key::{KeyError, validate_key};

/// Why a root directory could not be used.
#[derive(Debug)]
pub enum CanonicalDirError {
    /// The path could not be resolved: it does not exist or cannot be read.
    Io(io::Error),
    /// The path exists but is no directory.
    NotADirectory,
}

/// `path` as a canonical directory, the form [`resolve_under`] needs its root in.
///
/// # Errors
///
/// [`CanonicalDirError`] when the path does not resolve or is no directory.
pub fn canonical_dir(path: &Path) -> Result<PathBuf, CanonicalDirError> {
    let canonical = std::fs::canonicalize(path).map_err(CanonicalDirError::Io)?;
    if canonical.is_dir() {
        Ok(canonical)
    } else {
        Err(CanonicalDirError::NotADirectory)
    }
}

/// The file `key` names under `root` (a directory from [`canonical_dir`]), canonical and inside
/// `root`. The first segment of the key is its root's name and is not part of the path.
///
/// This asks the file system (`canonicalize`), so an async caller runs it on a blocking thread.
/// The answer is a path for a read; a symlink swapped in afterwards is not seen, which is why no
/// code of ours creates one under a root and the engine mounts its roots read-only.
///
/// # Errors
///
/// [`KeyError::Invalid`] or [`KeyError::UnknownRoot`] when the key breaks the rule,
/// [`KeyError::NotLocal`] for a root with no local directory, [`KeyError::Unresolvable`] when the
/// file does not exist or cannot be read, [`KeyError::OutsideRoot`] when it resolves outside
/// `root` (a symlink that leaves it).
pub fn resolve_under(root: &Path, key: &str) -> Result<PathBuf, KeyError> {
    if !validate_key(key)?.is_local() {
        return Err(KeyError::NotLocal {
            key: key.to_owned(),
        });
    }
    let joined = key
        .split('/')
        .skip(1)
        .fold(root.to_path_buf(), |p, segment| p.join(segment));
    let resolved = std::fs::canonicalize(&joined).map_err(|source| KeyError::Unresolvable {
        key: key.to_owned(),
        source,
    })?;
    if resolved.starts_with(root) {
        Ok(resolved)
    } else {
        Err(KeyError::OutsideRoot {
            key: key.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::symlink;

    use tempfile::TempDir;

    use super::*;

    struct Fixture {
        _dir: TempDir,
        root: PathBuf,
        outside: PathBuf,
    }

    fn fixture() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let base = fs::canonicalize(dir.path()).unwrap();
        for d in ["root", "outside"] {
            fs::create_dir(base.join(d)).unwrap();
        }
        fs::write(base.join("outside/secret"), b"secret").unwrap();
        Fixture {
            root: canonical_dir(&base.join("root")).unwrap(),
            outside: base.join("outside"),
            _dir: dir,
        }
    }

    #[test]
    fn a_key_resolves_to_its_canonical_file() {
        let f = fixture();
        fs::create_dir(f.root.join("trip")).unwrap();
        fs::write(f.root.join("trip/a.jpg"), b"x").unwrap();
        assert_eq!(
            resolve_under(&f.root, "source/trip/a.jpg").unwrap(),
            f.root.join("trip/a.jpg")
        );
        assert_eq!(
            resolve_under(&f.root, "landing/trip/a.jpg").unwrap(),
            f.root.join("trip/a.jpg")
        );
    }

    #[test]
    fn a_symlink_that_leaves_the_root_is_refused_whether_file_or_folder() {
        let f = fixture();
        symlink(f.outside.join("secret"), f.root.join("link")).unwrap();
        symlink(&f.outside, f.root.join("dir")).unwrap();
        for key in ["source/link", "source/dir/secret"] {
            let err = resolve_under(&f.root, key).unwrap_err();
            assert!(matches!(err, KeyError::OutsideRoot { .. }), "{key}: {err}");
        }
    }

    #[test]
    fn a_symlink_that_stays_inside_the_root_is_followed() {
        let f = fixture();
        fs::write(f.root.join("real.jpg"), b"x").unwrap();
        symlink("real.jpg", f.root.join("alias.jpg")).unwrap();
        assert_eq!(
            resolve_under(&f.root, "source/alias.jpg").unwrap(),
            f.root.join("real.jpg")
        );
    }

    #[test]
    fn a_sibling_that_shares_the_root_name_as_prefix_is_outside() {
        let f = fixture();
        let sibling = f.root.with_file_name("root_evil");
        fs::create_dir(&sibling).unwrap();
        fs::write(sibling.join("x"), b"x").unwrap();
        symlink(&sibling, f.root.join("evil")).unwrap();
        let err = resolve_under(&f.root, "source/evil/x").unwrap_err();
        assert!(matches!(err, KeyError::OutsideRoot { .. }), "{err}");
    }

    #[test]
    fn a_missing_file_a_bad_key_and_a_root_with_no_directory_are_refused() {
        let f = fixture();
        assert!(matches!(
            resolve_under(&f.root, "source/nope.jpg"),
            Err(KeyError::Unresolvable { .. })
        ));
        assert!(matches!(
            resolve_under(&f.root, "source/../outside/secret"),
            Err(KeyError::Invalid { .. })
        ));
        assert!(matches!(
            resolve_under(&f.root, "projects/1/originals/ab/cd"),
            Err(KeyError::NotLocal { .. })
        ));
    }

    #[test]
    fn a_root_must_be_an_existing_directory() {
        let f = fixture();
        let file = f.root.join("a_file");
        fs::write(&file, b"x").unwrap();
        assert!(matches!(
            canonical_dir(&file),
            Err(CanonicalDirError::NotADirectory)
        ));
        assert!(matches!(
            canonical_dir(&f.root.join("absent")),
            Err(CanonicalDirError::Io(_))
        ));
    }
}
