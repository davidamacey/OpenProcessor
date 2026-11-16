//! Keys, roots and files on a temporary directory: what a key may name, and each way a read ends.

use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;

use op_local_file::{FileClaim, FileRef};
use tempfile::TempDir;

use super::*;

const LIMIT: u64 = 1 << 20;

struct Disk {
    _dir: TempDir,
    base: PathBuf,
    roots: KeyedRoots,
}

fn disk() -> Disk {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let base = fs::canonicalize(dir.path()).expect("canonical");
    for d in ["landing/7", "source/trip", "outside"] {
        fs::create_dir_all(base.join(d)).expect("directory");
    }
    let roots = KeyedRoots::new(
        Some(&base.join("landing")),
        Some(&base.join("source")),
        LIMIT,
    )
    .expect("roots");
    Disk {
        _dir: dir,
        base,
        roots,
    }
}

fn photo(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + i / 97) as u8).collect()
}

fn named(key: &str, bytes: &[u8]) -> FileRef {
    FileRef::new(key, FileClaim::of(bytes)).expect("a local key")
}

fn status(roots: &KeyedRoots, file: &FileRef) -> &'static str {
    roots.read(file).expect_err("refused").status
}

#[test]
fn a_file_under_each_root_is_read_whole_and_equal_to_what_was_claimed() {
    let d = disk();
    let bytes = photo(300_000);
    fs::write(d.base.join("landing/7/a.0"), &bytes).unwrap();
    fs::write(d.base.join("source/trip/IMG_1.jpg"), &bytes).unwrap();
    for key in ["landing/7/a.0", "source/trip/IMG_1.jpg"] {
        let got = d.roots.read(&named(key, &bytes)).expect("read");
        assert_eq!(got.as_ref(), bytes.as_slice(), "{key}");
    }
}

#[test]
fn an_empty_file_is_read_as_no_bytes() {
    let d = disk();
    fs::write(d.base.join("source/empty.jpg"), b"").unwrap();
    let got = d.roots.read(&named("source/empty.jpg", b"")).expect("read");
    assert!(got.is_empty());
}

#[test]
fn a_missing_file_a_missing_folder_and_a_folder_in_the_way_are_missing() {
    let d = disk();
    let claim = |key: &str| named(key, b"x");
    assert_eq!(status(&d.roots, &claim("source/trip/nope.jpg")), "missing");
    assert_eq!(status(&d.roots, &claim("source/nowhere/a.jpg")), "missing");
    fs::write(d.base.join("source/file"), b"x").unwrap();
    assert_eq!(status(&d.roots, &claim("source/file/under.jpg")), "missing");
}

#[test]
fn another_size_is_changed_before_the_file_is_read_and_the_same_size_with_other_bytes_is_too() {
    let d = disk();
    let bytes = photo(200_000);
    fs::write(d.base.join("source/a.jpg"), &bytes).unwrap();
    let claim = FileClaim::of(&bytes);
    let longer = FileClaim {
        size: claim.size + 1,
        ..claim
    };
    let shorter = FileClaim {
        size: claim.size - 1,
        ..claim
    };
    for wrong in [longer, shorter] {
        let file = FileRef::new("source/a.jpg", wrong).unwrap();
        assert_eq!(status(&d.roots, &file), "changed");
    }
    // the same length, one byte inside a hashed sample differs
    let mut edited = bytes.clone();
    edited[100_000] ^= 1;
    fs::write(d.base.join("source/a.jpg"), &edited).unwrap();
    assert_eq!(status(&d.roots, &named("source/a.jpg", &bytes)), "changed");
}

#[test]
fn a_claim_above_the_limit_is_too_large_without_the_file_being_looked_at() {
    let d = disk();
    let claim = FileClaim {
        size: LIMIT + 1,
        ..FileClaim::of(b"")
    };
    // no such file: opening it would say `missing`
    let file = FileRef::new("source/not_there.jpg", claim).unwrap();
    assert_eq!(status(&d.roots, &file), "too_large");
    let at_limit = FileClaim {
        size: LIMIT,
        ..FileClaim::of(b"")
    };
    let file = FileRef::new("source/not_there.jpg", at_limit).unwrap();
    assert_eq!(status(&d.roots, &file), "missing");
}

#[test]
fn a_file_larger_than_its_claim_is_never_held_whole() {
    let d = disk();
    let bytes = photo(4096);
    let claim = FileClaim::of(&bytes);
    fs::write(d.base.join("source/grew.jpg"), photo(4096 + 5000)).unwrap();
    let file = FileRef::new("source/grew.jpg", claim).unwrap();
    // the open file's size is compared with the claim before a byte is read
    assert_eq!(status(&d.roots, &file), "changed");
}

#[test]
fn a_symlink_that_leaves_the_root_and_a_folder_are_unreadable() {
    let d = disk();
    fs::write(d.base.join("outside/secret.jpg"), photo(100)).unwrap();
    symlink(
        d.base.join("outside/secret.jpg"),
        d.base.join("source/link.jpg"),
    )
    .unwrap();
    symlink(d.base.join("outside"), d.base.join("source/dir")).unwrap();
    let claim = FileClaim::of(&photo(100));
    for key in ["source/link.jpg", "source/dir/secret.jpg", "source/trip"] {
        let file = FileRef::new(key, claim).unwrap();
        assert_eq!(status(&d.roots, &file), "unreadable", "{key}");
    }
}

#[test]
fn a_symlink_that_stays_inside_the_root_is_followed() {
    let d = disk();
    let bytes = photo(500);
    fs::write(d.base.join("source/trip/real.jpg"), &bytes).unwrap();
    symlink("trip/real.jpg", d.base.join("source/alias.jpg")).unwrap();
    let got = d.roots.read(&named("source/alias.jpg", &bytes)).unwrap();
    assert_eq!(got.as_ref(), bytes.as_slice());
}

#[test]
fn a_root_that_is_not_configured_refuses_its_keys_and_says_which_setting() {
    let d = disk();
    let landing_only =
        KeyedRoots::new(Some(&d.base.join("landing")), None, LIMIT).expect("one root");
    let source_key = named("source/a.jpg", b"x");
    let message = landing_only.check(&source_key).expect_err("no source root");
    assert!(
        message
            .to_string()
            .starts_with("root_not_configured: 'source'"),
        "{message}"
    );
    assert!(message.to_string().contains("OP_FRAME_ENGINE_SOURCE_ROOT"));
    assert!(landing_only.check(&named("landing/7/a.0", b"x")).is_ok());
    assert_eq!(status(&landing_only, &source_key), "unreadable");
    let none = KeyedRoots::none();
    assert!(none.check(&named("landing/7/a.0", b"x")).is_err());
    assert!(none.check(&source_key).is_err());
}

#[test]
fn a_root_that_is_no_directory_or_does_not_exist_refuses_the_start() {
    let d = disk();
    fs::write(d.base.join("a_file"), b"x").unwrap();
    let not_dir = KeyedRoots::new(Some(&d.base.join("a_file")), None, LIMIT).unwrap_err();
    assert!(not_dir.contains("landing root") && not_dir.contains("not a directory"));
    let absent = KeyedRoots::new(None, Some(&d.base.join("absent")), LIMIT).unwrap_err();
    assert!(absent.contains("source root") && absent.contains("not usable"));
}

/// Runs the read on another thread; a read that blocks fails the test after a few seconds.
fn read_bounded(roots: KeyedRoots, file: FileRef, what: &str) -> &'static str {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(roots.read(&file).expect_err("refused").status);
    });
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap_or_else(|_| {
            panic!("reading {what} blocked: a key must not open a non-regular file for ever")
        })
}

#[test]
fn a_named_pipe_a_socket_and_a_folder_are_unreadable_without_blocking() {
    use rustix::fs::{CWD, FileType, Mode, mknodat};
    let d = disk();
    mknodat(
        CWD,
        d.base.join("source/pipe.jpg"),
        FileType::Fifo,
        Mode::RUSR | Mode::WUSR,
        0,
    )
    .expect("a named pipe");
    let _socket = std::os::unix::net::UnixListener::bind(d.base.join("source/sock.jpg"))
        .expect("a socket file");
    let claim = FileClaim::of(&photo(100));
    for key in ["source/pipe.jpg", "source/sock.jpg", "source/trip"] {
        let file = FileRef::new(key, claim).unwrap();
        let roots = d.roots.clone();
        assert_eq!(read_bounded(roots, file, key), "unreadable", "{key}");
    }
}
