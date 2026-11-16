//! The photo list: one path per line (`#` starts a comment), every file read into memory before
//! the clock starts so the window measures the engine, not the disk. The list is identified by
//! the SHA-256 of the list file's bytes, the same figure `scripts/bench/frame_engine_sets.py`
//! writes beside each list.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use sha2::{Digest, Sha256};

/// The lowercase hexadecimal SHA-256 of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    let digest = Sha256::digest(data);
    let mut out = String::with_capacity(64);
    for byte in digest.as_slice() {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// The paths named by a list file's text, in order.
pub fn parse_paths(text: &str) -> Vec<PathBuf> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(PathBuf::from)
        .collect()
}

/// A list held in memory.
#[derive(Debug)]
pub struct PhotoList {
    /// The list file as given on the command line.
    pub path: PathBuf,
    /// Lowercase hexadecimal SHA-256 of the list file's bytes, so a row names the exact list.
    pub sha256: String,
    /// One entry per photo: the encoded file.
    pub photos: Vec<Bytes>,
    /// The path of each photo, in the same order.
    pub files: Vec<PathBuf>,
    /// Summed size of the photo files, in bytes.
    pub total_bytes: u64,
}

impl PhotoList {
    /// Read the list at `path` and every photo it names into memory.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the list or a photo cannot be read, the list is not UTF-8, or it names no photo.
    pub fn load(path: &Path) -> io::Result<PhotoList> {
        let raw = fs::read(path)?;
        let text = String::from_utf8(raw.clone())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let mut photos = Vec::new();
        let mut files = Vec::new();
        let mut total_bytes = 0u64;
        for file in parse_paths(&text) {
            let data = fs::read(&file)
                .map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", file.display())))?;
            total_bytes += data.len() as u64;
            photos.push(Bytes::from(data));
            files.push(file);
        }
        if photos.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: the list names no photo", path.display()),
            ));
        }
        Ok(PhotoList {
            path: path.to_path_buf(),
            sha256: sha256_hex(&raw),
            photos,
            files,
            total_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_is_the_standard_digest() {
        // FIPS 180-4 test vectors
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn comments_and_blank_lines_are_not_photos() {
        let paths = parse_paths("# set, seed 1\n/a/1.jpg\n\n  /a/2.jpg  \n#/a/3.jpg\n");
        assert_eq!(
            paths,
            [PathBuf::from("/a/1.jpg"), PathBuf::from("/a/2.jpg")]
        );
    }

    #[test]
    fn a_list_is_read_whole_and_hashed_as_a_file() {
        let dir = std::env::temp_dir().join(format!("op-fe-loadgen-list-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.jpg"), b"aaaa").unwrap();
        fs::write(dir.join("b.jpg"), b"bb").unwrap();
        let text = format!(
            "{}\n{}\n",
            dir.join("a.jpg").display(),
            dir.join("b.jpg").display()
        );
        fs::write(dir.join("list.txt"), &text).unwrap();
        let list = PhotoList::load(&dir.join("list.txt")).unwrap();
        assert_eq!(list.photos.len(), 2);
        assert_eq!(list.total_bytes, 6);
        assert_eq!(list.sha256, sha256_hex(text.as_bytes()));
        fs::write(dir.join("empty.txt"), "# nothing\n").unwrap();
        assert!(PhotoList::load(&dir.join("empty.txt")).is_err());
        fs::write(dir.join("missing.txt"), "/nonexistent/x.jpg\n").unwrap();
        assert!(PhotoList::load(&dir.join("missing.txt")).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
