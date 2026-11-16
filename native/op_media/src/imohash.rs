//! The duplicate hash: a port of the Python `imohash` package at the version `requirements.txt`
//! pins (1.1.0), so a photo uploaded twice, or once by Python and once by this crate, gets the
//! same key.
//!
//! The package hashes a whole file under [`SAMPLE_THRESHOLD`] bytes, and above it three samples of
//! [`SAMPLE_SIZE`] bytes (the start, the middle, the end). The 128-bit x64 `MurmurHash3` (seed 0)
//! of those bytes is written as `h1` then `h2`, each big-endian, and the file size as a base-128
//! varint replaces the first bytes (<https://github.com/kalafut/imohash>, "Details"). The
//! `murmur3` crate's `murmur3_x64_128` returns `h2 << 64 | h1` and the same seed handling as
//! `mmh3.hash_bytes` (both halves start at the seed); the goldens in
//! `tests/goldens/imohash.json`, written by `scripts/codegen/imohash_goldens.py` from the Python
//! package, hold the two equal.

use std::fmt;
use std::fs::File;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::PathBuf;

/// Files smaller than this are hashed whole.
pub const SAMPLE_THRESHOLD: u64 = 128 * 1024;
/// The size of each of the three samples of a larger file.
pub const SAMPLE_SIZE: u64 = 16 * 1024;

/// The 16-byte hash. Its lowercase hex form (32 characters) is the key stored today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImoHash([u8; 16]);

impl ImoHash {
    /// The 16 bytes: the size varint, then the tail of the big-endian hash.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// Lowercase hex, as `imohash.hashfileobject(f).hex()` writes it.
    #[must_use]
    pub fn to_hex(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for ImoHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

/// Hash the bytes behind `reader`, reading only the sampled ranges of a file of
/// [`SAMPLE_THRESHOLD`] bytes or more. This is blocking input; use [`hash_file`] from async code.
///
/// # Errors
///
/// Any error of the reader, including `UnexpectedEof` when it ends before a sample is complete
/// (the file shrank while it was hashed).
pub fn hash_reader<R: Read + Seek>(reader: &mut R) -> io::Result<ImoHash> {
    let size = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;
    let mut sampled = Vec::new();
    if size < SAMPLE_THRESHOLD {
        reader.read_to_end(&mut sampled)?;
    } else {
        let sample = SAMPLE_SIZE as usize;
        sampled.resize(3 * sample, 0);
        reader.read_exact(&mut sampled[..sample])?;
        reader.seek(SeekFrom::Start(size / 2))?;
        reader.read_exact(&mut sampled[sample..2 * sample])?;
        reader.seek(SeekFrom::End(-(SAMPLE_SIZE as i64)))?;
        reader.read_exact(&mut sampled[2 * sample..])?;
    }
    let both = murmur3::murmur3_x64_128(&mut Cursor::new(&sampled), 0)?;
    let h1 = (both as u64).to_be_bytes();
    let h2 = ((both >> 64) as u64).to_be_bytes();
    let mut digest = [0_u8; 16];
    digest[..8].copy_from_slice(&h1);
    digest[8..].copy_from_slice(&h2);
    let varint = varint(size);
    digest[..varint.len()].copy_from_slice(&varint);
    Ok(ImoHash(digest))
}

/// Hash the file at `path`. The reads run on tokio's blocking pool, as tokio's documentation asks
/// for blocking file input (<https://docs.rs/tokio/1.54.1/tokio/task/fn.spawn_blocking.html>),
/// so they never occupy an async worker thread.
///
/// # Errors
///
/// An error opening or reading the file, or the blocking task failing to finish.
pub async fn hash_file(path: impl Into<PathBuf>) -> io::Result<ImoHash> {
    let path = path.into();
    tokio::task::spawn_blocking(move || hash_reader(&mut File::open(path)?))
        .await
        .map_err(io::Error::other)?
}

/// `size` as an unsigned base-128 varint, low group first (the Go `binary.PutUvarint` and Python
/// `varint.encode` form).
fn varint(mut size: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(10);
    while size >= 0x80 {
        out.push((size & 0x7f) as u8 | 0x80);
        size >>= 7;
    }
    out.push(size as u8);
    out
}
