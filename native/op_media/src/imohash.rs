//! The duplicate hash. The port itself lives in `op_local_file`, which the frame engine uses too;
//! this module adds the one thing only the server needs: hashing a file from async code.

use std::fs::File;
use std::io;
use std::path::PathBuf;

pub use op_local_file::imohash::{ImoHash, SAMPLE_SIZE, SAMPLE_THRESHOLD, hash_reader};

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
