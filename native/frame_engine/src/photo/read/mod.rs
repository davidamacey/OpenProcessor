//! Reading a photo by key (the `read_by_key` extension of protocol version 1): the engine's
//! configured read-only roots, the check of a key against them that needs no disk, and the read
//! itself, which runs on the finish pool (`Device::host`), never on the network runtime.
//!
//! The rule that turns a key into a file is `op_local_file::resolve_under`, the one copy the
//! server's `op_storage` uses too: a key resolves only to a file inside the canonical root, and a
//! symlink that leaves the root is refused. After the read the photo is the same `Bytes` a bytes
//! request carries and follows the same path (route, budget, decode); only the way the bytes
//! arrive differs.
//!
//! Memory: a keyed request holds the file once, in one buffer of exactly its size, which is what
//! a bytes request holds in its message. The claim's size is checked against the message limit
//! before the file is opened, so a photo a bytes request would be refused for is refused here
//! without a read; the in-flight cap bounds the buffers, and the finish pool's threads bound the
//! files open at once.

use std::fs::{File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use op_local_file::{CanonicalDirError, FileRef, KeyError, Root, canonical_dir, resolve_under};
use rustix::fs::OFlags;

use crate::pyfmt::repr_str;
use crate::wire::request::RequestError;

/// Why a file was not read: the photo status that answers it, and what happened, for the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// `missing`, `changed`, `too_large` or `unreadable` (`read_by_key` in the contract).
    pub status: &'static str,
    /// What the operating system or the claim said.
    pub detail: String,
}

fn refuse(status: &'static str, detail: impl Into<String>) -> Refusal {
    Refusal {
        status,
        detail: detail.into(),
    }
}

/// The roots files are read from and the size no file may pass.
#[derive(Debug, Clone, Default)]
pub struct KeyedRoots {
    landing: Option<PathBuf>,
    source: Option<PathBuf>,
    max_bytes: u64,
}

impl KeyedRoots {
    /// Both roots canonical, each an existing directory when given; `max_bytes` is the largest
    /// file read (the message limit a bytes request is held to).
    ///
    /// # Errors
    ///
    /// The text naming the root that is not a usable directory.
    pub fn new(
        landing: Option<&Path>,
        source: Option<&Path>,
        max_bytes: u64,
    ) -> Result<Self, String> {
        let root = |name: &str, path: Option<&Path>| {
            path.map(|p| {
                canonical_dir(p).map_err(|e| match e {
                    CanonicalDirError::Io(e) => {
                        format!("the {name} root {} is not usable: {e}", p.display())
                    }
                    CanonicalDirError::NotADirectory => {
                        format!("the {name} root {} is not a directory", p.display())
                    }
                })
            })
            .transpose()
        };
        Ok(Self {
            landing: root("landing", landing)?,
            source: root("source", source)?,
            max_bytes,
        })
    }

    /// No root: every keyed request is refused.
    #[must_use]
    pub fn none() -> Self {
        Self::default()
    }

    fn dir(&self, root: Root) -> Option<&Path> {
        match root {
            Root::Landing => self.landing.as_deref(),
            Root::Source => self.source.as_deref(),
            _ => None,
        }
    }

    /// Whether this engine reads files under the key's root. Needs no disk.
    ///
    /// # Errors
    ///
    /// `root_not_configured`, naming the root and the setting that would configure it.
    pub fn check(&self, file: &FileRef) -> Result<(), RequestError> {
        if self.dir(file.root()).is_some() {
            return Ok(());
        }
        let setting = match file.root() {
            Root::Landing => "OP_FRAME_ENGINE_LANDING_ROOT",
            _ => "OP_FRAME_ENGINE_SOURCE_ROOT",
        };
        Err(RequestError(format!(
            "root_not_configured: {} (this engine reads no {} files: {setting} is not set)",
            repr_str(file.root().as_str()),
            file.root().as_str(),
        )))
    }

    /// The bytes of the file `file` names, checked against its claim. Blocking: the caller runs
    /// it on the finish pool.
    ///
    /// # Errors
    ///
    /// [`Refusal`]: `too_large` when the claimed size passes the limit (nothing is opened),
    /// `missing` when no file has the key, `changed` when the size or the sampled hash is not the
    /// claim's (a size that differs is seen before the file is read), `unreadable` for a key
    /// that resolves outside its root, a file that is no regular file and any other read error.
    pub fn read(&self, file: &FileRef) -> Result<Bytes, Refusal> {
        let size = file.claim().size;
        if size > self.max_bytes {
            return Err(refuse(
                "too_large",
                format!(
                    "the claim is {size} bytes, the limit is {} bytes",
                    self.max_bytes
                ),
            ));
        }
        let Some(dir) = self.dir(file.root()) else {
            return Err(refuse("unreadable", "no root is configured"));
        };
        let path = resolve_under(dir, file.key()).map_err(resolve_refusal)?;
        read_checked(&path, file)
    }
}

fn open_nonblocking(path: &Path) -> io::Result<File> {
    let flags = i32::try_from(OFlags::NONBLOCK.bits())
        .map_err(|_| io::Error::other("O_NONBLOCK does not fit an i32"))?;
    OpenOptions::new().read(true).custom_flags(flags).open(path)
}

fn resolve_refusal(e: KeyError) -> Refusal {
    match &e {
        KeyError::Unresolvable { source, .. }
            if matches!(
                source.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            refuse("missing", e.to_string())
        }
        _ => refuse("unreadable", e.to_string()),
    }
}

fn io_refusal(e: &io::Error) -> Refusal {
    match e.kind() {
        io::ErrorKind::NotFound => refuse("missing", e.to_string()),
        _ => refuse("unreadable", e.to_string()),
    }
}

/// Opens `path` with `O_NONBLOCK`, so that opening a named pipe returns at once instead of
/// waiting for a writer (fifo(7), open(2)); the flag has no effect on a regular file. The type is
/// then checked on the open descriptor, which leaves no gap between the check and the open.
/// Compares the size of the open file with the claim before reading it, reads it
/// into one buffer (a byte more than claimed, to see a file that grew) and checks the claim.
fn read_checked(path: &Path, file: &FileRef) -> Result<Bytes, Refusal> {
    let claim = file.claim();
    let open = open_nonblocking(path).map_err(|e| io_refusal(&e))?;
    let meta = open.metadata().map_err(|e| io_refusal(&e))?;
    if !meta.is_file() {
        return Err(refuse("unreadable", "the key does not name a regular file"));
    }
    if meta.len() != claim.size {
        return Err(refuse(
            "changed",
            format!("the file has {} bytes", meta.len()),
        ));
    }
    let wanted = usize::try_from(claim.size).map_err(|_| refuse("too_large", "past usize"))?;
    let mut buffer = Vec::with_capacity(wanted.saturating_add(1));
    open.take(claim.size.saturating_add(1))
        .read_to_end(&mut buffer)
        .map_err(|e| io_refusal(&e))?;
    claim
        .check(&buffer)
        .map_err(|e| refuse("changed", e.to_string()))?;
    Ok(Bytes::from(buffer))
}

#[cfg(test)]
mod tests;
