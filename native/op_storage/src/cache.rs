//! A byte-capped local cache of derived files in front of a remote store.
//!
//! One file per key, laid out like the key under the cache root (keys are validated, so a key
//! never leaves it). The file holds a 2-byte etag length, the etag, then the body. The cache never
//! serves a body without the store confirming its etag still current (a conditional get, see
//! [`crate::derived`]), so a deleted or replaced file is never served from here.
//!
//! The directory is the truth, not memory: several processes may share it. Recency is the file's
//! modification time, set when a file is read; when the total passes the cap the files read
//! longest ago are deleted. A scan costs one directory walk, so it runs when a fraction of the cap
//! has been added since the last one (a pacing counter, not state anything relies on), and each
//! process may overshoot the cap by that fraction.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use bytes::Bytes;

use crate::error::StorageError;
use crate::keys::StorageKey;

/// A scan runs after this fraction of the cap (one in `SCAN_EVERY`) has been added.
const SCAN_EVERY: u64 = 16;

/// The cache directory and its cap.
#[derive(Debug)]
pub struct DiskCache {
    root: PathBuf,
    max_bytes: u64,
    added_since_scan: AtomicU64,
}

/// A cached body with the etag it was stored under.
pub(crate) struct Entry {
    pub(crate) etag: String,
    pub(crate) body: Bytes,
}

impl DiskCache {
    /// A cache in `root` (made if it does not exist) holding at most about `max_bytes`.
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] when the directory cannot be made.
    pub fn new(root: &Path, max_bytes: u64) -> Result<Self, StorageError> {
        std::fs::create_dir_all(root).map_err(|e| StorageError::io("create folder", root, e))?;
        Ok(Self {
            root: root.to_path_buf(),
            max_bytes,
            added_since_scan: AtomicU64::new(0),
        })
    }

    fn file(&self, key: &StorageKey) -> PathBuf {
        key.segments()
            .fold(self.root.join(key.root().as_str()), |p, s| p.join(s))
    }

    pub(crate) async fn read(&self, key: &StorageKey) -> Option<Entry> {
        let raw = tokio::fs::read(self.file(key)).await.ok()?;
        let len = usize::from(u16::from_be_bytes([*raw.first()?, *raw.get(1)?]));
        let etag = String::from_utf8(raw.get(2..2 + len)?.to_vec()).ok()?;
        let body = Bytes::from(raw).slice(2 + len..);
        Some(Entry { etag, body })
    }

    /// Mark a file as read just now, so it is the last to be evicted.
    pub(crate) async fn touch(&self, key: &StorageKey) {
        let path = self.file(key);
        let _ = tokio::task::spawn_blocking(move || {
            std::fs::File::options()
                .write(true)
                .open(path)
                .and_then(|f| f.set_modified(SystemTime::now()))
        })
        .await;
    }

    /// Store a body. A cache that cannot be written is not an error: the caller has the body.
    pub(crate) async fn insert(&self, key: &StorageKey, etag: &str, body: &Bytes) {
        let Ok(etag_len) = u16::try_from(etag.len()) else {
            return;
        };
        let size = 2 + u64::from(etag_len) + body.len() as u64;
        if size > self.max_bytes {
            return;
        }
        let mut data = Vec::with_capacity(usize::try_from(size).unwrap_or(0));
        data.extend_from_slice(&etag_len.to_be_bytes());
        data.extend_from_slice(etag.as_bytes());
        data.extend_from_slice(body);
        let path = self.file(key);
        if write_atomically(&path, &data).await.is_err() {
            return;
        }
        let before = self.added_since_scan.fetch_add(size, Ordering::Relaxed);
        if before + size >= (self.max_bytes / SCAN_EVERY).max(1) {
            self.added_since_scan.store(0, Ordering::Relaxed);
            self.evict().await;
        }
    }

    pub(crate) async fn remove(&self, key: &StorageKey) {
        let _ = tokio::fs::remove_file(self.file(key)).await;
    }

    /// Remove everything cached under `prefix`.
    pub(crate) async fn remove_prefix(&self, prefix: &StorageKey) {
        let _ = tokio::fs::remove_dir_all(self.file(prefix)).await;
    }

    /// Delete the files read longest ago until the total is within the cap.
    pub(crate) async fn evict(&self) {
        let root = self.root.clone();
        let max = self.max_bytes;
        let _ = tokio::task::spawn_blocking(move || evict_oldest(&root, max)).await;
    }
}

async fn write_atomically(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("cache file has no folder"))?;
    tokio::fs::create_dir_all(dir).await?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    tokio::fs::write(&tmp, data).await?;
    if let Err(e) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    Ok(())
}

fn evict_oldest(root: &Path, max_bytes: u64) {
    let mut files = Vec::new();
    let mut folders = vec![root.to_path_buf()];
    while let Some(folder) = folders.pop() {
        let Ok(entries) = std::fs::read_dir(&folder) else {
            continue;
        };
        for entry in entries.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if meta.is_dir() {
                folders.push(entry.path());
            } else if let Ok(modified) = meta.modified() {
                files.push((modified, meta.len(), entry.path()));
            }
        }
    }
    let mut total: u64 = files.iter().map(|f| f.1).sum();
    files.sort();
    for (_, len, path) in files {
        if total <= max_bytes {
            break;
        }
        if std::fs::remove_file(path).is_ok() {
            total -= len;
        }
    }
}
