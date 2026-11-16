//! A byte-capped local cache of derived files in front of a remote store.
//!
//! One file per key, laid out like the key under the cache root (keys are validated, so a key
//! never leaves it). The file holds a 2-byte etag length, the etag, then the body. The cache never
//! serves a body without the store confirming its etag still current (a conditional get, see
//! [`crate::derived`]), so a deleted or replaced file is never served from here.
//!
//! The directory is the truth, not memory: all server processes of a host share it, and no
//! in-process index is kept. A file is written under a temporary name and renamed in, so a reader
//! sees a whole file or none; a reader that loses a race with an eviction or an erase finds no
//! file and goes to the store. Two times of a file carry two rules. Its modification time is when
//! it was written: a file older than the maximum age is removed by the next scan, and a read of
//! one confirms it with the store and writes it again. Its access time is when it was last read
//! (set on every hit, so mount options do not matter): when the total passes the cap the files
//! read longest ago are deleted. A scan costs one directory walk, so it runs when a fraction of the
//! cap has been added since the last one or a quarter of the maximum age has passed (a pacing
//! counter, not state anything relies on), and each process may overshoot the cap by that
//! fraction.
//!
//! After an erase, no process serves the file; this host's cached copy is removed at once (the
//! erasing process removes the subtree on disk); another host's copy is removed within the maximum
//! age (`OP_STORAGE_CACHE_MAX_AGE_S`, by the scan). The server runs [`DiskCache::evict`] on a timer
//! so an idle host's copies go too.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    max_age: Duration,
    /// Names this instance's temporary files apart from every other process's.
    writer: String,
    added_since_scan: AtomicU64,
    last_scan_s: AtomicU64,
}

/// A cached body with the etag it was stored under.
pub(crate) struct Entry {
    pub(crate) etag: String,
    pub(crate) body: Bytes,
    /// Written longer ago than the maximum age: it is confirmed and written again.
    pub(crate) expired: bool,
}

impl DiskCache {
    /// A cache in `root` (made if it does not exist) holding at most about `max_bytes`, keeping a
    /// file at most `max_age` after it was written.
    ///
    /// # Errors
    ///
    /// [`StorageError::Io`] when the directory cannot be made.
    pub fn new(root: &Path, max_bytes: u64, max_age: Duration) -> Result<Self, StorageError> {
        std::fs::create_dir_all(root).map_err(|e| StorageError::io("create folder", root, e))?;
        Ok(Self {
            root: root.to_path_buf(),
            max_bytes,
            max_age,
            writer: format!("{}.{}", std::process::id(), now_nanos()),
            added_since_scan: AtomicU64::new(0),
            last_scan_s: AtomicU64::new(now_secs()),
        })
    }

    fn file(&self, key: &StorageKey) -> PathBuf {
        key.segments()
            .fold(self.root.join(key.root().as_str()), |p, s| p.join(s))
    }

    pub(crate) async fn read(&self, key: &StorageKey) -> Option<Entry> {
        let path = self.file(key);
        let modified = tokio::fs::metadata(&path).await.ok()?.modified().ok()?;
        let raw = tokio::fs::read(&path).await.ok()?;
        let len = usize::from(u16::from_be_bytes([*raw.first()?, *raw.get(1)?]));
        let etag = String::from_utf8(raw.get(2..2 + len)?.to_vec()).ok()?;
        let body = Bytes::from(raw).slice(2 + len..);
        let expired = SystemTime::now()
            .duration_since(modified)
            .is_ok_and(|age| age > self.max_age);
        Some(Entry {
            etag,
            body,
            expired,
        })
    }

    /// Mark a file as read just now (its access time), so it is the last to be evicted.
    pub(crate) async fn touch(&self, key: &StorageKey) {
        let path = self.file(key);
        let _ = tokio::task::spawn_blocking(move || {
            std::fs::File::options()
                .write(true)
                .open(path)
                .and_then(|f| {
                    f.set_times(std::fs::FileTimes::new().set_accessed(SystemTime::now()))
                })
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
        if write_atomically(&path, &self.writer, &data).await.is_err() {
            return;
        }
        let before = self.added_since_scan.fetch_add(size, Ordering::Relaxed);
        let idle_s = now_secs().saturating_sub(self.last_scan_s.load(Ordering::Relaxed));
        if before + size >= (self.max_bytes / SCAN_EVERY).max(1)
            || idle_s >= (self.max_age.as_secs() / 4).max(1)
        {
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

    /// Delete the files written longer ago than the maximum age, then the files read longest ago
    /// until the total is within the cap.
    pub async fn evict(&self) {
        self.added_since_scan.store(0, Ordering::Relaxed);
        self.last_scan_s.store(now_secs(), Ordering::Relaxed);
        let root = self.root.clone();
        let (max, max_age) = (self.max_bytes, self.max_age);
        let _ = tokio::task::spawn_blocking(move || evict_files(&root, max, max_age)).await;
    }
}

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

async fn write_atomically(path: &Path, writer: &str, data: &[u8]) -> std::io::Result<()> {
    let dir = path
        .parent()
        .ok_or_else(|| std::io::Error::other("cache file has no folder"))?;
    tokio::fs::create_dir_all(dir).await?;
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let tmp = dir.join(format!(
        ".{writer}.{}.tmp",
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    tokio::fs::write(&tmp, data).await?;
    if let Err(e) = tokio::fs::rename(&tmp, path).await {
        let _ = tokio::fs::remove_file(&tmp).await;
        return Err(e);
    }
    Ok(())
}

fn evict_files(root: &Path, max_bytes: u64, max_age: Duration) {
    let now = SystemTime::now();
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
                continue;
            }
            let Ok(written) = meta.modified() else {
                continue;
            };
            if now.duration_since(written).is_ok_and(|age| age > max_age) {
                let _ = std::fs::remove_file(entry.path());
            } else {
                let read = meta.accessed().unwrap_or(written);
                files.push((read, meta.len(), entry.path()));
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
