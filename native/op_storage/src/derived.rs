//! Derived files: crops, thumbnails, exports, training copies and snapshots (plan 18 section 4.3).
//!
//! They live under `projects/<project>/derived/<kind>/<photo_key>/...`, apart from the originals,
//! and are deleted by prefix. Crops and thumbnails are a cache that can be made again from the
//! original, so `put` overwrites.
//!
//! In front of a network store a [`DiskCache`] keeps recently read files on local disk. A cached
//! file is served only after the store confirms its etag is still the current one: `get` sends
//! `If-None-Match` (<https://docs.rs/object_store/0.14.2/object_store/struct.GetOptions.html>) and
//! the store answers `NotModified` without a body. So a file that was replaced or erased is never
//! served from the cache, and an erase is not undone by a cache.

use std::path::Path;
use std::time::Duration;

use bytes::Bytes;
use object_store::{GetOptions, ObjectStore, PutMode, PutPayload};
use op_config::Env;

use crate::cache::DiskCache;
use crate::durable::Durable;
use crate::erase;
use crate::error::StorageError;
use crate::keys::{Root, StorageKey};

/// The setting for the byte cap of the derived-file cache (network stores only; 0 turns it off).
pub const CACHE_MAX_BYTES_KEY: &str = "OP_STORAGE_CACHE_MAX_BYTES";
const DEFAULT_CACHE_MAX_BYTES: u64 = 1024 * 1024 * 1024;
/// The setting for how long after it was written a cached file is kept at most, in seconds.
pub const CACHE_MAX_AGE_KEY: &str = "OP_STORAGE_CACHE_MAX_AGE_S";
const DEFAULT_CACHE_MAX_AGE_S: u64 = 86_400;
/// The setting for the cache directory (required with an S3 store and a cap above 0, else ignored).
pub const CACHE_ROOT_KEY: &str = "OP_STORAGE_CACHE_ROOT";

/// Segments of an object key under the root: project, `derived`, kind, photo, name.
const OBJECT_SEGMENTS: usize = 5;
/// Fewest segments of a prefix that may be deleted: project and `derived`.
const PREFIX_SEGMENTS: usize = 2;
const KINDS: [&str; 5] = ["crops", "thumbs", "exports", "training", "snapshots"];

/// Derived files of one durable store.
#[derive(Debug)]
pub struct Derived {
    durable: Durable,
    cache: Option<DiskCache>,
}

impl Derived {
    /// Derived files in `durable` with an optional cache.
    #[must_use]
    pub fn new(durable: Durable, cache: Option<DiskCache>) -> Self {
        Self { durable, cache }
    }

    /// Derived files with the cache the settings ask for: `OP_STORAGE_CACHE_MAX_BYTES` (default
    /// 1 GiB, 0 = off) in the directory `OP_STORAGE_CACHE_ROOT`, and only for a network store (a
    /// local store is already local disk), each file kept at most `OP_STORAGE_CACHE_MAX_AGE_S`
    /// (default 86400, above 0) after it was written. The directory is required then and is never inside the
    /// landing root; with a local store or a cap of 0 it is ignored.
    ///
    /// # Errors
    ///
    /// [`StorageError::Setting`] for an unparsable cap or age, [`StorageError::Config`] naming
    /// `OP_STORAGE_CACHE_ROOT` when it is needed and not set (or the age is 0), [`StorageError::Io`] when the cache
    /// directory cannot be made.
    pub fn from_settings(env: &Env, durable: Durable) -> Result<Self, StorageError> {
        let max: u64 = env
            .get(CACHE_MAX_BYTES_KEY, DEFAULT_CACHE_MAX_BYTES)
            .map_err(StorageError::Setting)?;
        let cache = if durable.is_remote() && max > 0 {
            let root: String = env
                .get(CACHE_ROOT_KEY, String::new())
                .map_err(StorageError::Setting)?;
            if root.is_empty() {
                return Err(StorageError::Config {
                    key: CACHE_ROOT_KEY,
                    reason: "required with an S3 store and OP_STORAGE_CACHE_MAX_BYTES above 0"
                        .to_owned(),
                });
            }
            let max_age_s: u64 = env
                .get(CACHE_MAX_AGE_KEY, DEFAULT_CACHE_MAX_AGE_S)
                .map_err(StorageError::Setting)?;
            if max_age_s == 0 {
                return Err(StorageError::Config {
                    key: CACHE_MAX_AGE_KEY,
                    reason: "must be above 0 seconds".to_owned(),
                });
            }
            Some(DiskCache::new(
                Path::new(&root),
                max,
                Duration::from_secs(max_age_s),
            )?)
        } else {
            None
        };
        Ok(Self::new(durable, cache))
    }

    /// Write a derived file, replacing any earlier one.
    ///
    /// # Errors
    ///
    /// [`StorageError::Refused`] when `key` is not a derived file's key, [`StorageError::Store`]
    /// when the store fails.
    pub async fn put(&self, key: &StorageKey, bytes: Bytes) -> Result<(), StorageError> {
        check_derived(key, OBJECT_SEGMENTS)?;
        self.durable
            .store()
            .put_opts(
                key.as_path(),
                PutPayload::from(bytes),
                PutMode::Overwrite.into(),
            )
            .await
            .map_err(|e| StorageError::store("put", key, e))?;
        if let Some(cache) = &self.cache {
            cache.remove(key).await;
        }
        Ok(())
    }

    /// Read a derived file.
    ///
    /// # Errors
    ///
    /// [`StorageError::NotFound`] when there is none (a cached copy is dropped),
    /// [`StorageError::Refused`] when `key` is not a derived file's key,
    /// [`StorageError::Store`] when the store fails.
    pub async fn get(&self, key: &StorageKey) -> Result<Bytes, StorageError> {
        check_derived(key, OBJECT_SEGMENTS)?;
        let store = self.durable.store();
        let Some(cache) = &self.cache else {
            return fetch(store.as_ref(), key, GetOptions::default())
                .await
                .map(|f| f.0);
        };
        let cached = cache.read(key).await;
        let options = GetOptions::new().with_if_none_match(cached.as_ref().map(|c| c.etag.clone()));
        match fetch(store.as_ref(), key, options).await {
            Ok((body, etag)) => {
                if let Some(etag) = etag {
                    cache.insert(key, &etag, &body).await;
                }
                Ok(body)
            }
            Err(StorageError::Store {
                source: object_store::Error::NotModified { .. },
                ..
            }) => match cached {
                Some(entry) if entry.expired => {
                    cache.insert(key, &entry.etag, &entry.body).await;
                    Ok(entry.body)
                }
                Some(entry) => {
                    cache.touch(key).await;
                    Ok(entry.body)
                }
                None => Err(StorageError::NotFound {
                    key: key.to_string(),
                }),
            },
            Err(e) => {
                if matches!(e, StorageError::NotFound { .. }) {
                    cache.remove(key).await;
                }
                Err(e)
            }
        }
    }

    /// Delete every derived file under `prefix`, for example
    /// [`StorageKey::derived_prefix`] (one kind of one photo) or `projects/<id>/derived` (all of a
    /// project's). Idempotent. Returns how many objects were deleted. After erase, no process serves
    /// the file; this host's cached copy is removed at once; another host's copy is removed within
    /// `OP_STORAGE_CACHE_MAX_AGE_S`. (No process serves it because `get` confirms the etag with the
    /// store; the cache directory is shared by this host's processes and laid out like the keys, so
    /// removing the subtree empties it for all of them.)
    ///
    /// # Errors
    ///
    /// [`StorageError::Refused`] when `prefix` is not under a project's `derived` folder,
    /// [`StorageError::Store`] when the store fails.
    pub async fn delete_prefix(&self, prefix: &StorageKey) -> Result<u64, StorageError> {
        check_derived(prefix, PREFIX_SEGMENTS)?;
        let deleted = erase::delete_prefix(&self.durable, prefix).await?;
        if let Some(cache) = &self.cache {
            cache.remove_prefix(prefix).await;
        }
        Ok(deleted)
    }
}

async fn fetch(
    store: &dyn ObjectStore,
    key: &StorageKey,
    options: GetOptions,
) -> Result<(Bytes, Option<String>), StorageError> {
    let result = match store.get_opts(key.as_path(), options).await {
        Ok(r) => r,
        Err(object_store::Error::NotFound { .. }) => {
            return Err(StorageError::NotFound {
                key: key.to_string(),
            });
        }
        Err(e) => return Err(StorageError::store("get", key, e)),
    };
    let etag = result.meta.e_tag.clone();
    let body = result
        .bytes()
        .await
        .map_err(|e| StorageError::store("get", key, e))?;
    Ok((body, etag))
}

/// The key is `projects/<project>/derived/<kind>/...` with at least `min_segments` segments after
/// the root.
fn check_derived(key: &StorageKey, min_segments: usize) -> Result<(), StorageError> {
    let segs: Vec<&str> = key.segments().collect();
    let shaped = key.root() == Root::Projects
        && segs.len() >= min_segments
        && segs.get(1) == Some(&"derived")
        && segs.get(2).is_none_or(|kind| KINDS.contains(kind));
    if shaped {
        Ok(())
    } else {
        Err(StorageError::Refused {
            key: key.to_string(),
            reason: "not a derived file's key (projects/<id>/derived/<kind>/<photo>/<name>)",
        })
    }
}
