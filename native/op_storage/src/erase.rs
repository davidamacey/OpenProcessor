//! Delete and erase primitives (plan 18 section 4.5): by key and by prefix, both idempotent.
//!
//! These only delete. Whether an object may be deleted (no catalogue row still names it, under
//! the `(project, imohash)` advisory lock) is the erasure job's decision (R5e). Source roots are
//! read in place and never written or erased, and the landing zone is not in the store.

use futures_util::{StreamExt, TryStreamExt};
use object_store::{ObjectStore, ObjectStoreExt};

use crate::durable::Durable;
use crate::error::StorageError;
use crate::keys::{Root, StorageKey};

fn check_deletable(key: &StorageKey) -> Result<(), StorageError> {
    match key.root() {
        Root::Source => Err(StorageError::Refused {
            key: key.to_string(),
            reason: "a source root is read in place and never erased",
        }),
        Root::Landing => Err(StorageError::Refused {
            key: key.to_string(),
            reason: "the landing zone is not in the object store",
        }),
        _ => Ok(()),
    }
}

/// Delete one object. An object that is already gone is not an error, so erasing twice is a no-op.
///
/// # Errors
///
/// [`StorageError::Refused`] for a source or landing key, [`StorageError::Store`] when the store
/// fails.
pub async fn delete(durable: &Durable, key: &StorageKey) -> Result<(), StorageError> {
    check_deletable(key)?;
    match durable.store().delete(key.as_path()).await {
        Ok(()) | Err(object_store::Error::NotFound { .. }) => Ok(()),
        Err(e) => Err(StorageError::store("delete", key, e)),
    }
}

/// Delete every object under `prefix` (whole path segments: `a/b` covers `a/b/c` and not `a/bc`)
/// through `delete_stream`, which batches where the store allows. Returns how many objects were
/// deleted. A prefix with nothing under it, or one that is deleted twice, is not an error.
///
/// # Errors
///
/// [`StorageError::Refused`] for a source or landing prefix, [`StorageError::Store`] when
/// listing or deleting fails.
pub async fn delete_prefix(durable: &Durable, prefix: &StorageKey) -> Result<u64, StorageError> {
    check_deletable(prefix)?;
    let store = durable.store();
    let locations = store
        .list(Some(prefix.as_path()))
        .map_ok(|meta| meta.location)
        .boxed();
    let mut deleted = 0_u64;
    let mut results = store.delete_stream(locations);
    while let Some(result) = results.next().await {
        match result {
            Ok(_) => deleted += 1,
            Err(object_store::Error::NotFound { .. }) => {}
            Err(e) => return Err(StorageError::store("delete prefix", prefix, e)),
        }
    }
    Ok(deleted)
}
