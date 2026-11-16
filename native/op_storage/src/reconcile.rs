//! The listing the reconcile tick needs to find objects no row references (plan 18 section 4.2
//! step 6): keys and ages of one project's originals, streamed. R3a ports the `upload_sweep` tick
//! and R1c adds this part; deciding what is unreferenced is theirs, this only lists.

use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use object_store::ObjectStore;

use crate::durable::Durable;
use crate::error::StorageError;
use crate::keys::{ProjectId, StorageKey};

/// One original in the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredOriginal {
    /// Its content key.
    pub key: StorageKey,
    /// Its size in bytes.
    pub size: u64,
    /// How long ago the store says it was last written (zero if its clock is ahead of ours).
    pub age: Duration,
}

/// Stream every original of `project`, in no particular order, without holding the listing in
/// memory.
pub fn list_originals(
    durable: &Durable,
    project: ProjectId,
) -> BoxStream<'static, Result<StoredOriginal, StorageError>> {
    let prefix = StorageKey::originals_prefix(project);
    let shown = prefix.clone();
    durable
        .store()
        .list(Some(prefix.as_path()))
        .map(move |meta| {
            let meta = meta.map_err(|e| StorageError::store("list", &shown, e))?;
            let key = StorageKey::parse(meta.location.as_ref()).map_err(StorageError::Key)?;
            let written = SystemTime::from(meta.last_modified);
            Ok(StoredOriginal {
                key,
                size: meta.size,
                age: SystemTime::now()
                    .duration_since(written)
                    .unwrap_or_default(),
            })
        })
        .boxed()
}
