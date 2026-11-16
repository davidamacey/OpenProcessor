use std::fs;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use bytes::Bytes;
use object_store::memory::InMemory;
use op_config::Env;
use op_storage::cache::DiskCache;
use op_storage::derived::{CACHE_MAX_BYTES_KEY, CACHE_ROOT_KEY, Derived};
use op_storage::durable::{Backend, Durable};
use op_storage::error::StorageError;
use op_storage::keys::{DerivedKind, ProjectId, StorageKey};

use crate::support::{World, world};

fn derived(w: &World, cap: Option<u64>) -> Derived {
    let durable = Durable::from_store(Arc::new(InMemory::new()), Backend::Streamed, true);
    let cache = cap.map(|c| DiskCache::new(&w.base.join("cache"), c).unwrap());
    Derived::new(durable, cache)
}

fn thumb(photo: i64, name: &str) -> StorageKey {
    StorageKey::derived(ProjectId(1), DerivedKind::Thumbs, photo, name).unwrap()
}

fn cache_file(w: &World, key: &StorageKey) -> std::path::PathBuf {
    w.base.join("cache").join(key.as_str())
}

#[tokio::test]
async fn a_derived_file_is_written_read_replaced_and_deleted_by_prefix() {
    let w = world();
    let d = derived(&w, None);
    let key = thumb(42, "256.jpg");
    d.put(&key, Bytes::from_static(b"v1")).await.unwrap();
    assert_eq!(d.get(&key).await.unwrap(), "v1");
    d.put(&key, Bytes::from_static(b"v2 longer")).await.unwrap();
    assert_eq!(d.get(&key).await.unwrap(), "v2 longer");

    let other = thumb(4, "256.jpg");
    d.put(&other, Bytes::from_static(b"keep")).await.unwrap();
    let prefix = StorageKey::derived_prefix(ProjectId(1), DerivedKind::Thumbs, 42);
    assert_eq!(d.delete_prefix(&prefix).await.unwrap(), 1);
    assert!(matches!(
        d.get(&key).await,
        Err(StorageError::NotFound { .. })
    ));
    assert_eq!(d.get(&other).await.unwrap(), "keep");
    assert_eq!(d.delete_prefix(&prefix).await.unwrap(), 0);
}

#[tokio::test]
async fn only_derived_keys_are_accepted() {
    let w = world();
    let d = derived(&w, None);
    let original = StorageKey::parse("projects/1/originals/ab/cd").unwrap();
    let unknown_kind = StorageKey::parse("projects/1/derived/other/1/x").unwrap();
    for key in [&original, &unknown_kind] {
        let err = d.put(key, Bytes::new()).await.unwrap_err();
        assert!(matches!(err, StorageError::Refused { .. }), "{err}");
        assert!(d.get(key).await.is_err());
    }
    let whole_project = StorageKey::parse("projects/1/originals").unwrap();
    assert!(matches!(
        d.delete_prefix(&whole_project).await,
        Err(StorageError::Refused { .. })
    ));
    let all_derived = StorageKey::parse("projects/1/derived").unwrap();
    assert_eq!(d.delete_prefix(&all_derived).await.unwrap(), 0);
}

#[tokio::test]
async fn a_cached_file_is_served_from_disk_while_the_store_says_it_is_current() {
    let w = world();
    let d = derived(&w, Some(1 << 20));
    let key = thumb(1, "a.jpg");
    d.put(&key, Bytes::from_static(b"the real bytes"))
        .await
        .unwrap();
    assert_eq!(d.get(&key).await.unwrap(), "the real bytes");

    // Change the cached body in place: it is what a hit serves.
    let path = cache_file(&w, &key);
    let mut raw = fs::read(&path).unwrap();
    let n = raw.len();
    raw[n - 5..].copy_from_slice(b"CACHE");
    fs::write(&path, raw).unwrap();
    assert_eq!(d.get(&key).await.unwrap(), "the real CACHE");
}

#[tokio::test]
async fn a_replaced_file_is_never_served_from_the_cache() {
    let w = world();
    let d = derived(&w, Some(1 << 20));
    let key = thumb(1, "a.jpg");
    d.put(&key, Bytes::from_static(b"old")).await.unwrap();
    assert_eq!(d.get(&key).await.unwrap(), "old");
    d.put(&key, Bytes::from_static(b"new")).await.unwrap();
    assert_eq!(d.get(&key).await.unwrap(), "new");
}

#[tokio::test]
async fn a_file_erased_behind_the_cache_is_not_served_and_its_copy_is_dropped() {
    let w = world();
    let durable = Durable::from_store(Arc::new(InMemory::new()), Backend::Streamed, true);
    let cache = DiskCache::new(&w.base.join("cache"), 1 << 20).unwrap();
    let writer = Derived::new(durable.clone(), None);
    let reader = Derived::new(durable, Some(cache));
    let key = thumb(1, "a.jpg");
    writer
        .put(&key, Bytes::from_static(b"secret"))
        .await
        .unwrap();
    assert_eq!(reader.get(&key).await.unwrap(), "secret");
    assert!(cache_file(&w, &key).exists());

    let prefix = StorageKey::derived_prefix(ProjectId(1), DerivedKind::Thumbs, 1);
    writer.delete_prefix(&prefix).await.unwrap();

    assert!(matches!(
        reader.get(&key).await,
        Err(StorageError::NotFound { .. })
    ));
    assert!(!cache_file(&w, &key).exists());
}

#[tokio::test]
async fn deleting_by_prefix_empties_this_processs_cache_for_it() {
    let w = world();
    let d = derived(&w, Some(1 << 20));
    let key = thumb(1, "a.jpg");
    d.put(&key, Bytes::from_static(b"x")).await.unwrap();
    d.get(&key).await.unwrap();
    assert!(cache_file(&w, &key).exists());
    d.delete_prefix(&StorageKey::derived_prefix(
        ProjectId(1),
        DerivedKind::Thumbs,
        1,
    ))
    .await
    .unwrap();
    assert!(!cache_file(&w, &key).exists());
}

#[tokio::test]
async fn the_cache_evicts_the_files_read_longest_ago_at_its_cap() {
    let w = world();
    let cache = DiskCache::new(&w.base.join("cache"), 3_600).unwrap();
    let d = Derived::new(
        Durable::from_store(Arc::new(InMemory::new()), Backend::Streamed, true),
        Some(cache),
    );
    let body = Bytes::from(vec![b'x'; 1_000]);
    let keys: Vec<StorageKey> = (1..=3).map(|i| thumb(i, "a.jpg")).collect();
    for key in &keys {
        d.put(key, body.clone()).await.unwrap();
        d.get(key).await.unwrap();
    }
    // Make the first file the oldest read and the second the next, then read the first again.
    let now = SystemTime::now();
    for (i, key) in keys.iter().enumerate() {
        let f = fs::File::options()
            .write(true)
            .open(cache_file(&w, key))
            .unwrap();
        f.set_modified(now - Duration::from_secs(100 - 10 * i as u64))
            .unwrap();
    }
    d.get(&keys[0]).await.unwrap();

    let fourth = thumb(4, "a.jpg");
    d.put(&fourth, body.clone()).await.unwrap();
    d.get(&fourth).await.unwrap();

    assert!(
        !cache_file(&w, &keys[1]).exists(),
        "the file read longest ago goes"
    );
    for kept in [&keys[0], &keys[2], &fourth] {
        assert!(cache_file(&w, kept).exists(), "{kept}");
    }
}

fn s3_durable(w: &World) -> Durable {
    let env = Env::from_pairs([("OP_STORAGE_URL", "s3://photos")]);
    Durable::from_settings(&env, &w.roots).unwrap()
}

#[test]
fn an_s3_store_with_a_cache_needs_its_own_cache_root_and_creates_nothing_in_landing() {
    let w = world();
    let err = Derived::from_settings(&Env::from_pairs([("X", "1")]), s3_durable(&w))
        .unwrap_err()
        .to_string();
    assert!(err.contains(CACHE_ROOT_KEY), "{err}");

    let root = w.base.join("derived_cache");
    let env = Env::from_pairs([(CACHE_ROOT_KEY, root.to_str().unwrap())]);
    Derived::from_settings(&env, s3_durable(&w)).unwrap();
    assert!(root.is_dir());
    assert_eq!(fs::read_dir(w.base.join("landing")).unwrap().count(), 0);
}

#[test]
fn the_cache_root_is_ignored_for_a_local_store_or_a_zero_cap() {
    let w = world();
    let local = Durable::from_store(Arc::new(InMemory::new()), Backend::Streamed, false);
    Derived::from_settings(&Env::from_pairs([("X", "1")]), local).unwrap();
    let off = Env::from_pairs([(CACHE_MAX_BYTES_KEY, "0")]);
    Derived::from_settings(&off, s3_durable(&w)).unwrap();
    assert_eq!(fs::read_dir(w.base.join("landing")).unwrap().count(), 0);
}
