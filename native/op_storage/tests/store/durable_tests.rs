use std::os::unix::fs::MetadataExt;
use std::sync::Arc;

use object_store::memory::InMemory;
use object_store::{ObjectStore, ObjectStoreExt};
use op_config::Env;
use op_storage::durable::{Backend, Durable, STORAGE_URL_KEY};
use op_storage::error::StorageError;
use op_storage::keys::{ProjectId, StorageKey};
use op_storage::local::{LANDING_ROOT_KEY, SOURCE_ROOT_KEY};

use crate::support::{World, land, pattern, sha, world};

fn memory() -> (Arc<InMemory>, Durable) {
    let store = Arc::new(InMemory::new());
    let durable = Durable::from_store(store.clone(), Backend::Streamed, true);
    (store, durable)
}

async fn stored_bytes(store: &dyn ObjectStore, key: &StorageKey) -> Vec<u8> {
    store
        .get(key.as_path())
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec()
}

fn local_store(w: &World, landing_in_store: bool) -> Durable {
    let store_dir = if landing_in_store {
        w.base.clone()
    } else {
        let d = w.base.join("store");
        std::fs::create_dir_all(&d).unwrap();
        d
    };
    let env = Env::from_pairs([
        (STORAGE_URL_KEY, format!("file://{}", store_dir.display())),
        (LANDING_ROOT_KEY, w.roots.landing.display().to_string()),
        (SOURCE_ROOT_KEY, w.roots.sources.display().to_string()),
    ]);
    Durable::from_settings(&env, &w.roots).unwrap()
}

#[tokio::test]
async fn a_photo_is_stored_under_its_content_key_with_the_right_size() {
    let w = world();
    let (store, durable) = memory();
    let data = pattern(100_000, 1);
    let photo = land(&w, 7, 1, &data).await;

    let object = durable.make_durable(&photo).await.unwrap();

    let key = StorageKey::original(ProjectId(7), &sha(&data));
    assert_eq!(object.key, key);
    assert_eq!((object.size, object.created), (data.len() as u64, true));
    assert_eq!(stored_bytes(store.as_ref(), &key).await, data);
    assert!(durable.exists(&key).await.unwrap());
    assert!(
        !durable
            .exists(&StorageKey::original(ProjectId(8), &sha(&data)))
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn equal_bytes_a_second_time_write_nothing_and_leave_both_landing_files() {
    let w = world();
    let (_, durable) = memory();
    let data = pattern(40_000, 2);
    let first = land(&w, 1, 1, &data).await;
    let second = land(&w, 1, 2, &data).await;

    assert!(durable.make_durable(&first).await.unwrap().created);
    let again = durable.make_durable(&second).await.unwrap();

    assert!(!again.created);
    assert_eq!(again.size, data.len() as u64);
    assert!(first.path.exists() && second.path.exists());
}

#[tokio::test]
async fn an_existing_object_of_another_size_is_refused_and_never_overwritten() {
    let w = world();
    let (store, durable) = memory();
    let data = pattern(40_000, 3);
    let photo = land(&w, 1, 1, &data).await;
    let key = StorageKey::original(ProjectId(1), &sha(&data));
    store
        .put(key.as_path(), b"something else".to_vec().into())
        .await
        .unwrap();

    let err = durable.make_durable(&photo).await.unwrap_err();

    assert!(
        matches!(
            err,
            StorageError::SizeMismatch {
                expected: 40_000,
                found: 14,
                ..
            }
        ),
        "{err}"
    );
    assert_eq!(stored_bytes(store.as_ref(), &key).await, b"something else");
}

#[tokio::test]
async fn a_photo_over_the_single_put_limit_is_streamed_in_parts() {
    let w = world();
    let (store, durable) = memory();
    let data = pattern(9 * 1024 * 1024 + 123, 4);
    let photo = land(&w, 1, 1, &data).await;

    let object = durable.make_durable(&photo).await.unwrap();

    assert!(object.created);
    let key = StorageKey::original(ProjectId(1), &sha(&data));
    assert!(stored_bytes(store.as_ref(), &key).await == data);

    let twin = land(&w, 1, 2, &data).await;
    assert!(!durable.make_durable(&twin).await.unwrap().created);
}

#[tokio::test]
async fn a_landing_file_that_changed_size_is_refused() {
    let w = world();
    let (_, durable) = memory();
    let mut photo = land(&w, 1, 1, &pattern(1000, 5)).await;
    photo.size = 999;
    let err = durable.make_durable(&photo).await.unwrap_err();
    assert!(matches!(err, StorageError::SizeMismatch { .. }), "{err}");
}

#[tokio::test]
async fn a_local_store_with_the_landing_zone_inside_links_without_copying() {
    let w = world();
    let durable = local_store(&w, true);
    let data = pattern(50_000, 6);
    let photo = land(&w, 3, 1, &data).await;
    let inode = std::fs::metadata(&photo.path).unwrap().ino();

    let object = durable.make_durable(&photo).await.unwrap();

    assert!(object.created);
    let final_path = w.base.join(object.key.as_str());
    assert_eq!(std::fs::read(&final_path).unwrap(), data);
    assert_eq!(std::fs::metadata(&final_path).unwrap().ino(), inode);
    assert!(
        !photo.path.exists(),
        "the landing name is gone after the move"
    );
}

#[tokio::test]
async fn a_local_store_with_the_landing_zone_inside_accepts_the_same_content_again() {
    let w = world();
    let durable = local_store(&w, true);
    let data = pattern(30_000, 7);
    let first = land(&w, 3, 1, &data).await;
    let second = land(&w, 3, 2, &data).await;
    assert!(durable.make_durable(&first).await.unwrap().created);

    let again = durable.make_durable(&second).await.unwrap();

    assert!(!again.created);
    assert!(
        second.path.exists(),
        "the caller removes its own landing file"
    );
}

#[tokio::test]
async fn a_local_store_elsewhere_is_written_through_the_store() {
    let w = world();
    let durable = local_store(&w, false);
    let data = pattern(50_000, 8);
    let photo = land(&w, 3, 1, &data).await;

    let object = durable.make_durable(&photo).await.unwrap();

    let final_path = w.base.join("store").join(object.key.as_str());
    assert_eq!(std::fs::read(&final_path).unwrap(), data);
    assert!(photo.path.exists());
    assert_ne!(
        std::fs::metadata(&final_path).unwrap().ino(),
        std::fs::metadata(&photo.path).unwrap().ino()
    );
}

#[test]
fn the_settings_are_checked_and_a_missing_one_names_its_key() {
    let w = world();
    let from = |url: Option<String>| {
        let mut pairs = vec![(SOURCE_ROOT_KEY.to_owned(), "x".to_owned())];
        pairs.extend(url.map(|u| (STORAGE_URL_KEY.to_owned(), u)));
        Durable::from_settings(&Env::from_pairs(pairs), &w.roots)
            .unwrap_err()
            .to_string()
    };
    assert!(from(None).contains("OP_STORAGE_URL"));
    assert!(from(Some("ftp://x".into())).contains("file:///absolute/dir or s3://bucket"));
    assert!(from(Some("file://relative".into())).contains("absolute"));
    assert!(from(Some("file:///does/not/exist/at/all".into())).contains("not usable"));
    assert!(from(Some("s3://bucket/with/path".into())).contains("no path"));
    assert!(from(Some("s3://".into())).contains("no path"));
}

#[test]
fn an_s3_store_builds_without_touching_the_network() {
    let w = world();
    let env = Env::from_pairs([
        (STORAGE_URL_KEY, "s3://photos"),
        ("OP_STORAGE_S3_ENDPOINT", "http://127.0.0.1:9"),
        ("OP_STORAGE_S3_ALLOW_HTTP", "true"),
    ]);
    let durable = Durable::from_settings(&env, &w.roots).unwrap();
    assert!(durable.is_remote());
    let bad = Env::from_pairs([
        (STORAGE_URL_KEY, "s3://photos"),
        ("OP_STORAGE_S3_ALLOW_HTTP", "maybe"),
    ]);
    let err = Durable::from_settings(&bad, &w.roots)
        .unwrap_err()
        .to_string();
    assert!(err.contains("OP_STORAGE_S3_ALLOW_HTTP"), "{err}");
}

#[test]
fn building_an_s3_store_installs_the_ring_tls_provider_once() {
    let w = world();
    let env = Env::from_pairs([(STORAGE_URL_KEY, "s3://photos")]);
    Durable::from_settings(&env, &w.roots).unwrap();
    Durable::from_settings(&env, &w.roots).unwrap();
    let ring = rustls::crypto::ring::default_provider();
    let installed = rustls::crypto::CryptoProvider::get_default().expect("a provider is installed");
    assert_eq!(installed.cipher_suites, ring.cipher_suites);
}

#[tokio::test]
async fn the_landing_file_is_synced_before_it_is_linked() {
    let w = world();
    let durable = local_store(&w, true);
    let photo = land(&w, 3, 1, &pattern(1_000, 9)).await;
    std::fs::remove_file(&photo.path).unwrap();

    let err = durable.make_durable(&photo).await.unwrap_err();

    assert!(
        matches!(err, StorageError::Io { op: "open", .. }),
        "the sync of the landing file comes first: {err}"
    );
}

#[test]
fn a_local_store_syncs_files_and_directories_on_every_write() {
    let w = world();
    for in_store in [true, false] {
        let shown = format!("{:?}", local_store(&w, in_store).store());
        assert!(shown.contains("fsync: true"), "{shown}");
    }
}
