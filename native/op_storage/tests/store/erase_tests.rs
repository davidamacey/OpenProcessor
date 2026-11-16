use std::sync::Arc;

use futures_util::TryStreamExt;
use object_store::ObjectStoreExt;
use object_store::memory::InMemory;
use op_storage::durable::{Backend, Durable};
use op_storage::erase;
use op_storage::error::StorageError;
use op_storage::keys::{ProjectId, StorageKey};
use op_storage::reconcile::list_originals;

use crate::support::{land, pattern, sha, world};

fn memory() -> (Arc<InMemory>, Durable) {
    let store = Arc::new(InMemory::new());
    (
        store.clone(),
        Durable::from_store(store, Backend::Streamed, true),
    )
}

async fn put(store: &InMemory, key: &str) {
    store
        .put(
            StorageKey::parse(key).unwrap().as_path(),
            vec![1_u8; 10].into(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn erasing_a_key_twice_is_a_no_op() {
    let (store, durable) = memory();
    put(&store, "projects/1/originals/ab/cd").await;
    let key = StorageKey::parse("projects/1/originals/ab/cd").unwrap();
    erase::delete(&durable, &key).await.unwrap();
    assert!(!durable.exists(&key).await.unwrap());
    erase::delete(&durable, &key).await.unwrap();
}

#[tokio::test]
async fn delete_by_prefix_covers_whole_segments_and_twice_is_a_no_op() {
    let (store, durable) = memory();
    for key in [
        "projects/1/derived/crops/4/a",
        "projects/1/derived/crops/4/sub/b",
        "projects/1/derived/crops/42/a",
        "projects/2/derived/crops/4/a",
    ] {
        put(&store, key).await;
    }
    let prefix = StorageKey::parse("projects/1/derived/crops/4").unwrap();
    assert_eq!(erase::delete_prefix(&durable, &prefix).await.unwrap(), 2);
    assert_eq!(erase::delete_prefix(&durable, &prefix).await.unwrap(), 0);
    for kept in [
        "projects/1/derived/crops/42/a",
        "projects/2/derived/crops/4/a",
    ] {
        assert!(
            durable
                .exists(&StorageKey::parse(kept).unwrap())
                .await
                .unwrap()
        );
    }
}

#[tokio::test]
async fn source_and_landing_keys_are_never_erased() {
    let (_, durable) = memory();
    for key in ["source/a/b.jpg", "landing/1/x"] {
        let key = StorageKey::parse(key).unwrap();
        assert!(matches!(
            erase::delete(&durable, &key).await,
            Err(StorageError::Refused { .. })
        ));
        assert!(matches!(
            erase::delete_prefix(&durable, &key).await,
            Err(StorageError::Refused { .. })
        ));
    }
}

#[tokio::test]
async fn the_originals_of_one_project_are_listed_with_size_and_age() {
    let w = world();
    let (store, durable) = memory();
    put(&store, "projects/1/derived/thumbs/1/a.jpg").await;
    let a = pattern(1_000, 1);
    let b = pattern(2_000, 2);
    durable
        .make_durable(&land(&w, 1, 1, &a).await)
        .await
        .unwrap();
    durable
        .make_durable(&land(&w, 1, 2, &b).await)
        .await
        .unwrap();
    durable
        .make_durable(&land(&w, 2, 3, &a).await)
        .await
        .unwrap();

    let mut listed: Vec<_> = list_originals(&durable, ProjectId(1))
        .try_collect()
        .await
        .unwrap();
    listed.sort_by_key(|o| o.size);

    let keys: Vec<_> = listed.iter().map(|o| (o.key.clone(), o.size)).collect();
    assert_eq!(
        keys,
        [
            (StorageKey::original(ProjectId(1), &sha(&a)), 1_000),
            (StorageKey::original(ProjectId(1), &sha(&b)), 2_000),
        ]
    );
    assert!(listed.iter().all(|o| o.age.as_secs() < 60));
}
