//! Against a real S3-compatible server. Owed to the sweep: a lane has no network service. Run with
//!
//! ```text
//! AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret> AWS_DEFAULT_REGION=us-east-1 \
//! OP_TEST_S3_ENDPOINT=http://127.0.0.1:<port> OP_TEST_S3_BUCKET=<existing bucket> \
//! cargo test -p op_storage --test store s3_ -- --ignored
//! ```
//!
//! Pass rule: both tests pass. The bucket must exist and its server must support conditional put
//! (`If-None-Match`), as Amazon S3, MinIO and Cloudflare R2 do.

use std::time::{SystemTime, UNIX_EPOCH};

use object_store::ObjectStoreExt;
use op_config::Env;
use op_storage::durable::{Durable, STORAGE_URL_KEY};
use op_storage::error::StorageError;
use op_storage::keys::StorageKey;

use crate::support::{land, pattern, sha, world};

fn server() -> Env {
    let var = |name: &str| std::env::var(name).unwrap_or_else(|_| panic!("{name} must be set"));
    Env::from_pairs([
        (
            STORAGE_URL_KEY.to_owned(),
            format!("s3://{}", var("OP_TEST_S3_BUCKET")),
        ),
        (
            "OP_STORAGE_S3_ENDPOINT".to_owned(),
            var("OP_TEST_S3_ENDPOINT"),
        ),
        ("OP_STORAGE_S3_ALLOW_HTTP".to_owned(), "true".to_owned()),
    ])
}

/// A project id and a data seed no earlier run used, so the bytes are new to the bucket.
fn fresh() -> (i64, usize) {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    (
        i64::try_from(nanos % 1_000_000_000).unwrap(),
        usize::try_from(nanos % 251).unwrap(),
    )
}

#[tokio::test]
#[ignore = "owed to the sweep: needs an S3-compatible server (see the module docs for the command)"]
async fn s3_a_small_photo_is_created_once_and_a_resend_is_accepted() {
    let w = world();
    let durable = Durable::from_settings(&server(), &w.roots).unwrap();
    let (project, seed) = fresh();
    let data = pattern(200_000, seed);
    let first = durable
        .make_durable(&land(&w, project, 1, &data).await)
        .await
        .unwrap();
    assert!(first.created);
    let second = durable
        .make_durable(&land(&w, project, 2, &data).await)
        .await
        .unwrap();
    assert!(!second.created);
    let stored = durable
        .store()
        .get(first.key.as_path())
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(stored.as_ref(), data.as_slice());

    let mut wrong = land(&w, project, 3, &data).await;
    wrong.size += 1;
    let err = durable.make_durable(&wrong).await.unwrap_err();
    assert!(matches!(err, StorageError::SizeMismatch { .. }), "{err}");
    let key = StorageKey::original(wrong.project, &sha(&data));
    assert!(durable.exists(&key).await.unwrap());
}

#[tokio::test]
#[ignore = "owed to the sweep: needs an S3-compatible server (see the module docs for the command)"]
async fn s3_a_big_photo_goes_through_a_multipart_upload_with_the_checksum() {
    let w = world();
    let durable = Durable::from_settings(&server(), &w.roots).unwrap();
    let (project, seed) = fresh();
    let data = pattern(21 * 1024 * 1024 + 7, seed);
    let object = durable
        .make_durable(&land(&w, project, 1, &data).await)
        .await
        .unwrap();
    assert!(object.created);
    assert_eq!(object.size, data.len() as u64);
    let again = durable
        .make_durable(&land(&w, project, 2, &data).await)
        .await
        .unwrap();
    assert!(!again.created);
}
