//! The landing zone through its public interface, as the upload receiver uses it.
#![cfg(test)]

use std::os::unix::fs::MetadataExt;
use std::path::Path;

use bytes::Bytes;
use op_storage::error::StorageError;
use op_storage::hash::Sha256Stream;
use op_storage::keys::{ContentHash, ProjectId, StorageKey};
use op_storage::landing::{LandedPart, Landing};
use op_storage::local::LocalRoots;
use tempfile::TempDir;
use uuid::Uuid;

struct Fixture {
    _dir: TempDir,
    landing: Landing,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(dir.path()).unwrap();
    std::fs::create_dir(base.join("landing")).unwrap();
    std::fs::create_dir(base.join("sources")).unwrap();
    let roots = LocalRoots::new(&base.join("landing"), &base.join("sources")).unwrap();
    Fixture {
        landing: Landing::new(&roots),
        _dir: dir,
    }
}

fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 31 + i / 7) % 251) as u8).collect()
}

fn one_shot(data: &[u8]) -> ContentHash {
    let mut s = Sha256Stream::new();
    s.update(data);
    s.finish()
}

async fn land(f: &Fixture, upload: u128, index: u32, data: &[u8], chunk: usize) -> LandedPart {
    let mut w = f.landing.begin_part(Uuid::from_u128(upload), index);
    for c in data.chunks(chunk.max(1)) {
        w.write(&Bytes::copy_from_slice(c)).await.unwrap();
    }
    w.finish().await.unwrap()
}

fn files_under(dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else {
                out.push(p.strip_prefix(dir).unwrap().display().to_string());
            }
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn the_part_hash_equals_a_one_shot_hash_whatever_the_chunking() {
    let f = fixture();
    let data = pattern(300_000);
    for (i, chunk) in [1_usize, 4096, 65_536, 65_537, 300_000]
        .into_iter()
        .enumerate()
    {
        let part = land(&f, 1, u32::try_from(i).unwrap(), &data, chunk).await;
        assert_eq!(part.sha256, one_shot(&data), "chunk size {chunk}");
        assert_eq!(part.size, data.len() as u64);
        assert_eq!(std::fs::read(&part.path).unwrap(), data);
    }
}

#[tokio::test]
async fn an_empty_part_is_a_zero_byte_file_with_the_empty_digest() {
    let f = fixture();
    let part = f
        .landing
        .begin_part(Uuid::from_u128(5), 0)
        .finish()
        .await
        .unwrap();
    assert_eq!(part.size, 0);
    assert_eq!(part.sha256, one_shot(b""));
    assert!(part.path.is_file());
}

#[tokio::test]
async fn an_unfinished_part_is_never_visible_under_a_content_name() {
    let f = fixture();
    let data = pattern(500_000);
    let hash = one_shot(&data);
    let project = ProjectId(4);
    let content_path = f.landing.root().join("4").join(hash.to_string());

    let mut w = f.landing.begin_part(Uuid::from_u128(9), 0);
    for c in data.chunks(100_000) {
        w.write(&Bytes::copy_from_slice(c)).await.unwrap();
        assert!(
            !content_path.exists(),
            "visible while the part is being written"
        );
    }
    let part = w.finish().await.unwrap();
    assert!(!content_path.exists(), "visible before it is named");

    let photo = f
        .landing
        .name_by_content(project, part.clone())
        .await
        .unwrap();
    assert_eq!(photo.path, content_path);
    assert_eq!(photo.key, StorageKey::landing(project, &hash));
    assert_eq!((photo.size, photo.sha256), (part.size, part.sha256));
    assert_eq!(std::fs::read(&photo.path).unwrap(), data);
    assert!(
        !part.path.exists(),
        "the part name is gone after the rename"
    );
}

#[tokio::test]
async fn naming_by_content_is_a_rename_that_keeps_the_inode() {
    let f = fixture();
    let part = land(&f, 2, 0, &pattern(10_000), 4096).await;
    let before = std::fs::metadata(&part.path).unwrap().ino();
    let photo = f.landing.name_by_content(ProjectId(1), part).await.unwrap();
    assert_eq!(std::fs::metadata(&photo.path).unwrap().ino(), before);
}

#[tokio::test]
async fn a_second_writer_for_the_same_part_is_refused() {
    let f = fixture();
    let first = land(&f, 3, 0, b"one", 3).await;
    let mut second = f.landing.begin_part(Uuid::from_u128(3), 0);
    let err = second.write(&Bytes::from_static(b"two")).await.unwrap_err();
    assert!(matches!(err, StorageError::Io { .. }), "{err}");
    assert_eq!(std::fs::read(&first.path).unwrap(), b"one");
}

#[tokio::test]
async fn discard_removes_the_part_and_is_fine_for_a_part_never_created() {
    let f = fixture();
    let mut w = f.landing.begin_part(Uuid::from_u128(6), 0);
    w.write(&Bytes::from_static(b"abc")).await.unwrap();
    w.discard().await.unwrap();
    assert!(files_under(f.landing.root()).is_empty());
    f.landing
        .begin_part(Uuid::from_u128(7), 0)
        .discard()
        .await
        .unwrap();
}

#[tokio::test]
async fn remove_deletes_the_photo_and_twice_is_a_no_op() {
    let f = fixture();
    let part = land(&f, 8, 0, b"photo", 5).await;
    let photo = f.landing.name_by_content(ProjectId(1), part).await.unwrap();
    f.landing.remove(&photo).await.unwrap();
    assert!(!photo.path.exists());
    f.landing.remove(&photo).await.unwrap();
}

#[tokio::test]
async fn paths_outside_the_landing_root_are_refused() {
    let f = fixture();
    let other = tempfile::tempdir().unwrap();
    let stray = other.path().join("x.part");
    std::fs::write(&stray, b"x").unwrap();
    let part = LandedPart {
        path: stray.clone(),
        size: 1,
        sha256: one_shot(b"x"),
    };
    let err = f
        .landing
        .name_by_content(ProjectId(1), part)
        .await
        .unwrap_err();
    assert!(matches!(err, StorageError::OutsideRoot { .. }), "{err}");
    assert!(stray.exists());

    let dotted = f.landing.root().join("..").join("escape");
    let part = LandedPart {
        path: dotted,
        size: 1,
        sha256: one_shot(b"x"),
    };
    assert!(matches!(
        f.landing.name_by_content(ProjectId(1), part).await,
        Err(StorageError::OutsideRoot { .. })
    ));
}

#[tokio::test]
async fn free_ratio_is_a_fraction_of_the_volume() {
    let f = fixture();
    let ratio = f.landing.free_ratio().await.unwrap();
    assert!((0.0..=1.0).contains(&ratio) && ratio > 0.0, "{ratio}");
}
