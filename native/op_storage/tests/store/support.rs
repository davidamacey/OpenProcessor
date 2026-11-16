//! Shared set-up of the storage integration tests: a temporary landing zone and photos landed in it.

use std::path::PathBuf;

use bytes::Bytes;
use op_storage::hash::Sha256Stream;
use op_storage::keys::{ContentHash, ProjectId};
use op_storage::landing::{LandedPhoto, Landing};
use op_storage::local::LocalRoots;
use tempfile::TempDir;
use uuid::Uuid;

pub struct World {
    _dir: TempDir,
    pub base: PathBuf,
    pub roots: LocalRoots,
    pub landing: Landing,
}

/// A base folder with `landing/` and `sources/` inside (the landing root is `<base>/landing`).
pub fn world() -> World {
    let dir = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(dir.path()).unwrap();
    std::fs::create_dir(base.join("landing")).unwrap();
    std::fs::create_dir(base.join("sources")).unwrap();
    let roots = LocalRoots::new(&base.join("landing"), &base.join("sources")).unwrap();
    World {
        landing: Landing::new(&roots),
        roots,
        base,
        _dir: dir,
    }
}

pub fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i * 31 + i / 7 + seed) % 251) as u8)
        .collect()
}

pub fn sha(data: &[u8]) -> ContentHash {
    let mut s = Sha256Stream::new();
    s.update(data);
    s.finish()
}

/// Land `data` as part `upload`/0 of `project` and give it its landing name.
pub async fn land(w: &World, project: i64, upload: u128, data: &[u8]) -> LandedPhoto {
    let mut part = w.landing.begin_part(Uuid::from_u128(upload), 0);
    for chunk in data.chunks(65_536) {
        part.write(&Bytes::copy_from_slice(chunk)).await.unwrap();
    }
    let part = part.finish().await.unwrap();
    w.landing
        .name_by_content(ProjectId(project), part)
        .await
        .unwrap()
}
