//! The pool against the real ExifTool. These tests need `exiftool` (12.15 or newer) and
//! `setpriv`, so they are `#[ignore]`d in the offline suite and run by the sweep, in the
//! `op-server` image:
//!
//! ```text
//! cargo test -p op_media --test exiftool_real -- --ignored --test-threads=1
//! ```
//!
//! The list test reads `OP_MEDIA_LIST` (one file path per line) and writes `OP_MEDIA_OUT`, one JSON
//! line per file: `{"path": ..., "capture": ...}` in the shape of `contracts/json/capture_cases.json`
//! (`expected`), or `{"path": ..., "error": ...}`. Python's `parse_capture(read_capture_file(...))`
//! over the same list must give the same `capture` on every line.

mod common;

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use common::capture_json;
use op_config::Env;
use op_media::capture::{READ_TAGS, parse_capture};
use op_media::exiftool::{ExifToolPool, ExifToolSettings};
use serde_json::{Value, json};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The pool and how many reads to keep in flight: the pool's size, so no read is refused for load.
async fn real_pool() -> (Arc<ExifToolPool>, usize) {
    let settings = ExifToolSettings::from_env(&Env::from_process()).expect("settings");
    let width = settings.pool_size;
    let pool = ExifToolPool::start(settings)
        .await
        .expect("a real exiftool starts");
    (Arc::new(pool), width)
}

#[tokio::test]
#[ignore = "owed to the sweep: needs the real exiftool and setpriv (the op-server image)"]
async fn the_real_exiftool_returns_the_recorded_rows_of_the_committed_images() {
    let recorded: Value = serde_json::from_str(
        &std::fs::read_to_string(
            repo_root().join("native/op_media/tests/fixtures/exiftool/recorded.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let (pool, width) = real_pool().await;
    let mut jobs = Vec::new();
    for item in recorded["rows"].as_array().unwrap() {
        let name = item["name"].as_str().unwrap().to_owned();
        if !name.starts_with("scenario/") {
            let mut expected = item["row"].as_object().unwrap().clone();
            expected.remove("SourceFile");
            jobs.push((name, expected));
        }
    }
    assert!(jobs.len() > 60);
    for chunk in jobs.chunks(width) {
        let tasks: Vec<_> = chunk
            .iter()
            .cloned()
            .map(|(name, expected)| {
                let (pool, path) = (Arc::clone(&pool), repo_root().join(&name));
                tokio::spawn(async move {
                    let mut got = pool
                        .read(&path, &READ_TAGS)
                        .await
                        .unwrap_or_else(|e| panic!("{name}: {e}"));
                    got.remove("SourceFile");
                    (name, expected, got)
                })
            })
            .collect();
        for task in tasks {
            let (name, expected, got) = task.await.unwrap();
            assert_eq!(got, expected, "{name}");
        }
    }
    pool.close().await;
}

#[tokio::test]
#[ignore = "owed to the sweep: needs the real exiftool and OP_MEDIA_LIST, OP_MEDIA_OUT (see the module doc)"]
async fn capture_lines_for_a_list_of_files() {
    let env = Env::from_process();
    let list: PathBuf = env.require("OP_MEDIA_LIST").expect("OP_MEDIA_LIST");
    let out: PathBuf = env.require("OP_MEDIA_OUT").expect("OP_MEDIA_OUT");
    let (pool, width) = real_pool().await;
    let paths: Vec<String> = std::fs::read_to_string(list)
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect();
    let mut lines = String::new();
    // One pool's worth of reads in flight at a time keeps the spawned tasks bounded.
    for chunk in paths.chunks(width) {
        let tasks: Vec<_> = chunk
            .iter()
            .map(|path| {
                let (pool, path) = (Arc::clone(&pool), path.clone());
                tokio::spawn(async move {
                    match pool.read(Path::new(&path), &READ_TAGS).await {
                        Ok(tags) => {
                            json!({"path": path, "capture": capture_json(&parse_capture(&tags))})
                        }
                        Err(error) => json!({"path": path, "error": error.to_string()}),
                    }
                })
            })
            .collect();
        for task in tasks {
            writeln!(lines, "{}", task.await.unwrap()).unwrap();
        }
    }
    std::fs::write(out, lines).unwrap();
    pool.close().await;
}
