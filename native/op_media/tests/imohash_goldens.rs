//! The port against the goldens `scripts/codegen/imohash_goldens.py` wrote from the Python
//! `imohash` package: synthetic sizes around the sampling threshold, a committed image tiled past
//! it, sparse files with long varints, and every committed test image.

use std::fs::File;
use std::io::{Cursor, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use op_media::imohash::{self, SAMPLE_SIZE, SAMPLE_THRESHOLD};
use serde_json::Value;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn goldens() -> Value {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/goldens/imohash.json"),
    )
    .expect("the goldens file is committed");
    serde_json::from_str(&text).expect("the goldens file is JSON")
}

fn rows(goldens: &Value, key: &str) -> Vec<Value> {
    goldens[key].as_array().expect("a list").clone()
}

fn pattern(size: usize) -> Vec<u8> {
    (0..size as u64)
        .map(|i| (((i * 2_654_435_761) >> 13) ^ (i >> 16)) as u8)
        .collect()
}

fn hex_of(data: &[u8]) -> String {
    imohash::hash_reader(&mut Cursor::new(data))
        .expect("an in-memory read")
        .to_hex()
}

#[test]
fn the_constants_are_the_package_defaults() {
    let g = goldens();
    assert_eq!(g["threshold"].as_u64().unwrap(), SAMPLE_THRESHOLD);
    assert_eq!(g["sample_size"].as_u64().unwrap(), SAMPLE_SIZE);
}

#[test]
fn synthetic_sizes_around_the_thresholds_equal_python() {
    let g = goldens();
    let cases = rows(&g, "synthetic");
    assert!(cases.len() > 30);
    for case in cases {
        let size = case["size"].as_u64().unwrap() as usize;
        assert_eq!(hex_of(&pattern(size)), case["hex"], "size {size}");
    }
}

#[test]
fn a_committed_image_tiled_past_the_threshold_equals_python() {
    for case in rows(&goldens(), "tiled") {
        let bytes = std::fs::read(repo_root().join(case["file"].as_str().unwrap())).unwrap();
        let data = bytes.repeat(case["copies"].as_u64().unwrap() as usize);
        assert_eq!(data.len() as u64, case["size"].as_u64().unwrap());
        assert!(data.len() as u64 >= SAMPLE_THRESHOLD);
        assert_eq!(hex_of(&data), case["hex"], "{}", case["file"]);
    }
}

#[test]
fn every_committed_test_image_equals_python() {
    let cases = rows(&goldens(), "files");
    assert!(cases.len() > 60, "{} images", cases.len());
    for case in cases {
        let path = repo_root().join(case["file"].as_str().unwrap());
        let hash = block_on(imohash::hash_file(path)).unwrap();
        assert_eq!(hash.to_hex(), case["hex"], "{}", case["file"]);
    }
}

#[test]
fn sparse_files_with_long_size_varints_equal_python() {
    let g = goldens();
    let marker = g["marker"].as_str().unwrap().as_bytes().to_vec();
    let dir = tempfile::tempdir().unwrap();
    for case in rows(&g, "sparse") {
        let size = case["size"].as_u64().unwrap();
        let path = dir.path().join("sparse.bin");
        let mut file = File::create(&path).unwrap();
        file.set_len(size).unwrap();
        for (index, offset) in [0, size / 2, size - SAMPLE_SIZE].into_iter().enumerate() {
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&marker).unwrap();
            file.write_all(&[index as u8]).unwrap();
        }
        drop(file);
        let hash = block_on(imohash::hash_file(&path)).unwrap();
        assert_eq!(hash.to_hex(), case["hex"], "size {size}");
        std::fs::remove_file(&path).unwrap();
    }
}

#[test]
fn only_the_sampled_ranges_decide_the_hash() {
    let size = 1_000_000_usize;
    let base = pattern(size);
    let reference = hex_of(&base);
    let mut changed = base.clone();
    changed[SAMPLE_SIZE as usize + 5] ^= 0xff;
    changed[size / 2 - 1] ^= 0xff;
    changed[size - SAMPLE_SIZE as usize - 1] ^= 0xff;
    assert_eq!(hex_of(&changed), reference, "bytes outside the samples");
    for inside in [0, SAMPLE_SIZE as usize - 1, size / 2, size - 1] {
        let mut changed = base.clone();
        changed[inside] ^= 0xff;
        assert_ne!(hex_of(&changed), reference, "byte {inside}");
    }
}

#[test]
fn a_short_read_is_an_error_and_not_a_hash() {
    struct Shrinking(Cursor<Vec<u8>>);
    impl std::io::Read for Shrinking {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            // The tail sample sits at the end: report the size of the whole file, serve less.
            let n = self.0.read(buf)?;
            Ok(if self.0.position() > 140_000 { 0 } else { n })
        }
    }
    impl Seek for Shrinking {
        fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
            self.0.seek(pos)
        }
    }
    let mut reader = Shrinking(Cursor::new(pattern(200_000)));
    let err = imohash::hash_reader(&mut reader).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}

#[test]
fn a_missing_file_is_an_error() {
    let err = block_on(imohash::hash_file("/nonexistent/photo.jpg")).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("a runtime")
        .block_on(future)
}
