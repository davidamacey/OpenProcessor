//! A photo read by key through the flow on the host fake: it answers what the same bytes in a
//! request answer, whatever the file is, and a file the engine cannot use answers its status
//! before any decode. The fake's host pool stands for the finish pool: the files open at once
//! are at most its threads.

use std::fs;
use std::os::unix::fs::symlink;
use std::path::PathBuf;

use op_local_file::{FileClaim, FileRef};
use tempfile::TempDir;

use super::read::KeyedRoots;
use super::*;
use crate::wire::contract::contract;

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/host");

struct Disk {
    _dir: TempDir,
    base: PathBuf,
}

fn disk() -> Disk {
    let dir = tempfile::tempdir().expect("a temporary directory");
    let base = fs::canonicalize(dir.path()).expect("canonical");
    for d in ["landing/7", "source"] {
        fs::create_dir_all(base.join(d)).expect("directory");
    }
    Disk { _dir: dir, base }
}

fn keyed_bed(d: &Disk) -> Bed {
    let mut b = bed();
    b.flow.settings.keyed = Arc::new(
        KeyedRoots::new(
            Some(&d.base.join("landing")),
            Some(&d.base.join("source")),
            256 * MIB as u64,
        )
        .expect("roots"),
    );
    b
}

/// A photo whose file is written under `key` and whose claim is made for `bytes`.
fn keyed(d: &Disk, key: &str, bytes: &Bytes, plan: &str) -> Photo {
    fs::write(d.base.join(key), bytes).expect("a file");
    let mut photo = photo_of(Bytes::new(), plan);
    photo.image = ImageInput::File(FileRef::new(key, FileClaim::of(bytes)).expect("a local key"));
    photo
}

fn corpus(name: &str) -> Bytes {
    Bytes::from(fs::read(format!("{CORPUS}/{name}")).expect("corpus file"))
}

/// The fake's log without each call's deadline, which is the request's clock and differs.
fn device_work(b: &Bed) -> Vec<String> {
    b.fake
        .events()
        .iter()
        .map(|e| {
            let text = format!("{e:?}");
            let Some(at) = text.find(", deadline: Instant {") else {
                return text;
            };
            let end = text[at..].find('}').map_or(text.len(), |n| at + n + 1);
            format!("{}{}", &text[..at], &text[end..])
        })
        .collect()
}

fn assert_same(keyed: &PhotoAnswer, plain: &PhotoAnswer, what: &str) {
    assert_eq!(keyed.status, plain.status, "{what}: status");
    assert_eq!(keyed.decode_route, plain.decode_route, "{what}: route");
    assert_eq!(keyed.branches, plain.branches, "{what}: branches");
    assert_eq!(keyed.outputs, plain.outputs, "{what}: outputs");
}

#[tokio::test]
async fn a_photo_read_by_key_answers_what_its_bytes_answer_for_every_kind_of_file() {
    let d = disk();
    let files = [
        ("a jpeg", jpeg(1000, 750, 1)),
        ("a rotated jpeg", jpeg(640, 480, 6)),
        ("a png", corpus("png_rgb.png")),
        ("a progressive jpeg", corpus("jpeg_progressive.jpg")),
        (
            "a truncated jpeg",
            corpus("damaged_truncated_no_eoi_gpu.jpg"),
        ),
        ("a text file", corpus("damaged_text.bin")),
        ("an empty file", Bytes::new()),
    ];
    for (n, (what, bytes)) in files.iter().enumerate() {
        let (by_key, by_bytes) = (keyed_bed(&d), bed());
        let key = format!("landing/7/photo{n}.0");
        let a = by_key
            .flow
            .process(keyed(&d, &key, bytes, INGEST))
            .await
            .expect("keyed answer");
        let b = by_bytes
            .flow
            .process(photo_of(bytes.clone(), INGEST))
            .await
            .expect("bytes answer");
        assert_same(&a, &b, what);
        assert_eq!(
            device_work(&by_key),
            device_work(&by_bytes),
            "{what}: device work"
        );
        assert_released(&by_key).await;
    }
}

#[tokio::test]
async fn a_source_key_reads_the_same_as_a_landing_key() {
    let d = disk();
    let b = keyed_bed(&d);
    let bytes = jpeg(1000, 750, 1);
    let answer = b
        .flow
        .process(keyed(&d, "source/IMG_1.jpg", &bytes, FRAME))
        .await
        .expect("answer");
    assert_eq!(
        (answer.status.as_str(), answer.decode_route.as_str()),
        ("ok", "gpu")
    );
}

#[tokio::test]
async fn a_file_the_engine_cannot_use_answers_its_status_before_any_decode_or_reservation() {
    let d = disk();
    let b = keyed_bed(&d);
    let bytes = jpeg(1000, 750, 1);
    // missing: the claim is made for a file that is not there
    let gone = keyed(&d, "source/gone.jpg", &bytes, INGEST);
    fs::remove_file(d.base.join("source/gone.jpg")).expect("remove");
    let mut cases = vec![("missing", gone)];
    // changed: the file was rewritten after the claim
    let changed = keyed(&d, "source/changed.jpg", &bytes, INGEST);
    fs::write(d.base.join("source/changed.jpg"), jpeg(999, 750, 1)).expect("rewrite");
    cases.push(("changed", changed));
    // too_large: the claim is above the message limit
    let mut big = photo(INGEST);
    big.image = ImageInput::File(
        FileRef::new(
            "source/huge.jpg",
            FileClaim {
                size: 257 * MIB as u64,
                ..FileClaim::of(b"")
            },
        )
        .expect("key"),
    );
    cases.push(("too_large", big));
    // unreadable: a symlink that leaves the root
    let outside = d.base.join("outside.jpg");
    fs::write(&outside, &bytes).expect("outside");
    symlink(&outside, d.base.join("source/leave.jpg")).expect("symlink");
    let mut leave = photo(INGEST);
    leave.image =
        ImageInput::File(FileRef::new("source/leave.jpg", FileClaim::of(&bytes)).expect("key"));
    cases.push(("unreadable", leave));
    for (status, photo) in cases {
        let plan = photo.plan.clone();
        let answer = b
            .flow
            .process(photo)
            .await
            .expect("an answer, not an error");
        assert_eq!(answer.status, status);
        assert_eq!(answer.decode_route, "none");
        assert!(answer.outputs.is_empty());
        assert_eq!(answer.branches, refused(&plan, status).branches);
        assert!(
            contract()
                .read_by_key
                .photo_statuses
                .iter()
                .any(|s| s.status == status),
            "{status} is in the contract"
        );
    }
    assert_eq!(
        b.fake.events(),
        [],
        "nothing reached the card or the model server"
    );
    assert_released(&b).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn many_keyed_photos_never_have_more_files_open_than_the_host_pool_has_threads() {
    let d = disk();
    let b = Arc::new(keyed_bed(&d));
    let bytes = Bytes::from(vec![7u8; 16 * MIB]);
    fs::write(d.base.join("source/real.bin"), &bytes).expect("one big file");
    let mut tasks = Vec::new();
    for n in 0..12 {
        let key = format!("source/l{n}.bin");
        symlink("real.bin", d.base.join(&key)).expect("alias inside the root");
        let mut photo = photo(FRAME);
        photo.image = ImageInput::File(FileRef::new(&key, FileClaim::of(&bytes)).expect("key"));
        let bed = Arc::clone(&b);
        tasks.push(tokio::spawn(async move { bed.flow.process(photo).await }));
    }
    for task in tasks {
        let answer = task.await.expect("joined").expect("answered");
        assert_eq!(answer.decode_route, "none", "not an image: decoded nowhere");
    }
    let peak = b.fake.host_peak.load(Ordering::SeqCst);
    assert!(
        (1..=2).contains(&peak),
        "peak {peak}: the fake has 2 host threads"
    );
}
