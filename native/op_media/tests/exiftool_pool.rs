//! The ExifTool pool against a fake ExifTool (`tests/fixtures/fake_exiftool.sh`) that speaks the
//! `-stay_open` protocol, so the suite needs no ExifTool. The fake's file-name suffixes make a
//! process hang, die, write to standard error or answer badly (see that script).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use op_config::Env;
use op_media::capture::{READ_TAGS, read_capture_file};
use op_media::exiftool::{Busy, ExifToolError, ExifToolPool, ExifToolSettings, Tags};

const TAGS: [&str; 2] = ["EXIF:Orientation", "ExifTool:Error"];

fn fake() -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fake_exiftool.sh")
        .display()
        .to_string()
}

fn settings(pairs: &[(&str, &str)]) -> ExifToolSettings {
    let mut all = vec![("OP_EXIFTOOL_BIN", fake())];
    all.extend(pairs.iter().map(|(k, v)| (*k, (*v).to_owned())));
    ExifToolSettings::from_env(&Env::from_pairs(all)).expect("valid settings")
}

async fn pool(pairs: &[(&str, &str)]) -> ExifToolPool {
    ExifToolPool::start(settings(pairs))
        .await
        .expect("the fake starts")
}

fn pid(tags: &Tags) -> u64 {
    tags["Fake:Pid"].as_u64().expect("the fake writes its pid")
}

fn alive(pid: u64) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

fn args(tags: &Tags) -> Vec<String> {
    tags["Fake:Args"]
        .as_array()
        .expect("the fake writes its arguments")
        .iter()
        .map(|a| a.as_str().expect("text").to_owned())
        .collect()
}

struct Dir(tempfile::TempDir);

impl Dir {
    fn new() -> Self {
        Self(tempfile::tempdir().expect("a temporary directory"))
    }

    fn file(&self, name: &str) -> PathBuf {
        self.0.path().join(name)
    }
}

#[tokio::test]
async fn a_read_sends_the_arguments_of_the_python_client_framed_with_execute() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let path = dir.file("a.jpg");
    let tags = pool.read(&path, &TAGS).await.unwrap();
    assert_eq!(
        args(&tags),
        ["-j", "-fast", "-EXIF:Orientation", "-ExifTool:Error"]
            .into_iter()
            .map(String::from)
            .chain([path.display().to_string()])
            .collect::<Vec<_>>()
    );
    assert_eq!(tags["SourceFile"], path.display().to_string());
}

#[tokio::test]
async fn each_request_on_a_process_has_its_own_execute_number() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap();
    let second = pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap();
    // The start-up `-ver` was request 1.
    assert_eq!(first["Fake:Execute"], 2);
    assert_eq!(second["Fake:Execute"], 3);
}

#[tokio::test]
async fn the_capture_tags_go_out_in_order_each_with_a_dash() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let sent = args(&pool.read(&dir.file("a.jpg"), &READ_TAGS).await.unwrap());
    assert_eq!(sent.len(), 2 + 21 + 1);
    assert_eq!(&sent[..3], ["-j", "-fast", "-EXIF:DateTimeOriginal"]);
    assert_eq!(sent[22], "-ExifTool:Error");
}

#[tokio::test]
async fn paths_with_spaces_and_non_ascii_names_arrive_whole() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let path = dir.file(" sp ace é.jpg ");
    let tags = pool.read(&path, &TAGS).await.unwrap();
    assert_eq!(args(&tags).last().unwrap(), &path.display().to_string());
}

#[tokio::test]
async fn paths_exiftool_would_misread_are_refused_before_any_process_is_used() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    for bad in [
        "",
        "a\nb.jpg",
        "a\rb.jpg",
        "-all=",
        "#comment.jpg",
        "a\0b.jpg",
    ] {
        assert_eq!(
            pool.read(Path::new(bad), &TAGS).await.unwrap_err(),
            ExifToolError::InvalidPath,
            "{bad:?}"
        );
    }
    let again = pid(&pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap());
    assert_eq!(first, again, "the process was not touched");
}

#[tokio::test]
async fn an_answer_split_over_many_reads_of_the_pipe_is_read_whole() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let tags = pool.read(&dir.file("a.big"), &TAGS).await.unwrap();
    assert_eq!(tags["Fake:Pad"].as_str().unwrap().len(), 300_000);
    let next = pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap();
    assert!(
        next.contains_key("Fake:Pid"),
        "the next answer is not shifted"
    );
}

#[tokio::test]
async fn one_process_serves_reads_in_turn() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    let second = pid(&pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap());
    assert_eq!(first, second);
}

#[tokio::test]
async fn a_timeout_kills_and_reaps_the_process_and_the_next_read_gets_a_new_one() {
    let dir = Dir::new();
    let pool = pool(&[
        ("OP_EXIFTOOL_POOL_SIZE", "1"),
        ("OP_EXIFTOOL_READ_TIMEOUT_S", "0.4"),
    ])
    .await;
    let old = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    let started = Instant::now();
    let err = pool.read(&dir.file("a.hang"), &TAGS).await.unwrap_err();
    assert_eq!(err, ExifToolError::Timeout);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "bounded by the read timeout"
    );
    assert!(
        !alive(old),
        "the hung process is killed and reaped, not a zombie"
    );
    let fresh = pid(&pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap());
    assert_ne!(old, fresh);
}

#[tokio::test]
async fn a_process_that_dies_is_replaced_and_the_read_is_tried_again_once() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    let retried = pool.read(&dir.file("a.diesonce"), &TAGS).await.unwrap();
    assert_ne!(
        pid(&retried),
        first,
        "the second attempt ran on a fresh process"
    );
    assert!(!alive(first));
}

#[tokio::test]
async fn a_read_that_kills_two_processes_fails_and_the_pool_recovers() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    assert_eq!(
        pool.read(&dir.file("a.die"), &TAGS).await.unwrap_err(),
        ExifToolError::Failed
    );
    let after = pid(&pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap());
    assert_ne!(first, after);
}

#[tokio::test]
async fn an_answer_that_is_not_json_is_an_error_and_the_process_is_replaced() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    assert_eq!(
        pool.read(&dir.file("a.badjson"), &TAGS).await.unwrap_err(),
        ExifToolError::BadOutput
    );
    assert_ne!(
        first,
        pid(&pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap())
    );
}

#[tokio::test]
async fn an_empty_answer_is_its_own_error_and_keeps_the_healthy_process() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    assert_eq!(
        pool.read(&dir.file("a.empty"), &TAGS).await.unwrap_err(),
        ExifToolError::EmptyOutput
    );
    assert_eq!(
        first,
        pid(&pool.read(&dir.file("b.jpg"), &TAGS).await.unwrap())
    );
}

#[tokio::test]
async fn an_answer_over_the_limit_is_a_failure_and_the_process_is_replaced() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    let first = pid(&pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap());
    assert_eq!(
        pool.read(&dir.file("a.huge"), &TAGS).await.unwrap_err(),
        ExifToolError::Failed
    );
    assert!(!alive(first));
    assert!(pool.read(&dir.file("b.jpg"), &TAGS).await.is_ok());
}

#[tokio::test]
async fn standard_error_is_drained_so_a_noisy_process_never_blocks() {
    let dir = Dir::new();
    let pool = pool(&[
        ("OP_EXIFTOOL_POOL_SIZE", "1"),
        ("OP_EXIFTOOL_READ_TIMEOUT_S", "20"),
    ])
    .await;
    let started = Instant::now();
    for i in 0..3 {
        let tags = pool
            .read(&dir.file(&format!("{i}.noisy")), &TAGS)
            .await
            .unwrap();
        assert!(tags.contains_key("Fake:Pid"));
    }
    assert!(started.elapsed() < Duration::from_secs(15));
}

#[tokio::test]
async fn concurrent_reads_spread_over_the_pool() {
    let dir = Dir::new();
    let pool = Arc::new(pool(&[("OP_EXIFTOOL_POOL_SIZE", "3")]).await);
    let mut tasks = Vec::new();
    for i in 0..6 {
        let (pool, path) = (Arc::clone(&pool), dir.file(&format!("{i}.slow")));
        tasks.push(tokio::spawn(async move { pool.read(&path, &TAGS).await }));
    }
    let mut pids = HashSet::new();
    for task in tasks {
        pids.insert(pid(&task.await.unwrap().unwrap()));
    }
    assert_eq!(pids.len(), 3, "three processes shared six reads");
}

#[tokio::test]
async fn a_full_queue_refuses_at_once_and_a_long_wait_is_refused_after_the_bound() {
    let dir = Dir::new();
    let pool = Arc::new(
        pool(&[
            ("OP_EXIFTOOL_POOL_SIZE", "1"),
            ("OP_EXIFTOOL_MAX_WAITING", "1"),
            ("OP_EXIFTOOL_QUEUE_WAIT_S", "0.1"),
        ])
        .await,
    );
    let running = {
        let (pool, path) = (Arc::clone(&pool), dir.file("a.slow"));
        tokio::spawn(async move { pool.read(&path, &TAGS).await })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    let waiting = {
        let (pool, path) = (Arc::clone(&pool), dir.file("b.slow"));
        tokio::spawn(async move { pool.read(&path, &TAGS).await })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;
    let refused = pool.read(&dir.file("c.jpg"), &TAGS).await.unwrap_err();
    assert_eq!(refused, ExifToolError::Busy(Busy::QueueFull));
    assert_eq!(
        waiting.await.unwrap().unwrap_err(),
        ExifToolError::Busy(Busy::WaitedTooLong)
    );
    assert!(
        running.await.unwrap().is_ok(),
        "the read that had a process finished"
    );
}

#[tokio::test]
async fn closing_quits_every_idle_process_and_reaps_it() {
    let dir = Dir::new();
    let pool = Arc::new(pool(&[("OP_EXIFTOOL_POOL_SIZE", "2")]).await);
    let mut tasks = Vec::new();
    for i in 0..2 {
        let (pool, path) = (Arc::clone(&pool), dir.file(&format!("{i}.slow")));
        tasks.push(tokio::spawn(async move { pool.read(&path, &TAGS).await }));
    }
    let mut pids = Vec::new();
    for task in tasks {
        pids.push(pid(&task.await.unwrap().unwrap()));
    }
    pool.close().await;
    for pid in pids {
        assert!(!alive(pid), "process {pid} was reaped");
    }
    assert_eq!(
        pool.read(&dir.file("a.jpg"), &TAGS).await.unwrap_err(),
        ExifToolError::Closed
    );
}

#[tokio::test]
async fn a_read_dropped_half_way_frees_its_slot_and_its_process_is_not_reused() {
    let dir = Dir::new();
    let pool = pool(&[
        ("OP_EXIFTOOL_POOL_SIZE", "1"),
        ("OP_EXIFTOOL_READ_TIMEOUT_S", "20"),
    ])
    .await;
    let abandoned = tokio::time::timeout(
        Duration::from_millis(150),
        pool.read(&dir.file("a.hang"), &TAGS),
    )
    .await;
    assert!(abandoned.is_err(), "the caller gave up");
    assert!(pool.read(&dir.file("b.jpg"), &TAGS).await.is_ok());
}

#[tokio::test]
async fn a_missing_program_and_a_program_that_is_not_exiftool_fail_at_start_naming_the_key() {
    for binary in ["/nonexistent/exiftool", "/bin/true"] {
        let settings = settings(&[("OP_EXIFTOOL_BIN", binary)]);
        let Err(err) = ExifToolPool::start(settings).await else {
            panic!("{binary} must not start");
        };
        assert_eq!(
            err,
            ExifToolError::Start {
                binary: binary.into()
            }
        );
        assert!(err.to_string().contains("OP_EXIFTOOL_BIN"), "{err}");
    }
}

#[tokio::test]
async fn a_capture_read_of_a_stored_file_asks_for_the_capture_tags_and_a_missing_file_is_none() {
    let dir = Dir::new();
    let pool = pool(&[("OP_EXIFTOOL_POOL_SIZE", "1")]).await;
    assert_eq!(
        read_capture_file(&pool, &dir.file("gone.jpg")).await,
        Ok(None)
    );
    let stored = dir.file("a.jpg");
    std::fs::write(&stored, b"x").unwrap();
    let meta = read_capture_file(&pool, &stored).await.unwrap().unwrap();
    assert_eq!(
        meta.exif_status,
        op_media::capture::ExifStatus::Absent,
        "no capture tag in the fake"
    );
    std::fs::write(dir.file("a.empty"), b"x").unwrap();
    assert_eq!(
        read_capture_file(&pool, &dir.file("a.empty")).await,
        Err(ExifToolError::EmptyOutput)
    );
}
