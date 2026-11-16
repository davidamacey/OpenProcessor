//! What no request leaves behind: no landing file and no permit, whichever way the request ends.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::time::Duration;

use axum::body::Body;
use axum::http::StatusCode;
use bytes::Bytes;
use op_ingest::upload::SinkError;
use tokio::sync::Notify;

use crate::common::{
    Pace, Rig, answer, chunked, counted, form, landing_files, multipart_request, paced, pattern,
    rig,
};

/// A form with `image_paths` first and one photo per entry of `sizes`.
fn photos_form(names: &[&str], sizes: &[usize]) -> (String, Vec<u8>) {
    let paths = serde_json::to_string(names).unwrap();
    let datas: Vec<Vec<u8>> = sizes
        .iter()
        .enumerate()
        .map(|(i, n)| pattern(u8::try_from(i).unwrap(), *n))
        .collect();
    let mut parts: Vec<(&str, Option<&str>, &[u8])> = vec![("image_paths", None, paths.as_bytes())];
    parts.extend(
        names
            .iter()
            .zip(&datas)
            .map(|(n, d)| ("images", Some(*n), d.as_slice())),
    );
    form(&parts)
}

fn assert_clean(rig: &Rig, before: op_ingest::admission::Available) {
    assert_eq!(
        landing_files(&rig.landing_root),
        Vec::<std::path::PathBuf>::new()
    );
    assert_eq!(
        rig.state.admission.available(),
        before,
        "a permit was not given back"
    );
}

#[tokio::test]
async fn a_client_that_disconnects_in_the_middle_of_a_photo_leaves_nothing() {
    let rig = rig(&[]);
    let before = rig.state.admission.available();
    let (content_type, body) = photos_form(&["a.jpg", "b.jpg"], &[40_000, 40_000]);
    let chunks = chunked(&body, 8192);
    // the connection drops during the second photo
    let drop_at = chunks.len() - 3;
    let pulled = Arc::new(AtomicU64::new(0));
    let request = multipart_request(
        &content_type,
        Some(body.len() as u64),
        paced(chunks, &pulled, vec![(drop_at, Pace::Fail)]),
    );
    let got = answer(&rig.app, request).await;
    assert_eq!(got.status, StatusCode::BAD_REQUEST, "{}", got.body);
    assert_clean(&rig, before);
    // the first photo had been handed over, so the sink is told to drop it
    assert_eq!(rig.sink.aborts.lock().unwrap().len(), 1);
    assert!(rig.sink.ends.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_request_that_is_dropped_by_the_server_cleans_up_through_the_tracker() {
    let rig = rig(&[]);
    let before = rig.state.admission.available();
    let (content_type, body) = photos_form(&["a.jpg", "b.jpg"], &[40_000, 40_000]);
    let chunks = chunked(&body, 8192);
    let stop_at = chunks.len() - 3;
    let reached = Arc::new(Notify::new());
    let pulled = Arc::new(AtomicU64::new(0));
    let request = multipart_request(
        &content_type,
        Some(body.len() as u64),
        paced(
            chunks,
            &pulled,
            vec![(
                stop_at,
                Pace::Pause {
                    reached: Arc::clone(&reached),
                    go: Arc::new(Notify::new()),
                },
            )],
        ),
    );
    let app = rig.app.clone();
    let task = tokio::spawn(async move { answer(&app, request).await });
    tokio::time::timeout(Duration::from_secs(20), reached.notified())
        .await
        .unwrap();
    // the first photo is landed and with the sink, the second is half written
    assert!(!landing_files(&rig.landing_root).is_empty());
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    rig.state.wait_for_cleanup().await;
    assert_clean(&rig, before);
    assert_eq!(rig.sink.aborts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_later_part_that_is_refused_removes_the_earlier_photos_of_the_request() {
    // the third photo is over the one MB file cap: 413, and photos 0 and 1 were already handed over
    let rig = rig(&[("MAX_FILE_SIZE_MB", "1")]);
    let before = rig.state.admission.available();
    let (content_type, body) =
        photos_form(&["a.jpg", "b.jpg", "c.jpg"], &[1000, 2000, 1024 * 1024 + 1]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(got.body["detail"], "File too large. Maximum: 1MB");
    assert_clean(&rig, before);
    assert_eq!(rig.sink.aborts.lock().unwrap().len(), 1);
    assert!(rig.sink.ends.lock().unwrap().is_empty());
}

#[tokio::test]
async fn too_many_photos_refuse_the_request_and_remove_every_photo() {
    let rig = rig(&[("OP_UPLOAD_MAX_IMAGES_PER_REQUEST", "2")]);
    let before = rig.state.admission.available();
    let (content_type, body) =
        photos_form(&["a.jpg", "b.jpg", "c.jpg", "d.jpg"], &[500, 500, 500, 500]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::PAYLOAD_TOO_LARGE);
    // image_paths came first and lists four, so the count is checked against the four sent
    assert_eq!(
        got.body["detail"],
        "4 images exceeds the per-request limit of 2"
    );
    assert_clean(&rig, before);
    assert_eq!(rig.sink.aborts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_panic_in_the_sink_is_a_500_and_leaves_no_file_and_no_permit() {
    let rig = rig(&[]);
    let before = rig.state.admission.available();
    *rig.sink.panic_on_accept.lock().unwrap() = Some(1);
    let (content_type, body) = photos_form(&["a.jpg", "b.jpg", "c.jpg"], &[3000, 3000, 3000]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(
        got.status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        got.body
    );
    assert_eq!(got.body["detail"], "Internal server error");
    assert_clean(&rig, before);
    assert_eq!(rig.sink.aborts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_sink_that_is_unavailable_answers_503_with_retry_after_and_keeps_nothing() {
    let rig = rig(&[]);
    let before = rig.state.admission.available();
    *rig.sink.refuse_accept.lock().unwrap() = Some(SinkError::Unavailable {
        detail: "The image engine is temporarily unavailable; retry shortly".into(),
        code: "engine_unavailable".into(),
    });
    let (content_type, body) = photos_form(&["a.jpg", "b.jpg"], &[3000, 3000]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::SERVICE_UNAVAILABLE, "{}", got.body);
    assert_eq!(got.headers["retry-after"], "5");
    assert_eq!(got.body["code"], "engine_unavailable");
    assert_clean(&rig, before);
}

#[tokio::test]
async fn a_sink_that_cannot_commit_refuses_the_request_and_is_told_to_abort() {
    let rig = rig(&[]);
    let before = rig.state.admission.available();
    *rig.sink.refuse_finish.lock().unwrap() =
        Some(SinkError::Failed("the database is gone".into()));
    let (content_type, body) = photos_form(&["a.jpg"], &[3000]);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, Body::from(body)),
    )
    .await;
    assert_eq!(got.status, StatusCode::INTERNAL_SERVER_ERROR);
    // the cause is logged, never sent
    assert!(!got.body.to_string().contains("database"));
    assert_clean(&rig, before);
    assert_eq!(rig.sink.aborts.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn a_body_that_streams_past_the_ceiling_is_cut_with_python_s_413() {
    // one MB per file, 512 MiB accepted bytes in the default, so lower the total to make the
    // ceiling 2 MiB: three 700 KB photos are 2.1 MB with no declared length to refuse early
    let rig = rig(&[
        ("MAX_FILE_SIZE_MB", "1"),
        ("OP_UPLOAD_MAX_BYTES_PER_REQUEST", "1048576"),
    ]);
    let before = rig.state.admission.available();
    let (content_type, body) =
        photos_form(&["a.jpg", "b.jpg", "c.jpg"], &[700_000, 700_000, 700_000]);
    let pulled = Arc::new(AtomicU64::new(0));
    let chunks: Vec<Bytes> = chunked(&body, 32 * 1024);
    let got = answer(
        &rig.app,
        multipart_request(&content_type, None, counted(chunks, &pulled)),
    )
    .await;
    assert_eq!(got.status, StatusCode::PAYLOAD_TOO_LARGE, "{}", got.body);
    assert_eq!(got.body["detail"], "File too large. Maximum: 1MB");
    assert_eq!(got.body["max_file_size_mb"], 1);
    assert!(
        pulled.load(std::sync::atomic::Ordering::SeqCst) <= 2 * 1024 * 1024 + 64 * 1024,
        "the body was read past the ceiling"
    );
    assert_clean(&rig, before);
}

#[tokio::test]
async fn a_declared_length_over_the_ceiling_is_refused_without_reading_the_body() {
    let rig = rig(&[
        ("MAX_FILE_SIZE_MB", "1"),
        ("OP_UPLOAD_MAX_BYTES_PER_REQUEST", "1048576"),
    ]);
    let (content_type, body) = form(&[("images", Some("a.jpg"), b"abc")]);
    let pulled = Arc::new(AtomicU64::new(0));
    let request = multipart_request(
        &content_type,
        Some(2 * 1024 * 1024 + 1),
        counted(vec![Bytes::from(body)], &pulled),
    );
    let got = answer(&rig.app, request).await;
    assert_eq!(got.status, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(pulled.load(std::sync::atomic::Ordering::SeqCst), 0);
}
