//! What the receiver does with a body while it arrives: field order, per-photo hand-off, back
//! pressure, bounded memory, admission, the disk guard, and the idle limit.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::http::{StatusCode, header};
use bytes::Bytes;
use futures_util::stream::{self, StreamExt};
use op_upload_loadgen::multipart::{FieldOrder, Fields, Photo, boundary, build};
use tokio::sync::Notify;

use crate::common::{
    Pace, answer, chunked, counted, form, landing_files, multipart_request, paced, pattern, rig,
};

fn photos(n: usize, len: usize) -> Vec<Photo> {
    (0..n)
        .map(|i| Photo {
            identifier: format!("set/p{i}.jpg"),
            filename: format!("p{i}.jpg"),
            data: Bytes::from(pattern(u8::try_from(i).unwrap(), len + i * 777)),
        })
        .collect()
}

/// The browser-shaped request of the load generator, as chunks.
fn browser(order: FieldOrder, list: &[Photo], chunk: usize) -> (String, u64, Vec<Bytes>) {
    let fields = Fields {
        source: "upload".into(),
        run_id: "run-7".into(),
    };
    let built = build(order, &boundary(1), &fields, list, chunk);
    let chunks = built
        .frames
        .map(|f| f.unwrap().into_data().unwrap())
        .collect();
    (built.content_type, built.content_length, chunks)
}

#[tokio::test]
async fn photos_arrive_in_order_with_their_bytes_in_both_field_orders() {
    for order in [FieldOrder::Browser, FieldOrder::Legacy] {
        let rig = rig(&[]);
        let list = photos(5, 40_000);
        let (content_type, length, chunks) = browser(order, &list, 16 * 1024);
        let pulled = Arc::new(AtomicU64::new(0));
        rig.sink.count_pulled(&pulled);
        let request = multipart_request(&content_type, Some(length), counted(chunks, &pulled));
        let got = answer(&rig.app, request).await;
        assert_eq!(got.status, StatusCode::OK, "{order:?}: {}", got.body);
        assert_eq!(got.body["status"], "success");
        let seen = rig.sink.seen.lock().unwrap().clone();
        assert_eq!(seen.len(), 5, "{order:?}");
        for (i, s) in seen.iter().enumerate() {
            assert_eq!(s.index, i);
            assert_eq!(s.identifier, list[i].identifier);
            assert_eq!(s.extension, ".jpg");
            assert_eq!(s.filename.as_deref(), Some(list[i].filename.as_str()));
            assert_eq!(s.bytes, list[i].data.to_vec(), "{order:?} photo {i}");
            // the file sits under its content name inside the project's folder
            assert!(
                s.path.starts_with(rig.landing_root.join("7")),
                "{:?}",
                s.path
            );
        }
        let results = got.body["results"].as_array().unwrap();
        let identifiers: Vec<&str> = results
            .iter()
            .map(|r| r["source_identifier"].as_str().unwrap())
            .collect();
        assert_eq!(
            identifiers,
            [
                "set/p0.jpg",
                "set/p1.jpg",
                "set/p2.jpg",
                "set/p3.jpg",
                "set/p4.jpg"
            ]
        );
        let end = rig.sink.ends.lock().unwrap()[0].clone();
        assert_eq!(end.source, "upload");
        let want_run = (order == FieldOrder::Browser).then(|| "run-7".to_string());
        assert_eq!(end.run_id, want_run, "{order:?}");
        if order == FieldOrder::Legacy {
            // identifiers were not known until the end of the body, so nothing was handed over before
            assert!(
                seen.iter().all(|s| s.pulled_at_accept == length),
                "{order:?}"
            );
        }
        // only the photos remain: the upload folder is gone
        let files = landing_files(&rig.landing_root);
        assert_eq!(files.len(), 5, "{files:?}");
    }
}

#[tokio::test]
async fn with_the_fields_first_a_photo_is_handed_on_before_the_body_ends() {
    let rig = rig(&[]);
    let list = photos(5, 40_000);
    let (content_type, length, chunks) = browser(FieldOrder::Browser, &list, 16 * 1024);
    // Before the last frame (the closing boundary) the first four photos are complete. The body
    // waits there until the sink has taken all four, so a receiver that held photos back until
    // the end would never get the body to go on.
    let last = chunks.len() - 1;
    let reached = Arc::new(Notify::new());
    let go = Arc::new(Notify::new());
    let pulled = Arc::new(AtomicU64::new(0));
    let body = paced(
        chunks,
        &pulled,
        vec![(
            last,
            Pace::Pause {
                reached: Arc::clone(&reached),
                go: Arc::clone(&go),
            },
        )],
    );
    let request = multipart_request(&content_type, Some(length), body);
    let app = rig.app.clone();
    let task = tokio::spawn(async move { answer(&app, request).await });
    tokio::time::timeout(Duration::from_secs(20), reached.notified())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(20), rig.sink.done.acquire_many(4))
        .await
        .expect("the sink never got four photos before the body ended")
        .unwrap()
        .forget();
    go.notify_one();
    let got = task.await.unwrap();
    assert_eq!(got.status, StatusCode::OK);
    assert_eq!(rig.sink.seen.lock().unwrap().len(), 5);
}

#[tokio::test]
async fn a_full_sink_stops_the_receiver_from_reading_the_body() {
    let rig = rig(&[
        ("OP_UPLOAD_HANDOFF_QUEUE", "1"),
        ("OP_UPLOAD_PHOTOS_IN_WORK", "64"),
    ]);
    let gate = rig.sink.close_gate();
    let photo_len = 64 * 1024;
    let list: Vec<Photo> = (0..30)
        .map(|i| Photo {
            identifier: format!("p{i}.jpg"),
            filename: format!("p{i}.jpg"),
            data: Bytes::from(pattern(u8::try_from(i).unwrap(), photo_len)),
        })
        .collect();
    let (content_type, length, chunks) = browser(FieldOrder::Browser, &list, 16 * 1024);
    let pulled = Arc::new(AtomicU64::new(0));
    let request = multipart_request(&content_type, Some(length), counted(chunks, &pulled));
    let app = rig.app.clone();
    let task = tokio::spawn(async move { answer(&app, request).await });
    // photo 0 sits inside the blocked accept, photo 1 in the queue of one, photo 2 is landed and
    // waits for room in the queue: the receiver stops there
    tokio::time::timeout(Duration::from_secs(20), rig.sink.started.acquire())
        .await
        .unwrap()
        .unwrap()
        .forget();
    tokio::time::sleep(Duration::from_millis(300)).await;
    let first = pulled.load(Ordering::SeqCst);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let second = pulled.load(Ordering::SeqCst);
    assert_eq!(
        first, second,
        "the receiver kept reading while the sink was full"
    );
    assert!(
        second <= 4 * (photo_len as u64 + 1024) + 4096,
        "{second} bytes read with the sink full; the whole body is {length}"
    );
    assert!(second * 5 < length, "{second} of {length}");
    gate.add_permits(1000);
    let got = task.await.unwrap();
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    assert_eq!(rig.sink.seen.lock().unwrap().len(), 30);
    assert_eq!(pulled.load(Ordering::SeqCst), length);
}

#[tokio::test]
async fn memory_per_photo_does_not_grow_with_its_size() {
    let rig = rig(&[("MAX_FILE_SIZE_MB", "50")]);
    let size = 12 * 1024 * 1024;
    let data = pattern(3, size);
    let (content_type, body) = form(&[("images", Some("big.jpg"), &data)]);
    // pause in the middle of the photo, after its first 8 MiB were read
    let chunks = chunked(&body, 64 * 1024);
    let pause_at = 8 * 1024 * 1024 / (64 * 1024);
    let reached = Arc::new(Notify::new());
    let go = Arc::new(Notify::new());
    let pulled = Arc::new(AtomicU64::new(0));
    let request = multipart_request(
        &content_type,
        Some(body.len() as u64),
        paced(
            chunks,
            &pulled,
            vec![(
                pause_at,
                Pace::Pause {
                    reached: Arc::clone(&reached),
                    go: Arc::clone(&go),
                },
            )],
        ),
    );
    let app = rig.app.clone();
    let task = tokio::spawn(async move { answer(&app, request).await });
    tokio::time::timeout(Duration::from_secs(30), reached.notified())
        .await
        .unwrap();
    let files = landing_files(&rig.landing_root);
    assert_eq!(files.len(), 1, "{files:?}");
    let on_disk = std::fs::metadata(&files[0]).unwrap().len();
    let read = pulled.load(Ordering::SeqCst);
    // what was read but is not on disk is the writer's buffer (256 KiB), the chunk in hand and the
    // form's header, not the photo
    assert!(read >= 8 * 1024 * 1024);
    assert!(
        read - on_disk < 512 * 1024,
        "{} bytes read are not on disk: the part is buffered in memory",
        read - on_disk
    );
    go.notify_one();
    let got = task.await.unwrap();
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    assert_eq!(rig.sink.seen.lock().unwrap()[0].bytes, data);
}

#[tokio::test(start_paused = true)]
async fn a_full_admission_answers_503_with_retry_after_and_keeps_nothing() {
    let rig = rig(&[
        ("OP_UPLOAD_PHOTOS_IN_WORK", "1"),
        ("OP_UPLOAD_ADMISSION_WAIT_S", "1"),
    ]);
    let before = rig.state.admission.available();
    let held = rig.state.admission.admit_photo().await.unwrap();
    let (content_type, body) = form(&[("images", Some("a.jpg"), b"abc")]);
    let request = multipart_request(&content_type, None, Body::from(body.clone()));
    let got = answer(&rig.app, request).await;
    assert_eq!(got.status, StatusCode::SERVICE_UNAVAILABLE, "{}", got.body);
    assert_eq!(got.headers[header::RETRY_AFTER], "5");
    assert_eq!(got.body["code"], "upload_busy");
    assert!(got.body["detail"].as_str().unwrap().contains("busy"));
    assert!(landing_files(&rig.landing_root).is_empty());
    drop(held);
    assert_eq!(rig.state.admission.available(), before);
    // and with the bound free again the same request goes through
    let request = multipart_request(&content_type, None, Body::from(body));
    assert_eq!(answer(&rig.app, request).await.status, StatusCode::OK);
}

#[tokio::test(start_paused = true)]
async fn a_full_byte_bound_answers_503_and_gives_the_photo_permit_back() {
    let rig = rig(&[
        ("MAX_FILE_SIZE_MB", "1"),
        ("OP_UPLOAD_BYTES_IN_FLIGHT", "1048576"),
        ("OP_UPLOAD_ADMISSION_WAIT_S", "1"),
    ]);
    let before = rig.state.admission.available();
    let mut held = rig.state.admission.admit_photo().await.unwrap();
    rig.state
        .admission
        .reserve_bytes(&mut held, 1024 * 1024)
        .await
        .unwrap();
    let data = pattern(5, 200_000);
    let (content_type, body) = form(&[("images", Some("a.jpg"), &data)]);
    let total = body.len() as u64;
    let pulled = Arc::new(AtomicU64::new(0));
    let request = multipart_request(
        &content_type,
        Some(total),
        counted(chunked(&body, 16 * 1024), &pulled),
    );
    let got = answer(&rig.app, request).await;
    assert_eq!(got.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(got.headers[header::RETRY_AFTER], "5");
    // the bytes are reserved when the photo's first chunk arrives, so the body is not read on
    assert!(
        pulled.load(Ordering::SeqCst) < 64 * 1024,
        "the body was read past the first chunk"
    );
    drop(held);
    assert_eq!(rig.state.admission.available(), before);
    assert!(landing_files(&rig.landing_root).is_empty());
}

#[tokio::test]
async fn a_full_disk_refuses_before_the_body_is_read() {
    let rig = rig(&[("OP_DISK_LOW_PCT", "100"), ("OP_DISK_CRITICAL_PCT", "100")]);
    let (content_type, body) = form(&[("images", Some("a.jpg"), b"abc")]);
    let pulled = Arc::new(AtomicU64::new(0));
    let request = multipart_request(
        &content_type,
        Some(body.len() as u64),
        counted(vec![Bytes::from(body)], &pulled),
    );
    let got = answer(&rig.app, request).await;
    assert_eq!(got.status, StatusCode::INSUFFICIENT_STORAGE);
    assert_eq!(got.body["detail"]["error"], "insufficient_storage");
    assert_eq!(got.body["detail"]["volume"], "data");
    let message = got.body["detail"]["message"].as_str().unwrap();
    assert!(message.starts_with("the data volume has "), "{message}");
    assert!(
        message.ends_with("% free; free space before starting new writes"),
        "{message}"
    );
    assert_eq!(pulled.load(Ordering::SeqCst), 0, "the body was read");
    assert!(landing_files(&rig.landing_root).is_empty());
}

#[tokio::test(start_paused = true)]
async fn a_slow_body_is_not_cut_off_while_bytes_still_arrive_and_an_idle_one_is() {
    // each chunk arrives 600 ms after the one before: the whole body takes far longer than the
    // 1 s idle limit, and no single wait does
    let rig = rig(&[("OP_UPLOAD_BODY_IDLE_TIMEOUT_S", "1")]);
    let data = pattern(9, 8000);
    let (content_type, body) = form(&[("images", Some("slow.jpg"), &data)]);
    let chunks = chunked(&body, 1000);
    let count = chunks.len();
    let slow = Body::from_stream(
        stream::iter(chunks)
            .then(|c| async move {
                tokio::time::sleep(Duration::from_millis(600)).await;
                c
            })
            .map(Ok::<_, std::io::Error>),
    );
    let got = answer(&rig.app, multipart_request(&content_type, None, slow)).await;
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    assert!(
        count as u64 * 600 > 1000,
        "the body must outlast the idle limit"
    );
    assert_eq!(rig.sink.seen.lock().unwrap()[0].bytes, data);

    // now the same body goes silent for 2 s after its fourth chunk
    let rig = self::rig(&[("OP_UPLOAD_BODY_IDLE_TIMEOUT_S", "1")]);
    let pulled = Arc::new(AtomicU64::new(0));
    let stalled = paced(
        chunked(&body, 1000),
        &pulled,
        vec![(4, Pace::Sleep(Duration::from_secs(2)))],
    );
    let got = answer(&rig.app, multipart_request(&content_type, None, stalled)).await;
    assert_eq!(got.status, StatusCode::REQUEST_TIMEOUT, "{}", got.body);
    assert!(
        got.body["detail"]
            .as_str()
            .unwrap()
            .contains("no bytes arrived for 1 s")
    );
    assert!(landing_files(&rig.landing_root).is_empty());
    assert_eq!(rig.sink.accept_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test(start_paused = true)]
async fn a_pause_for_a_full_sink_is_not_counted_as_a_silent_body() {
    let rig = rig(&[
        ("OP_UPLOAD_BODY_IDLE_TIMEOUT_S", "1"),
        ("OP_UPLOAD_HANDOFF_QUEUE", "1"),
    ]);
    let gate = rig.sink.close_gate();
    let ids: Vec<String> = (0..6).map(|i| format!("p{i}.jpg")).collect();
    let datas: Vec<Vec<u8>> = (0..6).map(|i| pattern(i, 5000)).collect();
    let paths = serde_json::to_string(&ids).unwrap();
    let mut parts: Vec<(&str, Option<&str>, &[u8])> = vec![("image_paths", None, paths.as_bytes())];
    parts.extend(
        ids.iter()
            .zip(&datas)
            .map(|(id, d)| ("images", Some(id.as_str()), d.as_slice())),
    );
    let (content_type, body) = form(&parts);
    let request = multipart_request(&content_type, None, Body::from(body));
    let app = rig.app.clone();
    let task = tokio::spawn(async move { answer(&app, request).await });
    // the sink stays shut for ten times the idle limit while the receiver waits to hand over
    tokio::time::sleep(Duration::from_secs(10)).await;
    gate.add_permits(100);
    let got = task.await.unwrap();
    assert_eq!(got.status, StatusCode::OK, "{}", got.body);
    assert_eq!(rig.sink.seen.lock().unwrap().len(), 6);
}
