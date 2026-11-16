//! The load generator against an in-process axum server: the body it streams parses with
//! `axum::extract::Multipart` into the same fields and bytes, the closed loop keeps exactly N
//! requests in flight on N connections, and every way a request can fail is counted under its key.

#![expect(
    clippy::unwrap_used,
    reason = "the in-process sink's handlers and helpers are test code, outside #[test] functions"
)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::serve::{Listener, ListenerExt};
use bytes::Bytes;
use frame_engine_loadgen::exchange::Probe;
use op_upload_loadgen::closed_loop::{self, LoopConfig};
use op_upload_loadgen::commands::upload_uri;
use op_upload_loadgen::multipart::{FieldOrder, Fields, Photo};
use op_upload_loadgen::pack::pack;
use op_upload_loadgen::record::Outcome;
use op_upload_loadgen::send::{Sender, pooled_client};
use tokio::net::TcpListener;

#[derive(Debug, Clone, PartialEq)]
struct Field {
    name: String,
    filename: Option<String>,
    content_type: Option<String>,
    data: Vec<u8>,
}

#[derive(Debug)]
struct Seen {
    content_length: Option<u64>,
    chunked: bool,
    fields: Vec<Field>,
}

#[derive(Clone)]
struct Sink {
    seen: Arc<Mutex<Vec<Seen>>>,
    inflight: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
    connections: Arc<AtomicUsize>,
    delay: Duration,
    status: StatusCode,
}

impl Sink {
    fn new(delay: Duration, status: StatusCode) -> Sink {
        Sink {
            seen: Arc::default(),
            inflight: Arc::default(),
            peak: Arc::default(),
            connections: Arc::default(),
            delay,
            status,
        }
    }
}

async fn upload(State(sink): State<Sink>, headers: HeaderMap, mut mp: Multipart) -> Response {
    let now = sink.inflight.fetch_add(1, Ordering::SeqCst) + 1;
    sink.peak.fetch_max(now, Ordering::SeqCst);
    let mut fields = Vec::new();
    while let Some(field) = mp.next_field().await.unwrap() {
        fields.push(Field {
            name: field.name().unwrap().to_string(),
            filename: field.file_name().map(str::to_string),
            content_type: field.content_type().map(str::to_string),
            data: field.bytes().await.unwrap().to_vec(),
        });
    }
    tokio::time::sleep(sink.delay).await;
    let images = fields.iter().filter(|f| f.name == "images").count();
    sink.seen.lock().unwrap().push(Seen {
        content_length: headers
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok()?.parse().ok()),
        chunked: headers.contains_key(header::TRANSFER_ENCODING),
        fields,
    });
    sink.inflight.fetch_sub(1, Ordering::SeqCst);
    if !sink.status.is_success() {
        return sink.status.into_response();
    }
    let results = vec![r#"{"status":"success"}"#; images].join(",");
    (
        [(header::CONTENT_TYPE, "application/json")],
        format!(r#"{{"status":"success","results":[{results}]}}"#),
    )
        .into_response()
}

async fn serve(sink: Sink) -> SocketAddr {
    let counter = Arc::clone(&sink.connections);
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap()
        .tap_io(move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
        });
    let addr = listener.local_addr().unwrap();
    let app = Router::new()
        .route("/curation/projects/{project}/ingest/upload", post(upload))
        .layer(DefaultBodyLimit::disable())
        .with_state(sink);
    tokio::spawn(async move { axum::serve(listener, app).await });
    addr
}

fn photos(n: usize) -> Vec<Photo> {
    let mut state = 0x1234_5678_9abc_def0u64;
    (0..n)
        .map(|i| {
            let len = 5000 + i * 777;
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state.to_le_bytes()[0]
                })
                .collect();
            Photo {
                identifier: format!("set/p{i}.jpg"),
                filename: format!("p{i}.jpg"),
                data: Bytes::from(data),
            }
        })
        .collect()
}

fn sender(addr: SocketAddr, photos: Vec<Photo>, order: FieldOrder, max_images: usize) -> Sender {
    let sizes: Vec<u64> = photos.iter().map(|p| p.data.len() as u64).collect();
    Sender {
        client: pooled_client(4),
        uri: upload_uri(&format!("http://{addr}"), "demo").unwrap(),
        timeout: Duration::from_secs(10),
        order,
        fields: Fields {
            source: "upload".into(),
            run_id: "run-7".into(),
        },
        chunk: 1000,
        packs: pack(&sizes, max_images, 512 * 1024 * 1024).unwrap(),
        photos: Arc::new(photos),
        t0: Instant::now(),
    }
}

fn text(f: &Field) -> String {
    String::from_utf8(f.data.clone()).unwrap()
}

#[tokio::test]
async fn the_streamed_body_parses_into_the_same_fields_and_bytes() {
    for order in [FieldOrder::Browser, FieldOrder::Legacy] {
        let sink = Sink::new(Duration::ZERO, StatusCode::OK);
        let addr = serve(sink.clone()).await;
        let list = photos(7);
        let s = sender(addr, list.clone(), order, 3);
        assert_eq!(s.packs.len(), 3);
        for seq in 0..3 {
            let record = s.send(seq).await;
            assert!(
                matches!(record.outcome, Outcome::Answered { read: true, .. }),
                "{record:?}"
            );
            assert_eq!(record.parts_taken_ms.len(), record.photos);
            assert!(record.parts_taken_ms.windows(2).all(|w| w[0] <= w[1]));
            assert!(
                record
                    .parts_taken_ms
                    .iter()
                    .all(|&t| t <= record.response_ms)
            );
        }
        let seen = sink.seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        let mut sent_photos = Vec::new();
        for (req, range) in seen.iter().zip(&s.packs) {
            assert!(
                !req.chunked,
                "the body length is known: no chunked encoding"
            );
            let names: Vec<&str> = req.fields.iter().map(|f| f.name.as_str()).collect();
            let images = range.len();
            let mut expected: Vec<&str> = match order {
                FieldOrder::Browser => vec!["source", "image_paths", "run_id"],
                FieldOrder::Legacy => vec![],
            };
            if order == FieldOrder::Legacy {
                expected.extend(std::iter::repeat_n("images", images));
                expected.extend(["image_paths", "source"]);
            } else {
                expected.extend(std::iter::repeat_n("images", images));
            }
            assert_eq!(names, expected);
            let by_name = |n: &str| req.fields.iter().find(|f| f.name == n).map(text);
            assert_eq!(by_name("source").as_deref(), Some("upload"));
            assert_eq!(
                by_name("run_id").as_deref(),
                (order == FieldOrder::Browser).then_some("run-7")
            );
            let identifiers: Vec<String> =
                serde_json::from_str(&by_name("image_paths").unwrap()).unwrap();
            let expected_ids: Vec<&str> = list[range.clone()]
                .iter()
                .map(|p| p.identifier.as_str())
                .collect();
            assert_eq!(identifiers, expected_ids);
            let body_len: usize = req.fields.iter().map(|f| f.data.len()).sum();
            assert!(req.content_length.unwrap() as usize > body_len);
            for f in req.fields.iter().filter(|f| f.name == "images") {
                assert_eq!(f.content_type.as_deref(), Some("image/jpeg"));
                sent_photos.push((f.filename.clone().unwrap(), f.data.clone()));
            }
        }
        let expected: Vec<(String, Vec<u8>)> = list
            .iter()
            .map(|p| (p.filename.clone(), p.data.to_vec()))
            .collect();
        assert_eq!(sent_photos, expected);
    }
}

struct NoProbe;

impl Probe for NoProbe {
    async fn window_open(&mut self) {}
    async fn window_close(&mut self) {}
}

#[tokio::test]
async fn the_loop_keeps_n_requests_in_flight_on_n_connections() {
    let sink = Sink::new(Duration::from_millis(40), StatusCode::OK);
    let addr = serve(sink.clone()).await;
    let s = Arc::new(sender(addr, photos(6), FieldOrder::Browser, 2));
    let config = LoopConfig {
        in_flight: 3,
        warmup: Duration::ZERO,
        window: Duration::from_millis(400),
    };
    let out = closed_loop::run(Arc::clone(&s), config, &mut NoProbe).await;
    assert_eq!(sink.peak.load(Ordering::SeqCst), 3);
    assert_eq!(
        sink.connections.load(Ordering::SeqCst),
        3,
        "the pool reuses its connections"
    );
    assert!(out.records.len() >= 9, "{} requests", out.records.len());
    assert!(
        out.records
            .iter()
            .all(|r| matches!(r.outcome, Outcome::Answered { .. }))
    );
    assert_eq!(out.requests_started as usize, out.records.len());
}

#[tokio::test]
async fn refusals_timeouts_and_refused_connections_are_counted_by_key() {
    for (code, key) in [(413, "http:413"), (503, "http:503")] {
        let addr = serve(Sink::new(
            Duration::ZERO,
            StatusCode::from_u16(code).unwrap(),
        ))
        .await;
        let r = sender(addr, photos(2), FieldOrder::Browser, 2)
            .send(0)
            .await;
        assert_eq!(r.outcome, Outcome::Failed { key: key.into() });
    }
    let addr = serve(Sink::new(Duration::from_millis(600), StatusCode::OK)).await;
    let mut slow = sender(addr, photos(2), FieldOrder::Browser, 2);
    slow.timeout = Duration::from_millis(150);
    let r = slow.send(0).await;
    assert_eq!(
        r.outcome,
        Outcome::Failed {
            key: "client:timeout".into()
        }
    );
    assert_eq!(
        r.photos, 2,
        "a failed request still says how many photos it carried"
    );

    let closed = {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap()
    };
    let r = sender(closed, photos(2), FieldOrder::Browser, 2)
        .send(0)
        .await;
    assert_eq!(
        r.outcome,
        Outcome::Failed {
            key: "client:connect".into()
        }
    );
}

#[tokio::test]
async fn a_run_writes_one_row_and_the_stamps_file() {
    use clap::Parser;
    use op_upload_loadgen::commands::{Args, run};

    let dir = std::env::temp_dir().join(format!("op-upload-loadgen-run-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let list: Vec<Photo> = photos(5);
    let mut text = String::new();
    for p in &list {
        let path = dir.join(&p.filename);
        std::fs::write(&path, &p.data).unwrap();
        text.push_str(&format!("{}\n", path.display()));
    }
    std::fs::write(dir.join("list.txt"), &text).unwrap();
    let sink = Sink::new(Duration::from_millis(10), StatusCode::OK);
    let addr = serve(sink.clone()).await;
    let (out, stamps) = (dir.join("rows.jsonl"), dir.join("stamps.jsonl"));
    let argv: Vec<String> = [
        "op-upload-loadgen",
        "--base-url",
        &format!("http://{addr}"),
        "--project",
        "demo",
        "--list",
        dir.join("list.txt").to_str().unwrap(),
        "--in-flight",
        "2",
        "--warmup-s",
        "0",
        "--window-s",
        "0.4",
        "--max-images",
        "2",
        "--label",
        "t",
        "--out",
        out.to_str().unwrap(),
        "--stamps-out",
        stamps.to_str().unwrap(),
        "--server-note",
        "in-process axum",
    ]
    .map(String::from)
    .to_vec();
    let code = run(Args::parse_from(&argv), argv.clone()).await.unwrap();
    assert_eq!(
        format!("{code:?}"),
        format!("{:?}", std::process::ExitCode::SUCCESS)
    );
    let rows = std::fs::read_to_string(&out).unwrap();
    assert_eq!(rows.lines().count(), 1);
    let row: serde_json::Value = serde_json::from_str(rows.trim()).unwrap();
    assert_eq!(row["schema"], "op-upload-loadgen/1");
    assert_eq!(row["command"].as_array().unwrap().len(), argv.len());
    assert_eq!(row["list"]["photos"], 5);
    assert_eq!(row["list"]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(row["upload"]["packs_per_pass"], 3);
    assert_eq!(row["upload"]["http_version"], "HTTP/1.1");
    assert_eq!(row["server_note"], "in-process axum");
    assert!(row["photos_success"].as_u64().unwrap() > 0);
    assert!(row["photos_per_s"].as_f64().unwrap() > 0.0);
    assert!(
        row["stamps"]["bounds"][0]
            .as_str()
            .unwrap()
            .contains("client-side bound")
    );
    assert!(
        row["list_cycles_sent"].as_f64().unwrap() > 1.0,
        "a 0.4 s run repeats a 5-photo list"
    );
    let lines: Vec<serde_json::Value> = std::fs::read_to_string(&stamps)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(
        lines.len() as u64,
        row["requests_started"].as_u64().unwrap()
    );
    assert!(
        lines
            .iter()
            .all(|l| l["parts_taken_ms"].as_array().unwrap().len() as u64
                == l["photos"].as_u64().unwrap())
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
