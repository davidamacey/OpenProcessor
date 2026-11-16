//! A rig for the receiver's request tests: a temporary landing root, a fake sink, the route with
//! its layers, and helpers that send a body and read the answer.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode, header};
use bytes::Bytes;
use futures_util::stream::{self, StreamExt};
use op_config::Env;
use op_ingest::settings::UploadSettings;
use op_ingest::upload::{
    BatchSummary, ImageStatus, IngestImageResponse, PartSink, ReceivedPart, SinkBatch, SinkError,
    UploadEnd, UploadState, router,
};
use op_storage::keys::ProjectId;
use op_storage::landing::Landing;
use op_storage::local::LocalRoots;
use serde_json::Value;
use tempfile::TempDir;
use tokio::sync::{Notify, Semaphore};
use tower::ServiceExt;
use uuid::Uuid;

pub const PROJECT: ProjectId = ProjectId(7);
pub const URI: &str = "/curation/projects/demo/ingest/upload";

/// What the fake sink saw of one photo.
#[derive(Debug, Clone)]
pub struct Seen {
    pub index: usize,
    pub identifier: String,
    pub extension: String,
    pub filename: Option<String>,
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    /// Body bytes the client's stream had been asked for when the photo arrived.
    pub pulled_at_accept: u64,
}

pub struct FakeSink {
    pub seen: Mutex<Vec<Seen>>,
    pub ends: Mutex<Vec<UploadEnd>>,
    pub aborts: Mutex<Vec<Uuid>>,
    /// When set, `accept` waits for a permit of this semaphore before it returns.
    pub gate: Mutex<Option<Arc<Semaphore>>>,
    pub panic_on_accept: Mutex<Option<usize>>,
    pub refuse_accept: Mutex<Option<SinkError>>,
    pub refuse_finish: Mutex<Option<SinkError>>,
    pub pulled: Mutex<Option<Arc<AtomicU64>>>,
    pub accept_calls: AtomicU64,
    /// One permit per `accept` entered, before it waits for the gate.
    pub started: Semaphore,
    /// One permit per photo `accept` recorded.
    pub done: Semaphore,
}

impl FakeSink {
    pub fn new() -> Self {
        Self {
            seen: Mutex::default(),
            ends: Mutex::default(),
            aborts: Mutex::default(),
            gate: Mutex::default(),
            panic_on_accept: Mutex::default(),
            refuse_accept: Mutex::default(),
            refuse_finish: Mutex::default(),
            pulled: Mutex::default(),
            accept_calls: AtomicU64::new(0),
            started: Semaphore::new(0),
            done: Semaphore::new(0),
        }
    }

    pub fn close_gate(&self) -> Arc<Semaphore> {
        let gate = Arc::new(Semaphore::new(0));
        *self.gate.lock().unwrap() = Some(Arc::clone(&gate));
        gate
    }

    pub fn count_pulled(&self, pulled: &Arc<AtomicU64>) {
        *self.pulled.lock().unwrap() = Some(Arc::clone(pulled));
    }
}

impl PartSink for FakeSink {
    async fn accept(&self, part: ReceivedPart) -> Result<(), SinkError> {
        self.accept_calls.fetch_add(1, Ordering::SeqCst);
        self.started.add_permits(1);
        if *self.panic_on_accept.lock().unwrap() == Some(part.index) {
            panic!("the fake sink panics on photo {}", part.index);
        }
        if let Some(refusal) = self.refuse_accept.lock().unwrap().clone() {
            return Err(refusal);
        }
        let gate = self.gate.lock().unwrap().clone();
        if let Some(gate) = gate {
            gate.acquire().await.unwrap().forget();
        }
        let bytes = tokio::fs::read(&part.photo.path).await.unwrap();
        let pulled = self
            .pulled
            .lock()
            .unwrap()
            .as_ref()
            .map_or(0, |p| p.load(Ordering::SeqCst));
        self.seen.lock().unwrap().push(Seen {
            index: part.index,
            identifier: part.identifier.clone(),
            extension: part.extension.clone(),
            filename: part.filename.clone(),
            path: part.photo.path.clone(),
            bytes,
            pulled_at_accept: pulled,
        });
        drop(part.permit);
        self.done.add_permits(1);
        Ok(())
    }

    async fn finish(&self, end: UploadEnd) -> Result<SinkBatch, SinkError> {
        if let Some(refusal) = self.refuse_finish.lock().unwrap().clone() {
            return Err(refusal);
        }
        self.ends.lock().unwrap().push(end);
        let seen = self.seen.lock().unwrap().clone();
        let results: Vec<IngestImageResponse> = seen
            .iter()
            .map(|s| IngestImageResponse {
                status: ImageStatus::Success,
                image_id: format!("id-{}", s.index),
                image_path: s.path.display().to_string(),
                imohash: String::new(),
                n_crops: 1,
                n_regions: 0,
                n_embedded: 1,
                n_not_embedded: 0,
                n_embed_failed: 0,
                n_filtered: 0,
                error: None,
                error_kind: None,
                source_identifier: Some(s.identifier.clone()),
                secondary_detector_error: None,
            })
            .collect();
        let summary = BatchSummary {
            successful: results.len() as u64,
            crops_indexed: results.len() as u64,
            n_embedded: results.len() as u64,
            ..BatchSummary::default()
        };
        Ok(SinkBatch { results, summary })
    }

    async fn abort(&self, upload_id: Uuid) {
        self.aborts.lock().unwrap().push(upload_id);
    }
}

pub struct Rig {
    /// Keeps the temporary root alive for the rig's lifetime.
    _dir: TempDir,
    pub landing_root: PathBuf,
    pub sink: Arc<FakeSink>,
    pub state: Arc<UploadState<FakeSink>>,
    pub app: Router,
}

/// A rig whose settings are the defaults with `env` applied on top.
pub fn rig(env: &[(&str, &str)]) -> Rig {
    let dir = tempfile::tempdir().unwrap();
    let base = std::fs::canonicalize(dir.path()).unwrap();
    std::fs::create_dir(base.join("landing")).unwrap();
    std::fs::create_dir(base.join("sources")).unwrap();
    let roots = LocalRoots::new(&base.join("landing"), &base.join("sources")).unwrap();
    let settings = UploadSettings::from_env(&Env::from_pairs(
        env.iter().map(|(k, v)| (k.to_string(), v.to_string())),
    ))
    .unwrap();
    let landing = Landing::new(&roots);
    let sink = Arc::new(FakeSink::new());
    let state = UploadState::new(settings, landing, Arc::clone(&sink));
    let app = router(Arc::clone(&state)).layer(axum::Extension(PROJECT));
    Rig {
        landing_root: roots.landing,
        _dir: dir,
        sink,
        state,
        app,
    }
}

/// Every file under the landing root.
pub fn landing_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The answer of one request.
#[derive(Debug)]
pub struct Answer {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Value,
}

pub fn multipart_request(content_type: &str, length: Option<u64>, body: Body) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(URI)
        .header(header::CONTENT_TYPE, content_type);
    if let Some(n) = length {
        builder = builder.header(header::CONTENT_LENGTH, n);
    }
    builder.body(body).unwrap()
}

pub async fn answer(app: &Router, request: Request<Body>) -> Answer {
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = axum::body::to_bytes(response.into_body(), 16 * 1024 * 1024)
        .await
        .unwrap();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    Answer {
        status,
        headers,
        body,
    }
}

/// A body made of `chunks`, counting the bytes the server has asked for in `pulled`.
pub fn counted(chunks: Vec<Bytes>, pulled: &Arc<AtomicU64>) -> Body {
    let pulled = Arc::clone(pulled);
    Body::from_stream(
        stream::iter(chunks)
            .inspect(move |c| {
                pulled.fetch_add(c.len() as u64, Ordering::SeqCst);
            })
            .map(Ok::<_, std::io::Error>),
    )
}

/// Split a whole body into chunks of at most `size` bytes.
pub fn chunked(body: &[u8], size: usize) -> Vec<Bytes> {
    body.chunks(size.max(1))
        .map(Bytes::copy_from_slice)
        .collect()
}

/// A multipart body of `parts` (name, filename, data) in order, with a fixed boundary.
pub fn form(parts: &[(&str, Option<&str>, &[u8])]) -> (String, Vec<u8>) {
    let boundary = "----R1uBoundary";
    let mut out = Vec::new();
    for (name, filename, data) in parts {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        let mut head = format!("Content-Disposition: form-data; name=\"{name}\"");
        if let Some(f) = filename {
            head.push_str(&format!("; filename=\"{f}\"\r\nContent-Type: image/jpeg"));
        }
        out.extend_from_slice(format!("{head}\r\n\r\n").as_bytes());
        out.extend_from_slice(data);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), out)
}

/// Bytes that differ by `seed` and index, so no two photos are equal.
pub fn pattern(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_add((i % 251) as u8).wrapping_mul(3))
        .collect()
}

/// What the body does before it yields chunk number `at`.
pub enum Pace {
    /// Tell the test the body was polled for chunk `at`, then wait for `go`.
    Pause {
        reached: Arc<Notify>,
        go: Arc<Notify>,
    },
    /// Wait this long.
    Sleep(std::time::Duration),
    /// The connection is lost.
    Fail,
}

/// A body of `chunks` that counts the bytes asked for in `pulled` and follows `pace`.
pub fn paced(chunks: Vec<Bytes>, pulled: &Arc<AtomicU64>, pace: Vec<(usize, Pace)>) -> Body {
    let pace = Arc::new(pace);
    let pulled = Arc::clone(pulled);
    let rest: std::collections::VecDeque<Bytes> = chunks.into();
    Body::from_stream(stream::unfold((0_usize, rest), move |(i, mut rest)| {
        let pace = Arc::clone(&pace);
        let pulled = Arc::clone(&pulled);
        async move {
            for (at, step) in pace.iter().filter(|(at, _)| *at == i) {
                debug_assert_eq!(*at, i);
                match step {
                    Pace::Pause { reached, go } => {
                        reached.notify_one();
                        go.notified().await;
                    }
                    Pace::Sleep(d) => tokio::time::sleep(*d).await,
                    Pace::Fail => {
                        let lost = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "gone");
                        return Some((Err(lost), (i + 1, rest)));
                    }
                }
            }
            let chunk = rest.pop_front()?;
            pulled.fetch_add(chunk.len() as u64, Ordering::SeqCst);
            Some((Ok::<_, std::io::Error>(chunk), (i + 1, rest)))
        }
    }))
}
