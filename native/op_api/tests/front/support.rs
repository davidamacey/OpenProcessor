//! A stub Python API, a running front and a client, for the tests that need real sockets.
//!
//! Every wait in these tests is on an event the test itself causes (a channel, a permit, a
//! connection closing); nothing sleeps to let time pass.

use std::collections::BTreeMap;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use http::{HeaderMap, StatusCode};
use http_body_util::BodyExt;
use hyper_util::client::legacy::Client;
use hyper_util::client::legacy::connect::HttpConnector;
use hyper_util::rt::TokioExecutor;
use op_api::serve::{ServeConfig, Stopped, serve};
use op_api::settings::{ApiSettings, PY_API_URL};
use op_config::Env;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;

type EventReceiver = mpsc::Receiver<Result<Bytes, Infallible>>;

/// What the tests can do to the stub from outside.
pub struct StubControl {
    /// Send server-sent events from `/events`; the first request to `/events` takes the receiver.
    pub events: mpsc::Sender<Result<Bytes, Infallible>>,
    /// `/hold` answers when a permit is added.
    pub hold: Arc<Semaphore>,
    /// How many requests `/hold` has received.
    pub held: Arc<Semaphore>,
    /// How many requests of any path the stub has received.
    pub requests: Arc<AtomicUsize>,
}

#[derive(Clone)]
struct StubState {
    events: Arc<Mutex<Option<EventReceiver>>>,
    hold: Arc<Semaphore>,
    held: Arc<Semaphore>,
    requests: Arc<AtomicUsize>,
}

/// What `/echo` saw of a request.
pub async fn echo(request: Request) -> Response {
    let (parts, body) = request.into_parts();
    let bytes = body
        .collect()
        .await
        .map(http_body_util::Collected::to_bytes)
        .unwrap_or_default();
    let mut headers: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, value) in &parts.headers {
        headers
            .entry(name.as_str().to_owned())
            .or_default()
            .push(String::from_utf8_lossy(value.as_bytes()).into_owned());
    }
    let body = json!({
        "method": parts.method.as_str(),
        "target": parts.uri.to_string(),
        "version": format!("{:?}", parts.version),
        "headers": headers,
        "body_len": bytes.len(),
        "body_sum": bytes.iter().map(|b| u64::from(*b)).sum::<u64>(),
    });
    (
        StatusCode::OK,
        [("x-from-upstream", "yes"), ("server", "stub")],
        axum::Json(body),
    )
        .into_response()
}

fn stub_router(state: StubState) -> Router {
    Router::new()
        .route("/echo", any(echo))
        .route("/echo/{*rest}", any(echo))
        .route("/metrics", any(echo))
        .route("/stream-echo", any(stream_echo))
        .route("/events", get(events))
        .route("/hold", get(hold))
        .route("/headers-out", get(headers_out))
        .route("/cors-upstream", get(cors_upstream))
        .route("/status/{code}", any(status))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            count_request,
        ))
        .with_state(state)
}

async fn count_request(
    State(state): State<StubState>,
    request: Request,
    next: axum::middleware::Next,
) -> Response {
    state.requests.fetch_add(1, Ordering::SeqCst);
    next.run(request).await
}

/// Answers with the request body as it arrives, frame by frame.
async fn stream_echo(request: Request) -> Response {
    Body::from_stream(request.into_body().into_data_stream()).into_response()
}

async fn events(State(state): State<StubState>) -> Response {
    let receiver = state.events.lock().ok().and_then(|mut slot| slot.take());
    let Some(receiver) = receiver else {
        return StatusCode::GONE.into_response();
    };
    (
        [
            ("content-type", "text/event-stream"),
            ("cache-control", "no-cache"),
        ],
        Body::from_stream(ReceiverStream::new(receiver)),
    )
        .into_response()
}

async fn hold(State(state): State<StubState>) -> Response {
    state.held.add_permits(1);
    if let Ok(permit) = state.hold.acquire().await {
        permit.forget();
    }
    "released".into_response()
}

async fn headers_out() -> Response {
    let mut response = "body".into_response();
    let headers = response.headers_mut();
    headers.insert("connection", "x-secret".parse().unwrap());
    headers.insert("x-secret", "1".parse().unwrap());
    headers.insert("keep-alive", "timeout=5".parse().unwrap());
    headers.append("set-cookie", "a=1".parse().unwrap());
    headers.append("set-cookie", "b=2".parse().unwrap());
    response
}

/// An answer that carries the CORS headers the Python API's middleware would add.
async fn cors_upstream() -> Response {
    let mut response = "cors".into_response();
    let headers = response.headers_mut();
    headers.insert(
        "access-control-allow-origin",
        "http://upstream.test".parse().unwrap(),
    );
    headers.insert("access-control-allow-credentials", "true".parse().unwrap());
    response
}

async fn status(axum::extract::Path(code): axum::extract::Path<u16>) -> Response {
    (StatusCode::from_u16(code).unwrap(), "stub says so").into_response()
}

/// Start a stub Python API on a free loopback port.
pub async fn start_stub() -> (SocketAddr, StubControl) {
    let (events_tx, events_rx) = mpsc::channel(8);
    let hold = Arc::new(Semaphore::new(0));
    let held = Arc::new(Semaphore::new(0));
    let requests = Arc::new(AtomicUsize::new(0));
    let state = StubState {
        events: Arc::new(Mutex::new(Some(events_rx))),
        hold: Arc::clone(&hold),
        held: Arc::clone(&held),
        requests: Arc::clone(&requests),
    };
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, stub_router(state)).await });
    (
        addr,
        StubControl {
            events: events_tx,
            hold,
            held,
            requests,
        },
    )
}

/// A running front.
pub struct Front {
    /// Where it listens.
    pub addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    done: Option<tokio::task::JoinHandle<Stopped>>,
}

impl Front {
    /// Send the stop signal and wait for the server to return.
    pub async fn stop(mut self) -> Stopped {
        if let Some(stop) = self.stop.take() {
            stop.send(()).ok();
        }
        self.done.take().unwrap().await.unwrap()
    }

    /// Send the stop signal and do not wait.
    pub fn signal_stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            stop.send(()).ok();
        }
    }

    /// Wait for the server to return after a signal.
    pub async fn finished(mut self) -> Stopped {
        self.done.take().unwrap().await.unwrap()
    }
}

/// The settings a test front runs with: the upstream at `upstream`, the rest as the defaults.
pub fn settings(upstream: SocketAddr, extra: &[(&str, &str)]) -> ApiSettings {
    let url = format!("http://{upstream}");
    let mut pairs = vec![(PY_API_URL.to_owned(), url)];
    pairs.extend(
        extra
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned())),
    );
    ApiSettings::from_env(&Env::from_pairs(pairs)).unwrap()
}

/// Start a front in front of `upstream`; `extra` are environment settings.
pub async fn start_front(upstream: SocketAddr, extra: &[(&str, &str)]) -> Front {
    start_front_with(settings(upstream, extra), None).await
}

/// Start a front from settings, optionally with a router built by the test.
pub async fn start_front_with(settings: ApiSettings, router: Option<Router>) -> Front {
    let router = router.unwrap_or_else(|| op_api::build(&settings).unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let config = ServeConfig {
        keep_alive: settings.keep_alive,
        shutdown_grace: settings.shutdown_grace,
        max_connections: settings.max_connections,
    };
    let (stop, stopped) = oneshot::channel::<()>();
    let done = tokio::spawn(async move {
        serve(listener, router, config, async {
            stopped.await.ok();
        })
        .await
    });
    Front {
        addr,
        stop: Some(stop),
        done: Some(done),
    }
}

/// A client with its own pool.
pub fn client() -> Client<HttpConnector, Body> {
    Client::builder(TokioExecutor::new()).build_http()
}

/// What an answer holds once its body has been read.
pub struct Answer {
    /// The status line.
    pub status: StatusCode,
    /// The headers.
    pub headers: HeaderMap,
    /// The whole body.
    pub body: Bytes,
}

impl Answer {
    /// The body as JSON.
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or_else(|e| panic!("{e}: {:?}", self.body))
    }
}

/// Send `request` and read the whole answer, bounded so a hang fails the test.
pub async fn send(request: http::Request<Body>) -> Answer {
    let response = tokio::time::timeout(Duration::from_secs(20), client().request(request))
        .await
        .expect("the front did not answer in time")
        .expect("the request failed");
    let (parts, body) = response.into_parts();
    let body = tokio::time::timeout(Duration::from_secs(20), body.collect())
        .await
        .expect("the body did not end in time")
        .expect("reading the body failed")
        .to_bytes();
    Answer {
        status: parts.status,
        headers: parts.headers,
        body,
    }
}

/// A request to the front at `path`.
pub fn request(front: &Front, method: &str, path: &str) -> http::request::Builder {
    http::Request::builder()
        .method(method)
        .uri(format!("http://{}{path}", front.addr))
}

/// GET `path` and read the answer.
pub async fn get_path(front: &Front, path: &str) -> Answer {
    send(request(front, "GET", path).body(Body::empty()).unwrap()).await
}
