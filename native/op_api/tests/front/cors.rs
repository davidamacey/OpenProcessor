//! CORS on the answers the front builds itself: a browser on another origin must be able to read
//! the status, `Retry-After` and the body of the 503, 502, 400 and 500 the front produces, as it
//! can for an answer the Python API produces.

use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::routing::get;
use http::StatusCode;
use op_api::middleware;
use op_api::state::AppState;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::support::{
    Answer, Front, get_path, request, send, settings, start_front, start_front_with, start_stub,
};

const ORIGIN: &str = "http://localhost:5184";

async fn with_origin(front: &Front, method: &str, path: &str, origin: &str) -> Answer {
    send(
        request(front, method, path)
            .header("origin", origin)
            .body(Body::empty())
            .unwrap(),
    )
    .await
}

fn assert_readable_by(answer: &Answer, origin: &str) {
    assert_eq!(answer.headers["access-control-allow-origin"], origin);
    assert_eq!(answer.headers["access-control-allow-credentials"], "true");
    let exposed = answer.headers["access-control-expose-headers"]
        .to_str()
        .unwrap()
        .to_lowercase();
    assert!(
        exposed.contains("x-request-id") && exposed.contains("x-process-time"),
        "{exposed}"
    );
    let allow_origin = answer.headers.get_all("access-control-allow-origin");
    assert_eq!(allow_origin.iter().count(), 1, "one allow-origin header");
}

fn assert_closed_to(answer: &Answer) {
    assert!(answer.headers.get("access-control-allow-origin").is_none());
}

#[tokio::test]
async fn the_503_for_an_upstream_that_is_down_is_readable_by_an_allowed_origin() {
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let nobody_home = closed.local_addr().unwrap();
    drop(closed);
    let front = start_front(nobody_home, &[]).await;
    let answer = with_origin(&front, "POST", "/echo", ORIGIN).await;
    assert_eq!(answer.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(answer.headers["retry-after"], "5");
    assert_readable_by(&answer, ORIGIN);
    assert_closed_to(&with_origin(&front, "POST", "/echo", "https://evil.example").await);
    front.stop().await;
}

#[tokio::test]
async fn the_503_of_a_shed_request_is_readable_by_an_allowed_origin() {
    let (stub, control) = start_stub().await;
    let front = start_front(stub, &[("OP_API_CONCURRENCY_LIMIT", "1")]).await;
    let first = tokio::spawn(send(
        request(&front, "GET", "/hold").body(Body::empty()).unwrap(),
    ));
    control.held.acquire().await.unwrap().forget();
    let shed = with_origin(&front, "GET", "/echo", ORIGIN).await;
    assert_eq!(shed.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(shed.headers["retry-after"], "1");
    assert_readable_by(&shed, ORIGIN);
    control.hold.add_permits(1);
    first.await.unwrap();
    front.stop().await;
}

#[tokio::test]
async fn the_502_for_an_upstream_that_answers_nonsense_is_readable_by_an_allowed_origin() {
    let broken = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = broken.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = broken.accept().await else {
                return;
            };
            drop(stream);
        }
    });
    let front = start_front(addr, &[]).await;
    let answer = with_origin(&front, "GET", "/echo", ORIGIN).await;
    assert_eq!(answer.status, StatusCode::BAD_GATEWAY);
    assert_readable_by(&answer, ORIGIN);
    front.stop().await;
}

#[tokio::test]
async fn the_400_for_a_request_target_the_front_cannot_forward_is_readable_by_an_allowed_origin() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let mut stream = TcpStream::connect(front.addr).await.unwrap();
    let connect = format!(
        "CONNECT example.test:443 HTTP/1.1\r\nhost: example.test:443\r\norigin: {ORIGIN}\r\nconnection: close\r\n\r\n"
    );
    stream.write_all(connect.as_bytes()).await.unwrap();
    let mut seen = Vec::new();
    tokio::time::timeout(Duration::from_secs(20), stream.read_to_end(&mut seen))
        .await
        .unwrap()
        .unwrap();
    let text = String::from_utf8_lossy(&seen).to_lowercase();
    assert!(text.starts_with("http/1.1 400"), "{text}");
    assert!(
        text.contains(&format!("access-control-allow-origin: {ORIGIN}")),
        "{text}"
    );
    front.stop().await;
}

async fn boom() -> &'static str {
    panic!("handler bug")
}

#[tokio::test]
async fn the_500_of_a_panicking_handler_is_readable_by_an_allowed_origin() {
    let (stub, _control) = start_stub().await;
    let settings = settings(stub, &[]);
    let state = AppState::new(&settings);
    let router = middleware::common(
        Router::new()
            .route("/boom", get(boom))
            .with_state(state.clone()),
        &op_api::cors::origin_pattern().unwrap(),
    );
    let front = start_front_with(settings, Some(router)).await;
    let answer = with_origin(&front, "GET", "/boom", ORIGIN).await;
    assert_eq!(answer.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_readable_by(&answer, ORIGIN);
    front.stop().await;
}

#[tokio::test]
async fn an_answer_the_python_api_gave_keeps_its_own_cors_headers_once() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = with_origin(&front, "GET", "/cors-upstream", ORIGIN).await;
    assert_eq!(answer.status, StatusCode::OK);
    let allowed: Vec<_> = answer
        .headers
        .get_all("access-control-allow-origin")
        .iter()
        .collect();
    assert_eq!(allowed, ["http://upstream.test"]);
    let credentials: Vec<_> = answer
        .headers
        .get_all("access-control-allow-credentials")
        .iter()
        .collect();
    assert_eq!(credentials, ["true"]);
    front.stop().await;
}

#[tokio::test]
async fn a_request_without_an_origin_gets_no_cors_headers_on_a_front_built_answer() {
    let closed = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let nobody_home = closed.local_addr().unwrap();
    drop(closed);
    let front = start_front(nobody_home, &[]).await;
    let answer = get_path(&front, "/echo").await;
    assert_eq!(answer.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_closed_to(&answer);
    front.stop().await;
}
