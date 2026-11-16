//! What happens to a request over its life: the concurrency permit, a client that goes away, an
//! upstream that dies half way through its answer.

use std::time::Duration;

use axum::body::{Body, Bytes};
use http::StatusCode;
use http_body_util::BodyExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::support::{bounded, client, get_path, permit, request, start_front, start_stub};

const GUARD: Duration = Duration::from_secs(20);

async fn read_until(stream: &mut TcpStream, needle: &[u8]) {
    let mut seen = Vec::new();
    let mut buf = [0_u8; 1024];
    while !seen.windows(needle.len()).any(|w| w == needle) {
        let n = tokio::time::timeout(GUARD, stream.read(&mut buf))
            .await
            .expect("no data in time")
            .unwrap();
        assert!(
            n > 0,
            "closed before {needle:?}: {:?}",
            String::from_utf8_lossy(&seen)
        );
        seen.extend_from_slice(&buf[..n]);
    }
}

#[tokio::test]
async fn the_permit_is_released_when_the_response_head_is_out_not_when_its_body_ends() {
    bounded(
        "the_permit_is_released_when_the_response_head_is_out_not_when_its_body_ends",
        async {
            let (stub, control) = start_stub().await;
            let front = start_front(stub, &[("OP_API_CONCURRENCY_LIMIT", "1")]).await;
            let response = tokio::time::timeout(
                GUARD,
                client().request(
                    request(&front, "GET", "/events")
                        .body(Body::empty())
                        .unwrap(),
                ),
            )
            .await
            .expect("the head of an event stream did not arrive")
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let mut body = Body::new(response.into_body());
            control
                .events
                .send(Ok(Bytes::from_static(b"data: one\n\n")))
                .await
                .unwrap();
            assert!(body.frame().await.is_some(), "the stream is open");
            // The stream is still open, and the limit is 1: a second request is served, not shed.
            let second = get_path(&front, "/echo").await;
            assert_eq!(second.status, StatusCode::OK);
            drop(body);
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn a_client_that_goes_away_during_a_plain_request_cancels_the_upstream_request() {
    bounded(
        "a_client_that_goes_away_during_a_plain_request_cancels_the_upstream_request",
        async {
            let (stub, control) = start_stub().await;
            let front = start_front(stub, &[]).await;
            let mut stream = TcpStream::connect(front.addr).await.unwrap();
            stream
                .write_all(b"GET /watch HTTP/1.1\r\nhost: t\r\n\r\n")
                .await
                .unwrap();
            permit(&control.watching, "the stub to receive the watched request").await;
            drop(stream);
            tokio::time::timeout(GUARD, control.cancelled.acquire())
                .await
                .expect("the upstream request was not cancelled when the client left")
                .unwrap()
                .forget();
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn a_client_that_goes_away_during_an_event_stream_cancels_the_upstream_request() {
    bounded(
        "a_client_that_goes_away_during_an_event_stream_cancels_the_upstream_request",
        async {
            let (stub, control) = start_stub().await;
            let front = start_front(stub, &[]).await;
            let mut stream = TcpStream::connect(front.addr).await.unwrap();
            stream
                .write_all(b"GET /watch-events HTTP/1.1\r\nhost: t\r\n\r\n")
                .await
                .unwrap();
            read_until(&mut stream, b": open").await;
            drop(stream);
            tokio::time::timeout(GUARD, control.cancelled.acquire())
                .await
                .expect("the upstream stream was not cancelled when the client left")
                .unwrap()
                .forget();
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn a_client_that_goes_away_during_an_upload_cancels_the_upstream_request() {
    bounded(
        "a_client_that_goes_away_during_an_upload_cancels_the_upstream_request",
        async {
            let (stub, control) = start_stub().await;
            let front = start_front(stub, &[]).await;
            let mut stream = TcpStream::connect(front.addr).await.unwrap();
            stream
        .write_all(
            b"POST /watch-upload HTTP/1.1\r\nhost: t\r\ncontent-length: 1000000\r\n\r\nabcdef",
        )
        .await
        .unwrap();
            permit(&control.watching, "the stub to receive the watched request").await;
            drop(stream);
            tokio::time::timeout(GUARD, control.cancelled.acquire())
                .await
                .expect("the upstream upload was not cancelled when the client left")
                .unwrap()
                .forget();
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn an_upstream_that_dies_mid_body_gives_the_client_a_broken_stream_not_a_clean_end() {
    bounded(
        "an_upstream_that_dies_mid_body_gives_the_client_a_broken_stream_not_a_clean_end",
        async {
            let (stub, control) = start_stub().await;
            let front = start_front(stub, &[]).await;
            for (path, breaker) in [
                ("/truncate", &control.truncate),
                ("/truncate-chunked", &control.truncate_chunked),
            ] {
                let response = tokio::time::timeout(
                    GUARD,
                    client().request(request(&front, "GET", path).body(Body::empty()).unwrap()),
                )
                .await
                .unwrap()
                .unwrap();
                assert_eq!(response.status(), StatusCode::OK, "{path}");
                let mut body = Body::new(response.into_body());
                breaker
                    .send(Ok(Bytes::from_static(b"0123456789")))
                    .await
                    .unwrap();
                let first = tokio::time::timeout(GUARD, body.frame()).await.unwrap();
                assert!(
                    first.is_some_and(|f| f.is_ok()),
                    "{path}: the first piece arrives"
                );
                breaker
                    .send(Err(std::io::Error::other("the upstream died")))
                    .await
                    .unwrap();
                let next = tokio::time::timeout(GUARD, body.frame()).await.unwrap();
                assert!(
                    matches!(next, Some(Err(_))),
                    "{path}: the client must see an error, not a clean end: {next:?}"
                );
            }
            // The front is fine afterwards.
            assert_eq!(get_path(&front, "/live").await.status, StatusCode::OK);
            front.stop().await;
        },
    )
    .await;
}
