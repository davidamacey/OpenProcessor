//! A client that gives up on its upload is not an upstream failure: it is counted and logged as
//! what it is, so an alert on upstream failures does not fire on a browser's cancelled upload.

use std::io;
use std::net::SocketAddr;

use axum::body::{Body, Bytes};
use axum::extract::Request;
use http::StatusCode;
use op_api::state::AppState;

use crate::support::{settings, start_stub};

const PEER: &str = "192.0.2.7:50000";

#[tokio::test]
async fn a_client_that_aborts_its_upload_is_counted_as_that_and_not_as_an_upstream_failure() {
    let (stub, _control) = start_stub().await;
    let state = AppState::new(&settings(stub, &[]));
    let pieces: Vec<Result<Bytes, io::Error>> = vec![
        Ok(Bytes::from_static(b"abc")),
        Err(io::Error::new(
            io::ErrorKind::ConnectionReset,
            "client went away",
        )),
    ];
    let request = Request::post("/echo")
        .body(Body::from_stream(tokio_stream::iter(pieces)))
        .unwrap();
    let peer: SocketAddr = PEER.parse().unwrap();
    let response = state.forwarder.forward(peer, request).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let text = state.metrics.render().unwrap();
    assert!(
        text.contains("op_api_forward_failures_total{reason=\"client_aborted\"} 1"),
        "{text}"
    );
    assert!(!text.contains("reason=\"upstream\""), "{text}");
    assert!(!text.contains("reason=\"connect\""), "{text}");
}

#[tokio::test]
async fn an_upstream_that_breaks_is_still_an_upstream_failure() {
    let broken = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = broken.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((stream, _)) = broken.accept().await {
            drop(stream);
        }
    });
    let state = AppState::new(&settings(addr, &[]));
    let request = Request::get("/echo").body(Body::empty()).unwrap();
    let response = state
        .forwarder
        .forward(PEER.parse().unwrap(), request)
        .await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let text = state.metrics.render().unwrap();
    assert!(
        text.contains("op_api_forward_failures_total{reason=\"upstream\"} 1"),
        "{text}"
    );
    assert!(!text.contains("client_aborted"), "{text}");
}
