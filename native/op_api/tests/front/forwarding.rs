//! What reaches the Python API and what comes back: the message, unchanged but for the hop.

use std::convert::Infallible;

use axum::body::{Body, Bytes};
use http::StatusCode;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::support::{get_path, request, send, start_front, start_stub};

#[tokio::test]
async fn method_path_query_headers_and_body_reach_the_upstream_unchanged() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let sent = request(&front, "POST", "/echo/a%20b/c?x=1&y=%C3%A9&x=2")
        .header("content-type", "application/json")
        .header("x-custom", "one")
        .header("authorization", "Bearer t")
        .body(Body::from(r#"{"k":1}"#))
        .unwrap();
    let answer = send(sent).await;
    assert_eq!(answer.status, StatusCode::OK);
    assert_eq!(answer.headers["x-from-upstream"], "yes");
    let seen = answer.json();
    assert_eq!(seen["method"], "POST");
    assert_eq!(seen["target"], "/echo/a%20b/c?x=1&y=%C3%A9&x=2");
    assert_eq!(seen["body_len"], 7);
    assert_eq!(seen["headers"]["content-type"][0], "application/json");
    assert_eq!(seen["headers"]["x-custom"][0], "one");
    assert_eq!(seen["headers"]["authorization"][0], "Bearer t");
    front.stop().await;
}

#[tokio::test]
async fn every_method_is_forwarded_and_a_status_and_body_come_back_as_they_are() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    for method in ["GET", "PUT", "PATCH", "DELETE", "OPTIONS"] {
        let seen = send(
            request(&front, method, "/echo")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(seen.json()["method"], method);
    }
    for code in [201_u16, 404, 422, 500, 503] {
        let answer = get_path(&front, &format!("/status/{code}")).await;
        assert_eq!(answer.status.as_u16(), code);
        assert_eq!(&answer.body[..], b"stub says so");
    }
    front.stop().await;
}

#[tokio::test]
async fn hop_by_hop_fields_of_the_request_are_not_sent_on_and_the_others_are() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let sent = request(&front, "GET", "/echo")
        .header("connection", "x-drop-me, upgrade")
        .header("upgrade", "websocket")
        .header("x-drop-me", "1")
        .header("te", "trailers")
        .header("keep-alive", "timeout=1")
        .header("proxy-authorization", "x")
        .header("x-keep", "1")
        .body(Body::empty())
        .unwrap();
    let answer = send(sent).await;
    assert_eq!(
        answer.status,
        StatusCode::OK,
        "an upgrade request is answered as a plain request"
    );
    let headers = answer.json()["headers"].clone();
    for gone in [
        "connection",
        "upgrade",
        "x-drop-me",
        "te",
        "keep-alive",
        "proxy-authorization",
    ] {
        assert!(
            headers.get(gone).is_none(),
            "{gone} reached the upstream: {headers}"
        );
    }
    assert_eq!(headers["x-keep"][0], "1");
    front.stop().await;
}

#[tokio::test]
async fn the_response_loses_its_hop_by_hop_fields_and_keeps_every_cookie() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = get_path(&front, "/headers-out").await;
    assert_eq!(&answer.body[..], b"body");
    assert!(answer.headers.get("x-secret").is_none());
    assert!(answer.headers.get("keep-alive").is_none());
    let cookies: Vec<_> = answer.headers.get_all("set-cookie").iter().collect();
    assert_eq!(cookies, ["a=1", "b=2"]);
    front.stop().await;
}

#[tokio::test]
async fn the_upstream_sees_the_host_and_forwarded_fields_the_origin_rule_reads_and_the_peer_as_the_client()
 {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let sent = request(&front, "GET", "/echo")
        .header("host", "photos.test:5184")
        .header("x-forwarded-host", "gallery.test")
        .header("x-forwarded-for", "203.0.113.9")
        .header("x-forwarded-for", "198.51.100.2")
        .body(Body::empty())
        .unwrap();
    let headers = send(sent).await.json()["headers"].clone();
    assert_eq!(headers["host"][0], "photos.test:5184");
    assert_eq!(headers["x-forwarded-host"][0], "gallery.test");
    assert_eq!(
        headers["x-forwarded-for"],
        serde_json::json!(["127.0.0.1"]),
        "a client-supplied chain is dropped: the front is the edge and says who connected"
    );
    assert_eq!(headers["x-forwarded-proto"][0], "http");
    front.stop().await;
}

#[tokio::test]
async fn a_body_with_a_length_keeps_it_and_one_without_is_sent_chunked() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let sized = send(
        request(&front, "POST", "/echo")
            .body(Body::from("12345"))
            .unwrap(),
    )
    .await
    .json();
    assert_eq!(sized["headers"]["content-length"][0], "5");
    assert!(sized["headers"].get("transfer-encoding").is_none());
    assert_eq!(sized["body_len"], 5);

    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(2);
    sender.send(Ok(Bytes::from_static(b"abc"))).await.unwrap();
    sender.send(Ok(Bytes::from_static(b"de"))).await.unwrap();
    drop(sender);
    let unsized_ = send(
        request(&front, "POST", "/echo")
            .body(Body::from_stream(ReceiverStream::new(receiver)))
            .unwrap(),
    )
    .await
    .json();
    assert_eq!(unsized_["headers"]["transfer-encoding"][0], "chunked");
    assert!(unsized_["headers"].get("content-length").is_none());
    assert_eq!(unsized_["body_len"], 5);
    front.stop().await;
}

#[tokio::test]
async fn an_upstream_that_cannot_be_reached_is_a_503_with_retry_after_and_is_counted() {
    let closed = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let nobody_home = closed.local_addr().unwrap();
    drop(closed);
    let front = start_front(nobody_home, &[]).await;
    let answer = get_path(&front, "/echo").await;
    assert_eq!(answer.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(answer.headers["retry-after"], "5");
    let body = answer.json();
    assert!(body["detail"].is_string(), "{body}");
    assert_eq!(
        body["request_id"],
        answer.headers["x-request-id"].to_str().unwrap()
    );
    let metrics = get_path(&front, "/metrics/front").await;
    let text = String::from_utf8_lossy(&metrics.body);
    assert!(
        text.contains("op_api_forward_failures_total{reason=\"connect\"} 1"),
        "{text}"
    );
    front.stop().await;
}
