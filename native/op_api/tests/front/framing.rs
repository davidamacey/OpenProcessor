//! How a request's length is framed on its way through: both `Content-Length` and
//! `Transfer-Encoding`, repeated or malformed lengths, and `Expect: 100-continue`. These pin what
//! hyper's server and client do with the front's streaming body, so a change of either (or of how
//! the front passes the fields) shows here.

use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::support::{bounded, start_front, start_stub};

const GUARD: Duration = Duration::from_secs(20);

/// Send `raw` and read until the front closes the connection (the request says
/// `connection: close`); returns the whole answer as text.
async fn raw(addr: std::net::SocketAddr, raw: &[u8]) -> String {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream.write_all(raw).await.unwrap();
    let mut seen = Vec::new();
    tokio::time::timeout(GUARD, stream.read_to_end(&mut seen))
        .await
        .expect("the front did not close the connection")
        .ok();
    String::from_utf8_lossy(&seen).into_owned()
}

fn status_line(answer: &str) -> &str {
    answer.lines().next().unwrap_or_default()
}

fn echoed(answer: &str) -> Value {
    let body = answer.split("\r\n\r\n").nth(1).unwrap_or_default();
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {answer:?}"))
}

#[tokio::test]
async fn both_content_length_and_transfer_encoding_reach_the_upstream_as_chunked_only() {
    bounded("both_content_length_and_transfer_encoding_reach_the_upstream_as_chunked_only", async {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let chunks = "3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n";
    for fields in [
        "content-length: 5\r\ntransfer-encoding: chunked",
        "transfer-encoding: chunked\r\ncontent-length: 5",
    ] {
        let request = format!(
            "POST /echo HTTP/1.1\r\nhost: t\r\n{fields}\r\nconnection: close\r\n\r\n{chunks}"
        );
        let answer = raw(front.addr, request.as_bytes()).await;
        assert!(status_line(&answer).contains("200"), "{fields}: {answer}");
        let seen = echoed(&answer);
        assert_eq!(seen["body_len"], 5, "{fields}");
        assert_eq!(
            seen["headers"]["transfer-encoding"][0], "chunked",
            "{fields}"
        );
        assert!(
            seen["headers"].get("content-length").is_none(),
            "{fields}: {seen}"
        );
    }
    front.stop().await;
    })
    .await;
}

#[tokio::test]
async fn repeated_or_malformed_lengths_are_refused_before_the_upstream_hears_of_them() {
    bounded(
        "repeated_or_malformed_lengths_are_refused_before_the_upstream_hears_of_them",
        async {
            let (stub, control) = start_stub().await;
            let front = start_front(stub, &[]).await;
            for fields in [
                "content-length: 5\r\ncontent-length: 6",
                "content-length: 5, 5",
                "content-length: abc",
                "content-length: -1",
                "content-length: 5\r\ntransfer-encoding: chunked, identity",
                "transfer-encoding: xchunked",
            ] {
                let request = format!(
                    "POST /echo HTTP/1.1\r\nhost: t\r\n{fields}\r\nconnection: close\r\n\r\nabcde"
                );
                let answer = raw(front.addr, request.as_bytes()).await;
                assert!(status_line(&answer).contains("400"), "{fields}: {answer}");
            }
            assert_eq!(
                control.requests.load(Ordering::SeqCst),
                0,
                "nothing reached the upstream"
            );
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn the_same_length_given_twice_is_sent_on_once() {
    bounded("the_same_length_given_twice_is_sent_on_once", async {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let request = "POST /echo HTTP/1.1\r\nhost: t\r\ncontent-length: 5\r\ncontent-length: 5\r\nconnection: close\r\n\r\nabcde";
    let answer = raw(front.addr, request.as_bytes()).await;
    assert!(status_line(&answer).contains("200"), "{answer}");
    let seen = echoed(&answer);
    assert_eq!(seen["body_len"], 5);
    assert_eq!(seen["headers"]["content-length"], serde_json::json!(["5"]));
    front.stop().await;
    })
    .await;
}

#[tokio::test]
async fn an_upload_that_expects_100_continue_is_told_to_go_on_and_is_answered() {
    bounded("an_upload_that_expects_100_continue_is_told_to_go_on_and_is_answered", async {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let mut stream = TcpStream::connect(front.addr).await.unwrap();
    stream
        .write_all(b"POST /echo HTTP/1.1\r\nhost: t\r\nexpect: 100-continue\r\ncontent-length: 5\r\nconnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut seen = Vec::new();
    let mut buf = [0_u8; 256];
    while !String::from_utf8_lossy(&seen).contains("100 Continue") {
        let n = tokio::time::timeout(GUARD, stream.read(&mut buf))
            .await
            .unwrap()
            .unwrap();
        assert!(
            n > 0,
            "closed before 100 Continue: {:?}",
            String::from_utf8_lossy(&seen)
        );
        seen.extend_from_slice(&buf[..n]);
    }
    stream.write_all(b"abcde").await.unwrap();
    let mut rest = Vec::new();
    tokio::time::timeout(GUARD, stream.read_to_end(&mut rest))
        .await
        .unwrap()
        .ok();
    let answer = String::from_utf8_lossy(&rest).into_owned();
    assert!(status_line(&answer).contains("200"), "{answer}");
    assert_eq!(echoed(&answer)["body_len"], 5);
    front.stop().await;
    })
    .await;
}
