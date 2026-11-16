//! The accept loop: idle connections, the connection bound and the stop.

use std::time::Duration;

use axum::body::Body;
use op_api::serve::Stopped;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::support::{bounded, permit, request, send, start_front, start_stub};

const GUARD: Duration = Duration::from_secs(20);
const GET_LIVE: &[u8] = b"GET /live HTTP/1.1\r\nhost: t\r\n\r\n";

async fn read_until_closed(stream: &mut TcpStream) -> Vec<u8> {
    let mut seen = Vec::new();
    tokio::time::timeout(GUARD, stream.read_to_end(&mut seen))
        .await
        .expect("the server did not close the connection")
        .ok();
    seen
}

#[tokio::test]
async fn a_connection_that_sends_nothing_is_closed_after_the_keep_alive_time() {
    bounded(
        "a_connection_that_sends_nothing_is_closed_after_the_keep_alive_time",
        async {
            let (stub, _control) = start_stub().await;
            let front = start_front(stub, &[("OP_API_KEEP_ALIVE_S", "1")]).await;
            let mut idle = TcpStream::connect(front.addr).await.unwrap();
            read_until_closed(&mut idle).await;
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn a_connection_idle_after_an_answer_is_closed_too() {
    bounded("a_connection_idle_after_an_answer_is_closed_too", async {
        let (stub, _control) = start_stub().await;
        let front = start_front(stub, &[("OP_API_KEEP_ALIVE_S", "1")]).await;
        let mut connection = TcpStream::connect(front.addr).await.unwrap();
        connection.write_all(GET_LIVE).await.unwrap();
        let seen = read_until_closed(&mut connection).await;
        let text = String::from_utf8_lossy(&seen);
        assert!(text.starts_with("HTTP/1.1 200"), "{text}");
        assert!(text.contains(r#"{"status":"alive"}"#), "{text}");
        front.stop().await;
    })
    .await;
}

#[tokio::test]
async fn past_the_connection_bound_a_client_waits_until_one_closes() {
    bounded(
        "past_the_connection_bound_a_client_waits_until_one_closes",
        async {
            let (stub, _control) = start_stub().await;
            let front = start_front(stub, &[("OP_API_MAX_CONNECTIONS", "1")]).await;
            let first = TcpStream::connect(front.addr).await.unwrap();
            let mut second = TcpStream::connect(front.addr).await.unwrap();
            second.write_all(GET_LIVE).await.unwrap();
            let mut byte = [0_u8; 1];
            let early =
                tokio::time::timeout(Duration::from_millis(300), second.read(&mut byte)).await;
            assert!(
                early.is_err(),
                "the second client was served while the first held the only slot"
            );
            drop(first);
            let mut seen = vec![0_u8; 12];
            tokio::time::timeout(GUARD, second.read_exact(&mut seen))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(&seen, b"HTTP/1.1 200");
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn the_stop_lets_a_request_in_flight_finish_and_then_returns() {
    bounded(
        "the_stop_lets_a_request_in_flight_finish_and_then_returns",
        async {
            let (stub, control) = start_stub().await;
            let mut front = start_front(stub, &[]).await;
            let in_flight = tokio::spawn(send(
                request(&front, "GET", "/hold").body(Body::empty()).unwrap(),
            ));
            permit(&control.held, "the stub to receive the /hold request").await;
            front.signal_stop();
            control.hold.add_permits(1);
            let answer = in_flight.await.unwrap();
            assert_eq!(&answer.body[..], b"released");
            assert_eq!(front.finished().await, Stopped::Drained);
        },
    )
    .await;
}

#[tokio::test]
async fn a_request_that_outlasts_the_grace_is_cut_and_the_server_returns() {
    bounded(
        "a_request_that_outlasts_the_grace_is_cut_and_the_server_returns",
        async {
            let (stub, control) = start_stub().await;
            let mut front = start_front(stub, &[("OP_API_SHUTDOWN_GRACE_S", "1")]).await;
            let stuck = tokio::spawn(send(
                request(&front, "GET", "/hold").body(Body::empty()).unwrap(),
            ));
            permit(&control.held, "the stub to receive the /hold request").await;
            front.signal_stop();
            assert_eq!(front.finished().await, Stopped::GraceExpired { open: 1 });
            control.hold.add_permits(1);
            stuck.abort();
        },
    )
    .await;
}
