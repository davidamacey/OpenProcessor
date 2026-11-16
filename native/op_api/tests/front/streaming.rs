//! Bodies are never held: a request body and a response body pass in pieces as they come, and
//! server-sent events arrive as they are sent.

use std::convert::Infallible;

use axum::body::{Body, Bytes};
use http_body_util::BodyExt;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::support::{bounded, client, request, send, start_front, start_stub};

async fn next_data(body: &mut Body) -> Option<Bytes> {
    let frame = tokio::time::timeout(std::time::Duration::from_secs(20), body.frame())
        .await
        .expect("no frame in time")?
        .expect("the body failed");
    frame.into_data().ok()
}

#[tokio::test]
async fn a_piece_of_the_request_comes_back_before_the_next_piece_is_sent() {
    bounded(
        "a_piece_of_the_request_comes_back_before_the_next_piece_is_sent",
        async {
            let (stub, _control) = start_stub().await;
            let front = start_front(stub, &[]).await;
            let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(1);
            let sent = request(&front, "POST", "/stream-echo")
                .body(Body::from_stream(ReceiverStream::new(receiver)))
                .unwrap();
            let pending = tokio::spawn(client().request(sent));
            sender.send(Ok(Bytes::from_static(b"first"))).await.unwrap();
            let response = tokio::time::timeout(std::time::Duration::from_secs(20), pending)
                .await
                .expect("the response head did not arrive while the request body was still open")
                .unwrap()
                .expect("the request failed");
            let mut body = Body::new(response.into_body());
            assert_eq!(next_data(&mut body).await.unwrap(), "first");
            sender
                .send(Ok(Bytes::from_static(b"second")))
                .await
                .unwrap();
            assert_eq!(next_data(&mut body).await.unwrap(), "second");
            drop(sender);
            assert!(next_data(&mut body).await.is_none());
            front.stop().await;
        },
    )
    .await;
}

#[tokio::test]
async fn a_large_upload_passes_with_every_byte() {
    bounded("a_large_upload_passes_with_every_byte", async {
        const CHUNK: usize = 64 * 1024;
        const CHUNKS: usize = 384;
        let (stub, _control) = start_stub().await;
        let front = start_front(stub, &[]).await;
        let chunk = Bytes::from(vec![7_u8; CHUNK]);
        let stream =
            tokio_stream::iter((0..CHUNKS).map(move |_| Ok::<_, Infallible>(chunk.clone())));
        let seen = send(
            request(&front, "POST", "/echo")
                .body(Body::from_stream(stream))
                .unwrap(),
        )
        .await
        .json();
        assert_eq!(seen["body_len"], CHUNK * CHUNKS);
        assert_eq!(seen["body_sum"], 7 * CHUNK * CHUNKS);
        front.stop().await;
    })
    .await;
}

#[tokio::test]
async fn server_sent_events_arrive_one_by_one_while_the_stream_stays_open() {
    bounded(
        "server_sent_events_arrive_one_by_one_while_the_stream_stays_open",
        async {
            let (stub, control) = start_stub().await;
            let front = start_front(stub, &[]).await;
            let response = client()
                .request(
                    request(&front, "GET", "/events")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .expect("the request failed");
            assert_eq!(response.headers()["content-type"], "text/event-stream");
            assert_eq!(response.headers()["cache-control"], "no-cache");
            assert!(response.headers().get("content-length").is_none());
            let mut body = Body::new(response.into_body());
            control
                .events
                .send(Ok(Bytes::from_static(b"data: one\n\n")))
                .await
                .unwrap();
            assert_eq!(next_data(&mut body).await.unwrap(), "data: one\n\n");
            control
                .events
                .send(Ok(Bytes::from_static(b"data: two\n\n")))
                .await
                .unwrap();
            assert_eq!(next_data(&mut body).await.unwrap(), "data: two\n\n");
            drop(control);
            assert!(next_data(&mut body).await.is_none());
            front.stop().await;
        },
    )
    .await;
}
