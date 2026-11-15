//! The front on a real HTTP/2 connection, below what a gRPC client shows: an error answered before
//! or after the handler runs is one HEADERS frame with the status in it (gRPC over HTTP/2,
//! "Trailers-Only"), an answer carries its status in trailers, a message over the limit is
//! `RESOURCE_EXHAUSTED`, and an idle connection that pings is never sent away.

use std::sync::Arc;
use std::time::Duration;

use bytes::{BufMut, Bytes, BytesMut};
use frame_engine::front::{Front, FrontConfig, ServeConfig, serve};
use frame_engine::metrics::Metrics;
use frame_engine::proto::{ModelInferRequest, model_infer_request::InferInputTensor};
use frame_engine::stub::StubPipeline;
use frame_engine::wire::bytes_tensor::pack;
use http_body_util::{BodyExt, Full};
use hyper::Request;
use hyper::client::conn::http2;
use hyper::rt::Executor;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use prost::Message;
use tokio::net::{TcpListener, TcpStream};

const FRAME: &str = r#"{"version": 1, "branches": ["detect"], "detector": {"model": "d", "input_size": 640, "letterbox_fill": 114}}"#;

struct Running {
    addr: std::net::SocketAddr,
    stop: tokio::sync::oneshot::Sender<()>,
    server: tokio::task::JoinHandle<()>,
}

async fn running(max_message_mib: u32) -> Running {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let config = FrontConfig {
        max_in_flight: 4,
        request_deadline_seconds: 30.0,
    };
    let (front, handle) = Front::new(Arc::new(StubPipeline::new(None)), Metrics::new(), config);
    let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
    let serve_config = ServeConfig {
        max_message_mib,
        drain: Duration::from_secs(2),
    };
    let server = tokio::spawn(async move {
        serve(listener, front, handle, serve_config, async {
            let _ = stopped.await;
        })
        .await
        .expect("served");
    });
    Running { addr, stop, server }
}

impl Running {
    async fn finish(self) {
        self.stop.send(()).expect("stop");
        tokio::time::timeout(Duration::from_secs(5), self.server)
            .await
            .expect("drained in time")
            .expect("joined");
    }
}

fn grpc_body(message: &[u8]) -> Bytes {
    let mut out = BytesMut::with_capacity(5 + message.len());
    out.put_u8(0);
    out.put_u32(u32::try_from(message.len()).expect("small"));
    out.put_slice(message);
    out.freeze()
}

fn infer_request(model: &str, image: &[u8], plan: &str) -> Vec<u8> {
    let mut r = ModelInferRequest {
        model_name: model.into(),
        ..Default::default()
    };
    for (name, value) in [("image", image), ("plan", plan.as_bytes())] {
        r.inputs.push(InferInputTensor {
            name: name.into(),
            datatype: "BYTES".into(),
            shape: vec![1],
            ..Default::default()
        });
        r.raw_input_contents.push(pack(&[value]));
    }
    r.encode_to_vec()
}

async fn connect(
    addr: std::net::SocketAddr,
    keep_alive: Option<Duration>,
) -> http2::SendRequest<Full<Bytes>> {
    let io = TokioIo::new(TcpStream::connect(addr).await.expect("connect"));
    let mut builder = http2::Builder::new(TokioExecutor::new());
    builder.timer(TokioTimer::new());
    if let Some(every) = keep_alive {
        builder
            .keep_alive_interval(every)
            .keep_alive_timeout(Duration::from_secs(2))
            .keep_alive_while_idle(true);
    }
    let (sender, connection) = builder.handshake(io).await.expect("handshake");
    TokioExecutor::new().execute(async move {
        let _ = connection.await;
    });
    sender
}

struct Reply {
    status: u16,
    grpc_status: Option<String>,
    grpc_message: Option<String>,
    body_len: usize,
    trailers: Option<http::HeaderMap>,
}

async fn call(sender: &mut http2::SendRequest<Full<Bytes>>, message: &[u8]) -> Reply {
    let request = Request::post("http://engine/inference.GRPCInferenceService/ModelInfer")
        .header("content-type", "application/grpc")
        .header("te", "trailers")
        .body(Full::new(grpc_body(message)))
        .expect("request");
    let response = sender.send_request(request).await.expect("a response");
    let (parts, body) = response.into_parts();
    let header = |name: &str| {
        parts
            .headers
            .get(name)
            .map(|v| v.to_str().expect("text").to_owned())
    };
    let collected = body.collect().await.expect("the body ends");
    let trailers = collected.trailers().cloned();
    Reply {
        status: parts.status.as_u16(),
        grpc_status: header("grpc-status"),
        grpc_message: header("grpc-message"),
        body_len: collected.to_bytes().len(),
        trailers,
    }
}

fn percent_decoded(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 3 <= bytes.len()
            && let Ok(v) = u8::from_str_radix(&text[i + 1..i + 3], 16)
        {
            out.push(v);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).expect("UTF-8")
}

#[tokio::test]
async fn an_error_answered_by_the_front_is_one_headers_frame_with_the_status_in_it() {
    let engine = running(64).await;
    let mut sender = connect(engine.addr, None).await;
    let cases: [(Vec<u8>, &str, &str); 3] = [
        (
            infer_request("nope", b"x", FRAME),
            "5",
            "unknown model 'nope'",
        ),
        (infer_request("frame_engine", b"x", "{"), "3", "bad_json"),
        (
            infer_request("frame_engine", b"stub:fail=internal", FRAME),
            "13",
            "frame engine: stub failure",
        ),
    ];
    for (message, status, text) in cases {
        let reply = call(&mut sender, &message).await;
        assert_eq!(
            reply.status, 200,
            "gRPC answers HTTP 200 and carries its own status"
        );
        assert_eq!(reply.grpc_status.as_deref(), Some(status), "{text}");
        assert!(
            reply.trailers.is_none(),
            "trailers-only: no trailing frame after the headers"
        );
        assert_eq!(reply.body_len, 0, "and no message");
        let message = percent_decoded(reply.grpc_message.as_deref().expect("a message"));
        assert!(message.starts_with(text), "{message}");
    }
    engine.finish().await;
}

#[tokio::test]
async fn an_answer_carries_its_status_in_trailers() {
    let engine = running(64).await;
    let mut sender = connect(engine.addr, None).await;
    let reply = call(&mut sender, &infer_request("frame_engine", b"x", FRAME)).await;
    assert_eq!(reply.status, 200);
    assert!(
        reply.grpc_status.is_none(),
        "the status follows the message"
    );
    assert!(reply.body_len > 100);
    let trailers = reply.trailers.expect("trailers");
    assert_eq!(
        trailers
            .get("grpc-status")
            .map(|v| v.to_str().expect("text")),
        Some("0")
    );
    engine.finish().await;
}

#[tokio::test]
async fn a_message_over_the_limit_is_resource_exhausted_not_out_of_range() {
    let engine = running(1).await;
    let mut sender = connect(engine.addr, None).await;
    let big = infer_request("frame_engine", &vec![0u8; 2 * 1024 * 1024], FRAME);
    let reply = call(&mut sender, &big).await;
    assert_eq!(reply.grpc_status.as_deref(), Some("8"));
    assert!(
        percent_decoded(reply.grpc_message.as_deref().expect("a message")).contains("too large")
    );
    let small = call(&mut sender, &infer_request("frame_engine", b"x", FRAME)).await;
    assert_eq!(
        small
            .trailers
            .expect("trailers")
            .get("grpc-status")
            .map(|v| v.to_str().expect("text")),
        Some("0")
    );
    engine.finish().await;
}

#[tokio::test]
async fn an_idle_connection_that_pings_is_not_sent_away() {
    let engine = running(64).await;
    let mut sender = connect(engine.addr, Some(Duration::from_secs(1))).await;
    let first = call(&mut sender, &infer_request("frame_engine", b"x", FRAME)).await;
    assert!(first.trailers.is_some());
    // five seconds idle at one ping a second: a server with a ping policy would have sent
    // GOAWAY after its second ping without data
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert!(!sender.is_closed(), "the connection was closed");
    let again = call(&mut sender, &infer_request("frame_engine", b"x", FRAME)).await;
    assert!(again.trailers.is_some());
    engine.finish().await;
}
