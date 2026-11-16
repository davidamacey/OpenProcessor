//! The routes Rust owns and the layers around every request: request id, CORS, metrics, the
//! concurrency limit and a panicking handler.

use axum::Router;
use axum::body::Body;
use axum::routing::get;
use http::StatusCode;
use op_api::middleware;
use op_api::state::AppState;

use crate::support::{
    get_path, request, send, settings, start_front, start_front_with, start_stub,
};

#[tokio::test]
async fn live_answers_alive_without_asking_the_upstream() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = get_path(&front, "/live").await;
    assert_eq!(answer.status, StatusCode::OK);
    assert_eq!(answer.json(), serde_json::json!({"status": "alive"}));
    front.stop().await;
}

#[tokio::test]
async fn a_wrong_method_on_an_owned_route_is_refused_here_and_not_forwarded() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = send(
        request(&front, "POST", "/live")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(answer.status, StatusCode::METHOD_NOT_ALLOWED);
    front.stop().await;
}

#[tokio::test]
async fn metrics_hold_the_owned_routes_and_not_the_forwarded_ones() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    get_path(&front, "/live").await;
    get_path(&front, "/echo").await;
    let answer = get_path(&front, "/metrics").await;
    assert_eq!(
        answer.headers["content-type"],
        "application/openmetrics-text; version=1.0.0; charset=utf-8"
    );
    let text = String::from_utf8(answer.body.to_vec()).unwrap();
    assert!(
        text.contains(
            "http_request_duration_seconds_count{method=\"GET\",route=\"/live\",status=\"200\"} 1"
        ),
        "{text}"
    );
    assert!(
        !text.contains("route=\"/echo\""),
        "a forwarded request is the Python API's to count: {text}"
    );
    front.stop().await;
}

#[tokio::test]
async fn a_request_without_an_id_gets_one_the_upstream_sees_and_the_answer_carries() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = get_path(&front, "/echo").await;
    let id = answer.headers["x-request-id"].to_str().unwrap().to_owned();
    assert!(!id.is_empty());
    assert_eq!(answer.json()["headers"]["x-request-id"][0], id);
    front.stop().await;
}

#[tokio::test]
async fn a_safe_id_is_kept_and_an_unsafe_or_repeated_one_is_replaced() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let kept = send(
        request(&front, "GET", "/echo")
            .header("x-request-id", "client-id_1.2")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(kept.headers["x-request-id"], "client-id_1.2");
    assert_eq!(kept.json()["headers"]["x-request-id"][0], "client-id_1.2");
    for bad in ["has space", "semi;colon", &"a".repeat(129)] {
        let replaced = send(
            request(&front, "GET", "/echo")
                .header("x-request-id", bad)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_ne!(replaced.headers["x-request-id"], bad);
    }
    let twice = send(
        request(&front, "GET", "/echo")
            .header("x-request-id", "one")
            .header("x-request-id", "two")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    let id = twice.headers["x-request-id"].to_str().unwrap();
    assert!(id != "one" && id != "two", "{id}");
    front.stop().await;
}

#[tokio::test]
async fn an_allowed_origin_gets_credentials_and_the_exposed_headers_on_an_owned_route() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    for origin in [
        "http://localhost:5184",
        "https://127.0.0.1",
        "http://host.docker.internal:4603",
        "http://192.168.1.20:5184",
        "http://10.0.0.5",
        "http://172.16.3.4:80",
        "http://172.31.255.255:1",
    ] {
        let answer = send(
            request(&front, "GET", "/live")
                .header("origin", origin)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
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
    }
    front.stop().await;
}

#[tokio::test]
async fn an_origin_outside_the_private_networks_gets_no_cors_headers() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    for origin in [
        "https://evil.example",
        "http://172.32.0.1",
        "http://172.15.0.1",
        "http://192.168.1.20.evil.example",
        "http://localhost.evil.example",
        "ftp://localhost",
    ] {
        let answer = send(
            request(&front, "GET", "/live")
                .header("origin", origin)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert!(
            answer.headers.get("access-control-allow-origin").is_none(),
            "{origin}"
        );
    }
    front.stop().await;
}

#[tokio::test]
async fn a_preflight_is_answered_here_with_the_requested_method_and_headers_echoed() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = send(
        request(&front, "OPTIONS", "/live")
            .header("origin", "http://localhost:5184")
            .header("access-control-request-method", "PUT")
            .header("access-control-request-headers", "x-custom, content-type")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert!(answer.status.is_success(), "{}", answer.status);
    assert_eq!(
        answer.headers["access-control-allow-origin"],
        "http://localhost:5184"
    );
    assert_eq!(answer.headers["access-control-allow-methods"], "PUT");
    assert_eq!(
        answer.headers["access-control-allow-headers"],
        "x-custom, content-type"
    );
    assert_eq!(answer.headers["access-control-max-age"], "600");
    front.stop().await;
}

#[tokio::test]
async fn a_forwarded_answer_carries_only_the_cors_headers_the_python_api_gave_it() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = send(
        request(&front, "GET", "/echo")
            .header("origin", "http://localhost:5184")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert!(answer.headers.get("access-control-allow-origin").is_none());
    front.stop().await;
}

#[tokio::test]
async fn the_request_over_the_limit_is_a_503_at_once_and_the_one_in_flight_finishes() {
    let (stub, control) = start_stub().await;
    let front = start_front(stub, &[("OP_API_CONCURRENCY_LIMIT", "1")]).await;
    let first = {
        let request = request(&front, "GET", "/hold").body(Body::empty()).unwrap();
        tokio::spawn(send(request))
    };
    control.held.acquire().await.unwrap().forget();
    let shed = get_path(&front, "/live").await;
    assert_eq!(shed.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(shed.headers["retry-after"], "1");
    assert_eq!(
        shed.json()["request_id"],
        shed.headers["x-request-id"].to_str().unwrap()
    );
    control.hold.add_permits(1);
    let finished = first.await.unwrap();
    assert_eq!(finished.status, StatusCode::OK);
    let metrics = get_path(&front, "/metrics").await;
    let text = String::from_utf8_lossy(&metrics.body);
    assert!(text.contains("op_api_requests_shed_total 1"), "{text}");
    front.stop().await;
}

async fn boom() -> &'static str {
    panic!("handler bug")
}

#[tokio::test]
async fn a_handler_that_panics_is_a_500_with_the_error_body_and_the_server_goes_on() {
    let (stub, _control) = start_stub().await;
    let settings = settings(stub, &[]);
    let state = AppState::new(&settings);
    let router = middleware::apply(
        Router::new()
            .route("/boom", get(boom))
            .route("/fine", get(|| async { "fine" }))
            .with_state(state.clone()),
        &state,
        settings.concurrency_limit,
    );
    let front = start_front_with(settings, Some(router)).await;
    let broken = get_path(&front, "/boom").await;
    assert_eq!(broken.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        broken.json(),
        serde_json::json!({"detail": "Internal server error"})
    );
    assert!(broken.headers.contains_key("x-request-id"));
    assert_eq!(get_path(&front, "/fine").await.status, StatusCode::OK);
    front.stop().await;
}
