//! The routes Rust owns and the layers around every request: request id, CORS, metrics, the
//! concurrency limit and a panicking handler.

use std::sync::atomic::Ordering;

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
async fn a_wrong_method_on_an_owned_route_answers_what_fastapi_answers_and_is_not_forwarded() {
    let (stub, control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    // FastAPI: 405, {"detail": "Method Not Allowed"}, allow: GET, for every method but GET,
    // HEAD included (an APIRoute from `@router.get` does not add HEAD).
    for path in ["/live", "/metrics/front"] {
        for method in ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"] {
            let answer = send(request(&front, method, path).body(Body::empty()).unwrap()).await;
            assert_eq!(
                answer.status,
                StatusCode::METHOD_NOT_ALLOWED,
                "{method} {path}"
            );
            assert_eq!(answer.headers["allow"], "GET", "{method} {path}");
            if method != "HEAD" {
                assert_eq!(
                    answer.json(),
                    serde_json::json!({"detail": "Method Not Allowed"}),
                    "{method} {path}"
                );
            }
        }
    }
    assert_eq!(control.requests.load(Ordering::SeqCst), 0);
    front.stop().await;
}

#[tokio::test]
async fn the_fronts_own_series_are_on_the_front_path_and_hold_the_owned_routes_only() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    get_path(&front, "/live").await;
    get_path(&front, "/echo").await;
    let answer = get_path(&front, "/metrics/front").await;
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
async fn metrics_on_the_public_path_are_the_python_apis_forwarded_like_any_other_path() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = get_path(&front, "/metrics").await;
    assert_eq!(answer.status, StatusCode::OK);
    let seen = answer.json();
    assert_eq!(
        seen["target"], "/metrics",
        "the Python API answered: {seen}"
    );
    assert_eq!(answer.headers["x-from-upstream"], "yes");
    front.stop().await;
}

#[tokio::test]
async fn a_request_without_an_id_gets_one_in_pythons_format_the_upstream_sees_and_the_answer_carries()
 {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    let answer = get_path(&front, "/echo").await;
    let id = answer.headers["x-request-id"].to_str().unwrap().to_owned();
    // `uuid4().hex` of asgi-correlation-id: 32 lowercase hex digits, no dashes.
    assert!(
        id.len() == 32 && id.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')),
        "{id}"
    );
    assert_eq!(answer.json()["headers"]["x-request-id"][0], id);
    front.stop().await;
}

#[tokio::test]
async fn the_safe_id_rule_is_pythons_first_value_checked_and_replaced_when_it_fails() {
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
    for bad in ["has space", "semi;colon", "", &"a".repeat(129)] {
        let replaced = send(
            request(&front, "GET", "/echo")
                .header("x-request-id", bad)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        let id = replaced.headers["x-request-id"].to_str().unwrap();
        assert_ne!(id, bad);
        assert_eq!(id.len(), 32, "{bad:?} -> {id}");
        assert_eq!(
            replaced.json()["headers"]["x-request-id"],
            serde_json::json!([id])
        );
    }
    // Starlette's Headers.get takes the first value: a safe first one is kept, and the upstream
    // is shown that one only.
    let first_safe = send(
        request(&front, "GET", "/echo")
            .header("x-request-id", "one")
            .header("x-request-id", "two words")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(first_safe.headers["x-request-id"], "one");
    assert_eq!(
        first_safe.json()["headers"]["x-request-id"],
        serde_json::json!(["one"])
    );
    let first_unsafe = send(
        request(&front, "GET", "/echo")
            .header("x-request-id", "bad id")
            .header("x-request-id", "two")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    let id = first_unsafe.headers["x-request-id"].to_str().unwrap();
    assert!(id != "two" && id.len() == 32, "{id}");
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
async fn a_preflight_is_answered_here_as_starlettes_cors_middleware_answers_it() {
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
    assert_eq!(answer.status, StatusCode::OK);
    assert_eq!(&answer.body[..], b"OK");
    assert_eq!(answer.headers["content-type"], "text/plain; charset=utf-8");
    assert_eq!(
        answer.headers["access-control-allow-origin"],
        "http://localhost:5184"
    );
    assert_eq!(answer.headers["access-control-allow-credentials"], "true");
    // allow_methods=['*'] is every method Starlette lists; allow_headers=['*'] mirrors the request.
    assert_eq!(
        answer.headers["access-control-allow-methods"],
        "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT, QUERY"
    );
    assert_eq!(
        answer.headers["access-control-allow-headers"],
        "x-custom, content-type"
    );
    assert_eq!(answer.headers["access-control-max-age"], "600");
    assert_eq!(
        answer.headers["vary"],
        "Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Access-Control-Request-Private-Network"
    );
    front.stop().await;
}

#[tokio::test]
async fn a_preflight_that_fails_is_a_400_naming_what_failed() {
    let (stub, _control) = start_stub().await;
    let front = start_front(stub, &[]).await;
    for (origin, method, text) in [
        ("https://evil.example", "GET", "Disallowed CORS origin"),
        ("http://localhost:5184", "FOO", "Disallowed CORS method"),
        (
            "https://evil.example",
            "FOO",
            "Disallowed CORS origin, method",
        ),
    ] {
        let answer = send(
            request(&front, "OPTIONS", "/live")
                .header("origin", origin)
                .header("access-control-request-method", method)
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{origin} {method}");
        assert_eq!(&answer.body[..], text.as_bytes());
    }
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
    let shed = get_path(&front, "/echo").await;
    assert_eq!(shed.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(shed.headers["retry-after"], "1");
    assert_eq!(
        shed.json()["request_id"],
        shed.headers["x-request-id"].to_str().unwrap()
    );
    control.hold.add_permits(1);
    let finished = first.await.unwrap();
    assert_eq!(finished.status, StatusCode::OK);
    let metrics = get_path(&front, "/metrics/front").await;
    let text = String::from_utf8_lossy(&metrics.body);
    assert!(text.contains("op_api_requests_shed_total 1"), "{text}");
    front.stop().await;
}

#[tokio::test]
async fn a_saturated_front_still_answers_its_health_probe_and_its_metrics() {
    let (stub, control) = start_stub().await;
    let front = start_front(stub, &[("OP_API_CONCURRENCY_LIMIT", "1")]).await;
    let first = tokio::spawn(send(
        request(&front, "GET", "/hold").body(Body::empty()).unwrap(),
    ));
    control.held.acquire().await.unwrap().forget();
    assert_eq!(
        get_path(&front, "/echo").await.status,
        StatusCode::SERVICE_UNAVAILABLE,
        "the limit is reached"
    );
    assert_eq!(get_path(&front, "/live").await.status, StatusCode::OK);
    assert_eq!(
        get_path(&front, "/metrics/front").await.status,
        StatusCode::OK
    );
    control.hold.add_permits(1);
    first.await.unwrap();
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
    let router = middleware::common(
        Router::new()
            .route("/boom", get(boom))
            .route("/fine", get(|| async { "fine" }))
            .with_state(state.clone()),
        &op_api::cors::origin_pattern().unwrap(),
    );
    let front = start_front_with(settings, Some(router)).await;
    let broken = send(
        request(&front, "GET", "/boom")
            .header("x-request-id", "panic-id-1")
            .body(Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(broken.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        broken.json(),
        serde_json::json!({"detail": "Internal server error", "request_id": "panic-id-1"})
    );
    assert_eq!(broken.headers["x-request-id"], "panic-id-1");
    let generated = get_path(&front, "/boom").await;
    assert_eq!(
        generated.json()["request_id"],
        generated.headers["x-request-id"].to_str().unwrap()
    );
    assert_eq!(get_path(&front, "/fine").await.status, StatusCode::OK);
    front.stop().await;
}
