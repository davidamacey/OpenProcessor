//! The CORS layers held to Starlette's `CORSMiddleware` by the cases
//! `scripts/codegen/cors_cases.py` generates from `src/core/cors.py`: each request is sent over a
//! socket through a router that has the real layers, and status, body and every `access-control-*`
//! and `vary` header must equal what Starlette answered. The expectations are never typed here.

use std::collections::BTreeMap;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::response::Response;
use axum::routing::any;
use http::HeaderValue;
use op_api::cors;
use op_api::errors::FrontAnswer;
use op_api::middleware;
use op_api::state::AppState;
use serde_json::Value;

use crate::support::{Answer, bounded, send, settings, start_front_with, start_stub};

type Seen = BTreeMap<String, Vec<String>>;

fn cases() -> Value {
    serde_json::from_str(include_str!("../../../../contracts/json/cors_cases.json")).unwrap()
}

fn answer_of_the_app(vary: bool, front_built: bool) -> Response {
    let mut builder = Response::builder().header("content-type", "text/plain; charset=utf-8");
    if vary {
        builder = builder.header("vary", "Accept-Encoding");
    }
    let mut response = builder.body(Body::from("app")).unwrap();
    if front_built {
        response.extensions_mut().insert(FrontAnswer);
    }
    response
}

async fn owned(_request: Request) -> Response {
    answer_of_the_app(false, false)
}

async fn owned_vary(_request: Request) -> Response {
    answer_of_the_app(true, false)
}

async fn built(_request: Request) -> Response {
    answer_of_the_app(false, true)
}

async fn built_vary(_request: Request) -> Response {
    answer_of_the_app(true, true)
}

/// `/owned*` has the layer of the routes Rust owns, `/built*` is an answer the front builds; the
/// layer of the front's own answers wraps both, as in `op_api::build`.
fn router(state: &AppState) -> Router {
    let pattern = cors::origin_pattern().unwrap();
    let owned_routes = Router::new()
        .route("/owned", any(owned))
        .route("/owned-vary", any(owned_vary))
        .layer(axum::middleware::from_fn_with_state(
            pattern.clone(),
            cors::on_owned_routes,
        ));
    let built_routes = Router::new()
        .route("/built", any(built))
        .route("/built-vary", any(built_vary));
    middleware::common(
        owned_routes.merge(built_routes).with_state(state.clone()),
        &pattern,
    )
}

/// A header value of a case: each character is one byte.
fn value(text: &str) -> HeaderValue {
    let bytes: Vec<u8> = text
        .chars()
        .map(|c| u8::try_from(c as u32).unwrap())
        .collect();
    HeaderValue::from_bytes(&bytes).unwrap()
}

fn text(bytes: &[u8]) -> String {
    bytes.iter().map(|b| char::from(*b)).collect()
}

fn seen_of(answer: &Answer) -> Seen {
    let mut seen = Seen::new();
    for (name, v) in &answer.headers {
        if name.as_str().starts_with("access-control-") || name == "vary" {
            seen.entry(name.as_str().to_owned())
                .or_default()
                .push(text(v.as_bytes()));
        }
    }
    seen
}

fn expected_of(case: &Value) -> Seen {
    case["answer"]["headers"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, values)| {
            let values = values.as_array().unwrap();
            (
                name.clone(),
                values
                    .iter()
                    .map(|v| v.as_str().unwrap().to_owned())
                    .collect(),
            )
        })
        .collect()
}

fn is_preflight(headers: &serde_json::Map<String, Value>, method: &str) -> bool {
    method == "OPTIONS"
        && headers.contains_key("origin")
        && headers.contains_key("access-control-request-method")
}

#[tokio::test]
async fn every_answer_equals_what_starlettes_middleware_answered() {
    bounded(
        "every_answer_equals_what_starlettes_middleware_answered",
        async {
            let (stub, _control) = start_stub().await;
            let settings = settings(stub, &[]);
            let front = start_front_with(settings.clone(), Some(router(&AppState::new(&settings)))).await;
            let document = cases();
            let mut failures: Vec<String> = Vec::new();
            let mut replayed = 0_usize;
            for case in document["cases"].as_array().unwrap() {
                let request = &case["request"];
                let method = request["method"].as_str().unwrap();
                let headers = request["headers"].as_object().unwrap();
                let vary = case["app_sets_vary"].as_bool().unwrap();
                let mut paths = vec![if vary { "/owned-vary" } else { "/owned" }];
                // A preflight request for a route the Python API owns is forwarded to it, so only
                // a front-built answer to a simple request is a case for the second layer.
                if !is_preflight(headers, method) {
                    paths.push(if vary { "/built-vary" } else { "/built" });
                }
                for path in paths {
                    let mut builder = crate::support::request(&front, method, path);
                    for (name, v) in headers {
                        builder = builder.header(name.as_str(), value(v.as_str().unwrap()));
                    }
                    let got = send(builder.body(Body::empty()).unwrap()).await;
                    replayed += 1;
                    let want_headers = expected_of(case);
                    let want_status = case["answer"]["status"].as_u64().unwrap();
                    let want_body = case["answer"]["body"].as_str().unwrap();
                    if u64::from(got.status.as_u16()) != want_status
                        || text(&got.body) != want_body
                        || seen_of(&got) != want_headers
                    {
                        failures.push(format!(
                            "{path} {method} {headers:?}\n  want {want_status} {want_body:?} {want_headers:?}\n  got  {} {:?} {:?}",
                            got.status,
                            text(&got.body),
                            seen_of(&got)
                        ));
                    }
                }
            }
            front.stop().await;
            assert!(replayed > 1000, "only {replayed} requests replayed");
            assert!(
                failures.is_empty(),
                "{} of {replayed} answers differ from Starlette's; the first ones:\n{}",
                failures.len(),
                failures.iter().take(8).cloned().collect::<Vec<_>>().join("\n")
            );
        },
    )
    .await;
}
