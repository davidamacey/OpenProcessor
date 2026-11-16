//! CORS with the rule and the answers of Starlette's `CORSMiddleware` as `src/core/cors.py`
//! configures it: a browser on `localhost`, `127.0.0.1`, `host.docker.internal` or a
//! private-network address (RFC 1918) may call the API with credentials, from any port.
//!
//! Two places apply it, never both to one answer:
//!
//! - The routes Rust owns get [`on_owned_routes`]: it answers their preflight requests and adds
//!   the simple-response headers to their answers.
//! - An answer the front builds itself (the 503 for an upstream that is down or a shed request,
//!   the 502, the 400, the 500 of a panic) gets [`on_front_answers`]: Starlette's middleware
//!   wraps every answer of the Python app, errors included, and a browser that cannot read the
//!   status and `Retry-After` of such an answer cannot back off as the frontend does.
//!
//! An answer forwarded from the Python API is left alone, and so is a preflight request for one of
//! its routes: the Python API answers it, and two `Access-Control-Allow-Origin` headers make a
//! browser refuse the response.
//!
//! tower-http's `CorsLayer` is not used: it treats every `OPTIONS` request as a preflight, where
//! Starlette (and so FastAPI's 405 for a plain `OPTIONS /live`) does so only with an `Origin` and
//! an `Access-Control-Request-Method`, and its answers differ in the details a browser console
//! shows (the method list, `Vary`, the 400 text).

use axum::body::Body;
use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use http::header::{
    ACCESS_CONTROL_ALLOW_CREDENTIALS, ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS,
    ACCESS_CONTROL_ALLOW_ORIGIN, ACCESS_CONTROL_EXPOSE_HEADERS, ACCESS_CONTROL_MAX_AGE,
    ACCESS_CONTROL_REQUEST_HEADERS, ACCESS_CONTROL_REQUEST_METHOD, CONTENT_TYPE, HeaderValue,
    ORIGIN, VARY,
};
use http::{HeaderMap, Method, StatusCode};
use regex::Regex;

use crate::errors::FrontAnswer;

/// The origins allowed, as `src/core/cors.py` writes them (`allow_origin_regex`).
const ORIGIN_PATTERN: &str = concat!(
    r"^https?://(localhost|127\.0\.0\.1|host\.docker\.internal",
    r"|192\.168\.\d+\.\d+",
    r"|10\.\d+\.\d+\.\d+",
    r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+",
    r")(:\d+)?$"
);

/// What `allow_methods=['*']` stands for in Starlette 1.7 (`ALL_METHODS`).
const ALL_METHODS: [&str; 8] = [
    "DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "QUERY",
];

/// Seconds a browser may keep a preflight answer (Starlette's default).
const PREFLIGHT_MAX_AGE: &str = "600";

/// `Vary` of a preflight answer, as Starlette writes it.
const PREFLIGHT_VARY: &str = "Origin, Access-Control-Request-Method, Access-Control-Request-Headers, Access-Control-Request-Private-Network";

/// `expose_headers` of `src/core/cors.py`, as Starlette joins them.
const EXPOSED: &str = "X-Request-ID, X-Process-Time";

/// The compiled origin rule.
///
/// # Errors
///
/// The regex error when the pattern does not compile, which a test rules out.
pub fn origin_pattern() -> Result<Regex, regex::Error> {
    Regex::new(ORIGIN_PATTERN)
}

fn allowed(pattern: &Regex, origin: &HeaderValue) -> bool {
    origin.to_str().is_ok_and(|o| pattern.is_match(o))
}

/// CORS for the routes Rust owns: a preflight request (`OPTIONS` with `Origin` and
/// `Access-Control-Request-Method`) is answered here, any other request goes on and its answer
/// gets the simple-response headers.
pub async fn on_owned_routes(
    State(pattern): State<Regex>,
    request: Request,
    next: Next,
) -> Response {
    let origin = request.headers().get(ORIGIN).cloned();
    if let Some(origin) = &origin
        && request.method() == Method::OPTIONS
        && request
            .headers()
            .contains_key(ACCESS_CONTROL_REQUEST_METHOD)
    {
        return preflight(&pattern, origin, request.headers());
    }
    let mut response = next.run(request).await;
    if let Some(origin) = origin {
        add_simple_headers(response.headers_mut(), &origin, &pattern);
    }
    response
}

fn preflight(pattern: &Regex, origin: &HeaderValue, request: &HeaderMap) -> Response {
    let mut failures: Vec<&str> = Vec::new();
    let mut builder = Response::builder()
        .header(VARY, PREFLIGHT_VARY)
        .header(ACCESS_CONTROL_ALLOW_METHODS, ALL_METHODS.join(", "))
        .header(ACCESS_CONTROL_MAX_AGE, PREFLIGHT_MAX_AGE)
        .header(ACCESS_CONTROL_ALLOW_CREDENTIALS, "true");
    if allowed(pattern, origin) {
        builder = builder.header(ACCESS_CONTROL_ALLOW_ORIGIN, origin);
    } else {
        failures.push("origin");
    }
    let method_ok = request
        .get(ACCESS_CONTROL_REQUEST_METHOD)
        .and_then(|m| m.to_str().ok())
        .is_some_and(|m| ALL_METHODS.contains(&m));
    if !method_ok {
        failures.push("method");
    }
    if let Some(requested) = request.get(ACCESS_CONTROL_REQUEST_HEADERS) {
        builder = builder.header(ACCESS_CONTROL_ALLOW_HEADERS, requested);
    }
    if request.contains_key("access-control-request-private-network") {
        failures.push("private-network");
    }
    let (status, text) = if failures.is_empty() {
        (StatusCode::OK, "OK".to_owned())
    } else {
        (
            StatusCode::BAD_REQUEST,
            format!("Disallowed CORS {}", failures.join(", ")),
        )
    };
    builder
        .status(status)
        .header(CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from(text))
        .unwrap_or_default()
}

/// Give an answer the front built itself the headers Starlette's `CORSMiddleware` gives a simple
/// response of the Python app. An answer from the Python API, and one that already carries CORS
/// headers (a probe's), is returned as it is.
pub async fn on_front_answers(
    State(pattern): State<Regex>,
    request: Request,
    next: Next,
) -> Response {
    let origin = request.headers().get(ORIGIN).cloned();
    let mut response = next.run(request).await;
    let Some(origin) = origin else {
        return response;
    };
    let built_here = response.extensions().get::<FrontAnswer>().is_some();
    if built_here && !has_cors(response.headers()) {
        add_simple_headers(response.headers_mut(), &origin, &pattern);
    }
    response
}

/// What Starlette adds to a response of a request with an `Origin`: credentials and the exposed
/// headers always, the origin itself for an allowed one, and `Vary: Origin`.
fn add_simple_headers(headers: &mut HeaderMap, origin: &HeaderValue, pattern: &Regex) {
    headers.insert(
        ACCESS_CONTROL_ALLOW_CREDENTIALS,
        HeaderValue::from_static("true"),
    );
    headers.insert(
        ACCESS_CONTROL_EXPOSE_HEADERS,
        HeaderValue::from_static(EXPOSED),
    );
    if allowed(pattern, origin) {
        headers.insert(ACCESS_CONTROL_ALLOW_ORIGIN, origin.clone());
    }
    append_vary_origin(headers);
}

fn has_cors(headers: &HeaderMap) -> bool {
    headers.contains_key(ACCESS_CONTROL_ALLOW_ORIGIN)
        || headers.contains_key(ACCESS_CONTROL_EXPOSE_HEADERS)
}

fn append_vary_origin(headers: &mut HeaderMap) {
    let mut joined: Vec<u8> = Vec::new();
    for value in headers.get_all(VARY) {
        joined.extend_from_slice(value.as_bytes());
        joined.extend_from_slice(b", ");
    }
    joined.extend_from_slice(b"Origin");
    if let Ok(value) = HeaderValue::from_bytes(&joined) {
        headers.insert(VARY, value);
    }
}
