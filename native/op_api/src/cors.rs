//! CORS with the rule of `src/core/cors.py`: a browser on `localhost`, `127.0.0.1`,
//! `host.docker.internal` or a private-network address (RFC 1918) may call the API with
//! credentials, from any port.
//!
//! Two places apply it, never both to one answer:
//!
//! - The routes Rust owns get tower-http's [`layer`], which also answers their preflight requests.
//! - An answer the front builds itself (the 503 for an upstream that is down or a shed request,
//!   the 502, the 400, the 500 of a panic) gets [`on_front_answers`]: Starlette's middleware
//!   wraps every answer of the Python app, errors included, and a browser that cannot read the
//!   status and `Retry-After` of such an answer cannot back off as the frontend does.
//!
//! An answer forwarded from the Python API is left alone: it carries its own headers, and two
//! `Access-Control-Allow-Origin` headers make a browser refuse the response.

use std::time::Duration;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::Response;
use http::HeaderMap;
use http::header::{
    ACCESS_CONTROL_ALLOW_CREDENTIALS, ACCESS_CONTROL_ALLOW_ORIGIN, ACCESS_CONTROL_EXPOSE_HEADERS,
    HeaderName, HeaderValue, ORIGIN, VARY,
};
use regex::Regex;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

use crate::errors::FrontAnswer;

/// The origins allowed, as `src/core/cors.py` writes them (`allow_origin_regex`).
const ORIGIN_PATTERN: &str = concat!(
    r"^https?://(localhost|127\.0\.0\.1|host\.docker\.internal",
    r"|192\.168\.\d+\.\d+",
    r"|10\.\d+\.\d+\.\d+",
    r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+",
    r")(:\d+)?$"
);

/// Seconds a browser may keep a preflight answer (Starlette's default).
const PREFLIGHT_MAX_AGE_S: u64 = 600;

/// The headers a browser script may read (`expose_headers` of `src/core/cors.py`).
const EXPOSED: [&str; 2] = ["x-request-id", "x-process-time"];

/// `EXPOSED` as Starlette writes it into `Access-Control-Expose-Headers`.
const EXPOSED_LIST: &str = "X-Request-ID, X-Process-Time";

/// The compiled origin rule.
///
/// # Errors
///
/// The regex error when the pattern does not compile, which a test rules out.
pub fn origin_pattern() -> Result<Regex, regex::Error> {
    Regex::new(ORIGIN_PATTERN)
}

/// The CORS layer for the routes Rust owns.
///
/// Methods and requested headers are echoed back, which is what Starlette does for
/// `allow_methods=['*']` and `allow_headers=['*']` with credentials (the wildcard is not allowed
/// with credentials, and tower-http refuses to build that combination).
pub fn layer(pattern: Regex) -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin: &HeaderValue, _| {
            origin.to_str().is_ok_and(|o| pattern.is_match(o))
        }))
        .allow_credentials(true)
        .allow_methods(AllowMethods::mirror_request())
        .allow_headers(AllowHeaders::mirror_request())
        .expose_headers(EXPOSED.map(HeaderName::from_static))
        .max_age(Duration::from_secs(PREFLIGHT_MAX_AGE_S))
}

/// Give an answer the front built itself the headers Starlette's `CORSMiddleware` gives a simple
/// response of the Python app: with an `Origin` on the request, credentials and the exposed
/// headers, and for an allowed origin that origin and `Vary: Origin`. An answer from the Python
/// API, and one that already carries CORS headers, is returned as it is.
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
    if !built_here || has_cors(response.headers()) {
        return response;
    }
    let allowed = origin.to_str().is_ok_and(|o| pattern.is_match(o));
    let headers = response.headers_mut();
    headers.insert(
        ACCESS_CONTROL_ALLOW_CREDENTIALS,
        HeaderValue::from_static("true"),
    );
    headers.insert(
        ACCESS_CONTROL_EXPOSE_HEADERS,
        HeaderValue::from_static(EXPOSED_LIST),
    );
    if allowed {
        headers.insert(ACCESS_CONTROL_ALLOW_ORIGIN, origin);
    }
    append_vary_origin(headers);
    response
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
