//! CORS for the routes Rust owns, with the rule of `src/core/cors.py`: a browser on `localhost`,
//! `127.0.0.1`, `host.docker.internal` or a private-network address (RFC 1918) may call the API
//! with credentials, from any port.
//!
//! Forwarded requests are not given CORS headers here: the Python API adds its own to its answers,
//! and two `Access-Control-Allow-Origin` headers make a browser refuse the response.

use std::time::Duration;

use http::header::{HeaderName, HeaderValue};
use regex::Regex;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

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

/// The CORS layer for the routes Rust owns.
///
/// Methods and requested headers are echoed back, which is what Starlette does for
/// `allow_methods=['*']` and `allow_headers=['*']` with credentials (the wildcard is not allowed
/// with credentials, and tower-http refuses to build that combination).
///
/// # Errors
///
/// The regex error when the origin pattern does not compile, which a test rules out.
pub fn layer() -> Result<CorsLayer, regex::Error> {
    let pattern = Regex::new(ORIGIN_PATTERN)?;
    Ok(CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(move |origin: &HeaderValue, _| {
            origin.to_str().is_ok_and(|o| pattern.is_match(o))
        }))
        .allow_credentials(true)
        .allow_methods(AllowMethods::mirror_request())
        .allow_headers(AllowHeaders::mirror_request())
        .expose_headers(EXPOSED.map(HeaderName::from_static))
        .max_age(Duration::from_secs(PREFLIGHT_MAX_AGE_S)))
}
