//! The request id: `x-request-id` in, kept when it is safe, made when it is not, and echoed on the
//! response.
//!
//! The layers are tower-http's (`SetRequestIdLayer` and `PropagateRequestIdLayer`,
//! <https://docs.rs/tower-http/0.7.1/tower_http/request_id/>); they leave an id that is already
//! there alone, so [`drop_unsafe`] runs before them and removes one the Python API would not
//! accept either: `src/core/logging.py` keeps `[A-Za-z0-9._-]{1,128}` because the id is logged and
//! echoed.

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use http::header::HeaderName;

/// The header that carries the id.
pub const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Whether a client-supplied id may be logged and echoed.
#[must_use]
pub fn is_safe(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// Remove an `x-request-id` that is not safe (or is given twice), so the next layer makes a new one.
pub async fn drop_unsafe(mut request: Request, next: Next) -> Response {
    let mut values = request.headers().get_all(&X_REQUEST_ID).iter();
    let safe = match (values.next(), values.next()) {
        (None, _) => true,
        (Some(value), None) => value.to_str().is_ok_and(is_safe),
        (Some(_), Some(_)) => false,
    };
    if !safe {
        request.headers_mut().remove(&X_REQUEST_ID);
    }
    next.run(request).await
}

/// The id of a request that passed [`SetRequestIdLayer`](tower_http::request_id::SetRequestIdLayer).
#[must_use]
pub fn of<B>(request: &http::Request<B>) -> Option<&str> {
    request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|v| v.to_str().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_python_rule_for_a_safe_id() {
        for ok in ["a", "abc-123_x.y", &"a".repeat(128)] {
            assert!(is_safe(ok), "{ok}");
        }
        for bad in ["", "a b", "a/b", "é", "a\n", &"a".repeat(129)] {
            assert!(!is_safe(bad), "{bad:?}");
        }
    }
}
