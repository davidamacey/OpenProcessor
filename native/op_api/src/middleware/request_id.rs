//! The request id: `x-request-id` in, kept when it is safe, made when it is not, and echoed on the
//! response, as `CorrelationIdMiddleware` of `src/app/factory.py` does it: the first value is the
//! one looked at, it must match `[A-Za-z0-9._-]{1,128}` (`src/core/logging.py`: the id is logged
//! and echoed), and a new one is `uuid4().hex`. tower-http's request-id layers make dashed ids
//! and look at no rule, so this is a small middleware of its own.

use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;
use http::header::{HeaderName, HeaderValue};

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

/// The id of the request, kept as an extension so the layers inside it (the panic catcher, the
/// overload answer) can write it into their bodies.
#[derive(Debug, Clone)]
pub struct RequestId(pub String);

/// An id as `asgi-correlation-id` makes it: `uuid4().hex`, 32 lowercase hex digits.
fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

/// Settle the id of the request the way `CorrelationIdMiddleware` does: the first `x-request-id`
/// value is kept when it is safe, otherwise (absent, empty, unsafe) a new one is made; the request
/// carries that one value; the answer carries it too unless the Python API's answer already does.
pub async fn assign(mut request: Request, next: Next) -> Response {
    let id = request
        .headers()
        .get(&X_REQUEST_ID)
        .and_then(|value| value.to_str().ok())
        .filter(|value| is_safe(value))
        .map_or_else(new_id, str::to_owned);
    if let Ok(value) = HeaderValue::from_str(&id) {
        request.headers_mut().insert(X_REQUEST_ID, value);
    }
    request.extensions_mut().insert(RequestId(id.clone()));
    let mut response = next.run(request).await;
    if !response.headers().contains_key(&X_REQUEST_ID)
        && let Ok(value) = HeaderValue::from_str(&id)
    {
        response.headers_mut().insert(X_REQUEST_ID, value);
    }
    response
}

/// The id of a request that passed [`assign`].
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
