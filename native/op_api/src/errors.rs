//! The error bodies of the API: the shapes FastAPI answered with, which the frontend reads
//! (`frontend/src/lib/api_errors.ts`) and the contract keeps (plan 18, section 2.4).
//!
//! | Shape | Body |
//! |---|---|
//! | [`ApiError::detail`] | `{"detail": "<text>"}` |
//! | [`ApiError::coded`] | `{"detail": {"error": "<code>", "message": "<text>"}}` |
//! | [`ApiError::validation`] | `{"detail": [{"type", "loc", "msg"}, ...]}`, status 422 |
//!
//! [`ApiError::with_request_id`] adds the top-level `request_id` the Python handlers add to a 5xx,
//! and [`ApiError::with_retry_after`] the `Retry-After` header a retryable 503 carries.

use axum::Json;
use axum::response::{IntoResponse, Response};
use http::StatusCode;
use http::header::{HeaderValue, RETRY_AFTER};
use serde_json::{Map, Value, json};

/// Set on every response the front builds itself, as opposed to one the Python API gave and the
/// front forwards. Layers that must treat the two differently (CORS: the Python API's answers
/// already carry their headers) read it.
#[derive(Debug, Clone, Copy)]
pub struct FrontAnswer;

/// One segment of the `loc` of a validation issue: a field name or a list index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Loc {
    /// An object key, or the part of the request (`body`, `query`, `path`).
    Field(String),
    /// A position in a list.
    Index(usize),
}

impl Loc {
    fn to_json(&self) -> Value {
        match self {
            Self::Field(name) => Value::String(name.clone()),
            Self::Index(i) => json!(i),
        }
    }
}

/// One entry of a 422 body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    /// Where the problem is, outermost first.
    pub loc: Vec<Loc>,
    /// What is wrong, in words. The text may differ from FastAPI's, except where the contract says
    /// it is word for word (a pattern mismatch).
    pub msg: String,
    /// The machine name of the problem (`missing`, `string_pattern_mismatch`, ...).
    pub kind: String,
}

/// An error answer: a status, a JSON body of one of the shapes above, and optionally
/// `Retry-After`.
#[derive(Debug, Clone, PartialEq)]
pub struct ApiError {
    status: StatusCode,
    body: Map<String, Value>,
    retry_after: Option<u32>,
}

impl ApiError {
    fn new(status: StatusCode, detail: Value) -> Self {
        let mut body = Map::new();
        body.insert("detail".into(), detail);
        Self {
            status,
            body,
            retry_after: None,
        }
    }

    /// `{"detail": "<text>"}`.
    #[must_use]
    pub fn detail(status: StatusCode, text: impl Into<String>) -> Self {
        Self::new(status, Value::String(text.into()))
    }

    /// `{"detail": {"error": "<code>", "message": "<text>"}}`.
    #[must_use]
    pub fn coded(status: StatusCode, code: &str, message: impl Into<String>) -> Self {
        Self::new(status, json!({ "error": code, "message": message.into() }))
    }

    /// A 422 with one entry per issue.
    #[must_use]
    pub fn validation(issues: &[ValidationIssue]) -> Self {
        let list = issues
            .iter()
            .map(|i| {
                json!({
                    "type": i.kind,
                    "loc": i.loc.iter().map(Loc::to_json).collect::<Vec<_>>(),
                    "msg": i.msg,
                })
            })
            .collect();
        Self::new(StatusCode::UNPROCESSABLE_ENTITY, Value::Array(list))
    }

    /// Add the request id to the body, as the Python handlers do for a 5xx.
    #[must_use]
    pub fn with_request_id(mut self, id: &str) -> Self {
        self.body
            .insert("request_id".into(), Value::String(id.into()));
        self
    }

    /// Add `Retry-After: <seconds>`.
    #[must_use]
    pub fn with_retry_after(mut self, seconds: u32) -> Self {
        self.retry_after = Some(seconds);
        self
    }

    /// The status of the answer.
    #[must_use]
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// The JSON body of the answer.
    #[must_use]
    pub fn body(&self) -> Value {
        Value::Object(self.body.clone())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut response = (self.status, Json(Value::Object(self.body))).into_response();
        response.extensions_mut().insert(FrontAnswer);
        if let Some(seconds) = self.retry_after {
            response
                .headers_mut()
                .insert(RETRY_AFTER, HeaderValue::from(seconds));
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http_body_util::BodyExt;

    async fn read(error: ApiError) -> (StatusCode, Option<String>, Value) {
        let response = error.into_response();
        let status = response.status();
        let retry = response
            .headers()
            .get(RETRY_AFTER)
            .map(|v| v.to_str().unwrap().to_owned());
        let content_type = response.headers()["content-type"]
            .to_str()
            .unwrap()
            .to_owned();
        assert_eq!(content_type, "application/json");
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, retry, serde_json::from_slice(&bytes).unwrap())
    }

    #[tokio::test]
    async fn detail_is_a_string_under_detail() {
        let (status, retry, body) =
            read(ApiError::detail(StatusCode::NOT_FOUND, "Not Found")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(retry, None);
        assert_eq!(body, json!({"detail": "Not Found"}));
    }

    #[tokio::test]
    async fn coded_is_an_error_and_message_under_detail() {
        let (status, _, body) = read(ApiError::coded(
            StatusCode::CONFLICT,
            "project_exists",
            "A project with that name exists",
        ))
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(
            body,
            json!({"detail": {"error": "project_exists", "message": "A project with that name exists"}})
        );
    }

    #[tokio::test]
    async fn validation_is_a_422_list_of_type_loc_msg_with_ints_kept_as_numbers() {
        let issue = ValidationIssue {
            loc: vec![
                Loc::Field("body".into()),
                Loc::Field("boxes".into()),
                Loc::Index(2),
            ],
            msg: "Field required".into(),
            kind: "missing".into(),
        };
        let (status, _, body) = read(ApiError::validation(&[issue])).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            body,
            json!({"detail": [{"type": "missing", "loc": ["body", "boxes", 2], "msg": "Field required"}]})
        );
    }

    #[tokio::test]
    async fn a_5xx_carries_the_request_id_and_a_retryable_503_its_retry_after() {
        let error = ApiError::detail(StatusCode::SERVICE_UNAVAILABLE, "Server is busy")
            .with_request_id("abc-1")
            .with_retry_after(5);
        let (status, retry, body) = read(error).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(retry.as_deref(), Some("5"));
        assert_eq!(
            body,
            json!({"detail": "Server is busy", "request_id": "abc-1"})
        );
    }
}
