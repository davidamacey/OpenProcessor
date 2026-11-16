//! Every way the route refuses a request, with the status and JSON body the Python route
//! answers (`src/routers/curation/ingest_upload.py`, `src/core/upload_limits.py`,
//! `src/core/disk_space.py`, `src/core/error_handlers.py`, and the FastAPI and Starlette forms
//! for what they reject before the route runs). Bodies that Python builds from an
//! `HTTPException` carry only `detail`; the ones its own handlers build also carry `request_id`.

use axum::Json;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::disk_space::{DiskError, InsufficientStorage};

/// Seconds a refused client is told to wait; the engine-unavailable answer says the same.
pub const RETRY_AFTER_SECONDS: u32 = 5;

/// A request that is refused as a whole. Nothing of it is kept.
#[derive(Debug)]
pub enum Refusal {
    /// FastAPI's 422 list: the `images` field is missing or holds text, or a text field arrived
    /// as a file.
    Validation(Vec<Value>),
    /// More photos than the per-request limit: 413.
    TooManyImages {
        /// Photos the request carried.
        sent: usize,
        /// The limit.
        limit: usize,
    },
    /// `image_paths` is not JSON: 422. The tail of the message is the JSON parser's own.
    PathsNotJson(String),
    /// `image_paths` is JSON but not a list of non-empty strings: 422.
    PathsNotList,
    /// `image_paths` has a different number of entries than there are photos: 422.
    PathsMismatch {
        /// Entries in the list.
        paths: usize,
        /// Photos sent.
        images: usize,
    },
    /// The accepted photos add up to more than the per-request limit: 413.
    TotalTooLarge {
        /// The limit in bytes.
        limit: u64,
    },
    /// A file, or the whole body, is over the size limit: 413.
    FileTooLarge {
        /// The file cap in MB, named in the answer.
        max_file_size_mb: u64,
    },
    /// The landing volume is too full: 507.
    InsufficientStorage(InsufficientStorage),
    /// The body is not a form the parser accepts: 400.
    Malformed(String),
    /// The body stayed silent for too long: 408.
    BodyIdle {
        /// The limit in seconds.
        seconds: u64,
    },
    /// The part could not be admitted inside the wait: 503 with `Retry-After`.
    Busy,
    /// The sink cannot take the photos now and nothing was kept: 503 with `Retry-After`.
    Unavailable {
        /// What to tell the client.
        detail: String,
        /// The stable machine code.
        code: String,
    },
    /// The server failed: 500. The cause is logged, not sent.
    Internal,
}

impl Refusal {
    /// The `Missing` form of FastAPI's list: no `images` field at all.
    #[must_use]
    pub fn missing_images() -> Self {
        Self::Validation(vec![missing_entry("images")])
    }

    /// The HTTP status of the answer.
    #[must_use]
    pub fn status(&self) -> StatusCode {
        match self {
            Self::Validation(_)
            | Self::PathsNotJson(_)
            | Self::PathsNotList
            | Self::PathsMismatch { .. } => StatusCode::UNPROCESSABLE_ENTITY,
            Self::TooManyImages { .. } | Self::TotalTooLarge { .. } | Self::FileTooLarge { .. } => {
                StatusCode::PAYLOAD_TOO_LARGE
            }
            Self::InsufficientStorage(_) => StatusCode::INSUFFICIENT_STORAGE,
            Self::Malformed(_) => StatusCode::BAD_REQUEST,
            Self::BodyIdle { .. } => StatusCode::REQUEST_TIMEOUT,
            Self::Busy | Self::Unavailable { .. } => StatusCode::SERVICE_UNAVAILABLE,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// The JSON body of the answer.
    #[must_use]
    pub fn body(&self, request_id: &str) -> Value {
        match self {
            Self::Validation(list) => json!({ "detail": list }),
            Self::TooManyImages { sent, limit } => {
                json!({ "detail": format!("{sent} images exceeds the per-request limit of {limit}") })
            }
            Self::PathsNotJson(why) => {
                json!({ "detail": format!("image_paths is not JSON: {why}") })
            }
            Self::PathsNotList => json!({ "detail": "image_paths must be a JSON list of strings" }),
            Self::PathsMismatch { paths, images } => json!({
                "detail": format!("image_paths has {paths} entries but {images} images were sent")
            }),
            Self::TotalTooLarge { limit } => json!({
                "detail": format!("total upload bytes exceeds the per-request limit of {limit} bytes")
            }),
            Self::FileTooLarge { max_file_size_mb } => json!({
                "detail": format!("File too large. Maximum: {max_file_size_mb}MB"),
                "max_file_size_mb": max_file_size_mb,
                "request_id": request_id,
            }),
            Self::InsufficientStorage(full) => json!({
                "detail": {
                    "error": "insufficient_storage",
                    "message": full.to_string(),
                    "volume": full.volume,
                }
            }),
            Self::Malformed(why) => json!({ "detail": why }),
            Self::BodyIdle { seconds } => json!({
                "detail": format!("the upload stopped sending data: no bytes arrived for {seconds} s")
            }),
            Self::Busy => json!({
                "detail": "The server is busy taking uploads; retry shortly",
                "code": "upload_busy",
                "request_id": request_id,
            }),
            Self::Unavailable { detail, code } => {
                json!({ "detail": detail, "code": code, "request_id": request_id })
            }
            Self::Internal => {
                json!({ "detail": "Internal server error", "request_id": request_id })
            }
        }
    }

    /// The answer, with `Retry-After` where a retry is the right reply.
    #[must_use]
    pub fn into_response(self, request_id: &str) -> Response {
        let mut response = (self.status(), Json(self.body(request_id))).into_response();
        if matches!(self, Self::Busy | Self::Unavailable { .. }) {
            response
                .headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from(RETRY_AFTER_SECONDS));
        }
        response
    }
}

impl From<DiskError> for Refusal {
    fn from(error: DiskError) -> Self {
        match error {
            DiskError::Critical(full) => Self::InsufficientStorage(full),
            DiskError::Unmeasurable(e) => {
                tracing::error!(error = %e, "the landing volume could not be measured");
                Self::Internal
            }
        }
    }
}

/// FastAPI's entry for a required field that is absent.
#[must_use]
pub fn missing_entry(field: &str) -> Value {
    json!({ "type": "missing", "loc": ["body", field], "msg": "Field required", "input": null })
}

/// FastAPI's entry for a part named `images` that is text: `ordinal` counts every `images`
/// value of the request, files and text alike.
#[must_use]
pub fn images_not_a_file(ordinal: usize, text: &str) -> Value {
    json!({
        "type": "value_error",
        "loc": ["body", "images", ordinal],
        "msg": "Value error, Expected UploadFile, received: <class 'str'>",
        "input": text,
        "ctx": { "error": {} },
    })
}

/// FastAPI's entry for a file sent where a text field belongs. Python also echoes its own
/// object's `repr` as `input`; the Rust answer leaves that out.
#[must_use]
pub fn text_sent_as_file(field: &str) -> Value {
    json!({
        "type": "string_type",
        "loc": ["body", field],
        "msg": "Input should be a valid string",
    })
}
