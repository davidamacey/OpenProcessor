//! `POST /curation/projects/{project}/ingest/upload`: one or many photos in one multipart request,
//! streamed to the landing zone at the client's pace and handed on photo by photo.
//!
//! What the route does, in order (plan 18 section 4.2, steps 1 and 2):
//!
//! 1. A request whose declared length is over the ceiling is refused before its body is read, as
//!    Python's middleware does, and a body that streams past the ceiling is cut by tower-http's
//!    `RequestBodyLimitLayer`
//!    (<https://docs.rs/tower-http/0.7.1/tower_http/limit/index.html>). axum's own 2 MB default
//!    for the `Multipart` extractor is switched off for this route with
//!    `DefaultBodyLimit::disable()`
//!    (<https://docs.rs/axum/0.8.9/axum/extract/struct.DefaultBodyLimit.html>), so tower-http's
//!    limit is the only one. `multer`, which parses the form under axum's `Multipart`, sets no
//!    size limit of its own, so the per-file and per-field caps are enforced here while chunks
//!    stream.
//! 2. `RequestBodyTimeoutLayer` ends a body that stays silent for `OP_UPLOAD_BODY_IDLE_TIMEOUT_S`
//!    (<https://docs.rs/tower-http/0.7.1/tower_http/timeout/struct.RequestBodyTimeoutLayer.html>).
//!    Its deadline restarts at every poll that finds a frame, and it is armed only while the body
//!    is being polled, so a body that arrives slowly is never cut and a pause caused by a full
//!    sink does not count as silence.
//! 3. Each `images` part is streamed with `Field::chunk` into
//!    [`op_storage::landing::Landing::begin_part`], under the photo and byte permits of
//!    [`crate::admission`], with the disk guard checked first. Nothing of a part is kept in memory
//!    beyond the chunk in hand and the writer's buffer.
//! 4. A landed photo is named by content and handed to the [`PartSink`] through a bounded channel
//!    as soon as its identifier is known: at once when `image_paths` came before the photos, at the
//!    end of the body when it comes after them or not at all.
//!
//! A request is all or nothing, as in Python: when it is refused at any point, the sink is told to
//! drop it and every landing file of the request is removed, whatever the reason: a limit, a bad
//! form, a full disk, a client that went away.

mod cleanup;
mod errors;
mod form;
mod handoff;
mod intake;
mod pace;
mod part;
mod refusal;
mod sink;
mod wire;
mod writer;

use std::sync::Arc;

use axum::extract::multipart::MultipartRejection;
use axum::extract::{DefaultBodyLimit, Extension, FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use op_storage::keys::ProjectId;
use op_storage::landing::Landing;
use tokio_util::task::TaskTracker;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::RequestBodyTimeoutLayer;
use uuid::Uuid;

use crate::admission::Admission;
use crate::disk_space::DiskGuard;
use crate::settings::UploadSettings;
use intake::Intake;
use pace::OneChunkPerPoll;

pub use errors::BODY_PARSE_DETAIL;
pub use form::{
    ERROR_KIND_EMPTY, ERROR_KIND_UNSUPPORTED_TYPE, default_identifier, is_accepted,
    parse_image_paths, path_suffix, python_repr, unsupported_message, upload_extension,
};
pub use refusal::{RETRY_AFTER_SECONDS, Refusal};
pub use sink::{PartSink, ReceivedPart, SinkError, UploadEnd};
pub use wire::{
    BatchResponse, BatchStatus, BatchSummary, ImageStatus, IngestImageResponse, SinkBatch, merge,
};

/// The route's path. The mounting slice adds a layer that resolves `{project}` and inserts the
/// resolved [`ProjectId`] as a request extension; a request without it is refused (500), never
/// guessed.
pub const UPLOAD_ROUTE: &str = "/curation/projects/{project}/ingest/upload";

/// The name of the landing volume in the disk guard's 507 answer (Python calls it `data`).
pub const LANDING_VOLUME: &str = "data";

/// What every request of the route shares.
pub struct UploadState<S: PartSink> {
    /// The limits the route enforces.
    pub settings: UploadSettings,
    /// The photo and byte bounds, shared by every request of the process.
    pub admission: Admission,
    /// The landing zone.
    pub landing: Landing,
    /// The disk guard of the landing volume.
    pub disk: DiskGuard,
    /// The pipeline's side.
    pub sink: Arc<S>,
    /// The cleanups of requests that were dropped; wait for it at shutdown.
    pub cleanup: TaskTracker,
}

impl<S: PartSink> UploadState<S> {
    /// The state for `settings`, the landing zone `landing` and the sink `sink`.
    #[must_use]
    pub fn new(settings: UploadSettings, landing: Landing, sink: Arc<S>) -> Arc<Self> {
        Arc::new(Self {
            admission: Admission::new(
                settings.photos_in_work,
                settings.bytes_in_flight,
                settings.admission_wait,
            ),
            disk: DiskGuard::new(landing.clone(), settings.disk, LANDING_VOLUME),
            settings,
            landing,
            sink,
            cleanup: TaskTracker::new(),
        })
    }

    /// Wait until every cleanup of a dropped request has run. Call it after the server stopped
    /// taking requests.
    pub async fn wait_for_cleanup(&self) {
        self.cleanup.close();
        self.cleanup.wait().await;
    }
}

/// The route with its layers, ready to be merged into the served router.
pub fn router<S: PartSink>(state: Arc<UploadState<S>>) -> Router {
    let ceiling = usize::try_from(state.settings.body_ceiling()).unwrap_or(usize::MAX);
    Router::new()
        .route(UPLOAD_ROUTE, post(upload::<S>))
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyTimeoutLayer::new(
            state.settings.body_idle_timeout,
        ))
        .layer(RequestBodyLimitLayer::new(ceiling))
        .layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            declared_length_guard::<S>,
        ))
        .with_state(state)
}

fn request_id(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("-")
        .to_string()
}

fn declared_length(headers: &HeaderMap) -> Option<u64> {
    let raw = headers.get(header::CONTENT_LENGTH)?.to_str().ok()?;
    if raw.is_empty() || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    raw.parse().ok()
}

fn is_multipart(headers: &HeaderMap) -> bool {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("multipart/form-data"))
}

/// `request_exceeds_ceiling` of `src/core/upload_limits.py`: a declared length over the ceiling
/// is refused before the body is read. A form's ceiling is the larger of one file and the accepted
/// bytes, plus the form around them; any other body is held to one file.
async fn declared_length_guard<S: PartSink>(
    State(state): State<Arc<UploadState<S>>>,
    request: Request,
    next: Next,
) -> Response {
    let headers = request.headers();
    let ceiling = if is_multipart(headers) {
        state.settings.body_ceiling()
    } else {
        state.settings.max_file_bytes
    };
    if declared_length(headers).is_some_and(|declared| declared > ceiling) {
        return Refusal::FileTooLarge {
            max_file_size_mb: state.settings.max_file_mb(),
        }
        .into_response(&request_id(headers));
    }
    next.run(request).await
}

async fn upload<S: PartSink>(
    State(state): State<Arc<UploadState<S>>>,
    Extension(project): Extension<ProjectId>,
    request: Request,
) -> Response {
    let (parts, body) = request.into_parts();
    let headers = parts.headers.clone();
    let request = Request::from_parts(parts, OneChunkPerPoll::wrap(body));
    let multipart = Multipart::from_request(request, &()).await;
    match receive(state, project, &headers, multipart).await {
        Ok(answer) => Json(answer).into_response(),
        Err(refusal) => refusal.into_response(&request_id(&headers)),
    }
}

async fn receive<S: PartSink>(
    state: Arc<UploadState<S>>,
    project: ProjectId,
    headers: &HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<BatchResponse, Refusal> {
    state.disk.check().await?;
    let multipart = multipart.map_err(|_| {
        if is_multipart(headers) {
            Refusal::Malformed("Missing boundary in multipart.".to_string())
        } else {
            Refusal::missing_images()
        }
    })?;
    Intake::new(state, project, Uuid::new_v4(), declared_length(headers))
        .run(multipart)
        .await
}
