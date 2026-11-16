//! The seam between the receiver and the ingest pipeline that comes after it.
//!
//! The receiver lands each photo of a request, names it by content and hands it to a
//! [`PartSink`] through a bounded channel; the pipeline's slice implements the trait. A request
//! goes through three calls:
//!
//! 1. [`PartSink::accept`] once per photo, in photo order, as soon as the photo is landed and its
//!    identifier is known. It takes responsibility for the photo and may start work that needs
//!    nothing from the rest of the request (the engine call, capture, the mover). It resolves
//!    when the sink has taken the photo, and may wait for the sink's own room: while it waits the
//!    channel fills and the receiver stops reading the body. The [`WorkPermit`] in the part is
//!    the photo's share of the admission bounds; the sink drops it when its work on the photo is
//!    done, **not** when the request ends, because the request may still be waiting for the
//!    room that permit holds.
//! 2. [`PartSink::finish`] once, after every photo was accepted and the request passed all its
//!    checks. It gets the form fields that may have arrived after the photos (`source`, `run_id`),
//!    commits, and returns one result per accepted photo.
//! 3. [`PartSink::abort`] once instead of `finish` when the request is refused or its client is
//!    gone after at least one photo was accepted. The sink drops what it holds for the upload and
//!    does not commit; a sink that commits a photo before `finish` accepts that an abort does not
//!    undo it. In-flight `accept` futures may be cancelled before `abort` is called.
//!
//! A landing file has one owner. The receiver owns a photo's file until `accept` takes it; from
//! then the sink owns it and removes it. The receiver still removes the files of a request that
//! is aborted, so a sink that forgets leaves nothing behind either; removing a file that is
//! already gone is not an error.

use std::future::Future;

use op_storage::keys::ProjectId;
use op_storage::landing::LandedPhoto;
use uuid::Uuid;

use super::wire::SinkBatch;
use crate::admission::WorkPermit;

/// One photo, landed and named by content.
#[derive(Debug)]
pub struct ReceivedPart {
    /// The request's id; the folder its parts landed in.
    pub upload_id: Uuid,
    /// The project the upload is for.
    pub project: ProjectId,
    /// The photo's position among the request's photos, from 0.
    pub index: usize,
    /// The client's identifier for the photo: `image_paths[index]`, or the default.
    pub identifier: String,
    /// The extension the photo is stored under, lower case with the dot.
    pub extension: String,
    /// The `filename` of the part.
    pub filename: Option<String>,
    /// The landed file. Whoever holds the part owns it.
    pub photo: LandedPhoto,
    /// The photo's share of the admission bounds, released when dropped.
    pub permit: WorkPermit,
}

/// What the sink needs from the end of the request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UploadEnd {
    /// The request's id.
    pub upload_id: Uuid,
    /// The project the upload is for.
    pub project: ProjectId,
    /// The provenance tag for every photo (`upload` when the form has none).
    pub source: String,
    /// The tag of the call, recorded as the ingest run id.
    pub run_id: Option<String>,
}

/// Why the sink could not take or finish the photos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SinkError {
    /// Not now, and nothing was kept: the client may resend. The route answers 503 with
    /// `Retry-After`.
    Unavailable {
        /// What to tell the client.
        detail: String,
        /// The stable machine code.
        code: String,
    },
    /// The sink failed. The route answers 500; the message is logged, not sent.
    Failed(String),
}

/// The ingest pipeline's side of the receiver.
pub trait PartSink: Send + Sync + 'static {
    /// Take responsibility for one photo.
    ///
    /// # Errors
    ///
    /// [`SinkError`] when the sink cannot take the photo; the request is refused.
    fn accept(&self, part: ReceivedPart) -> impl Future<Output = Result<(), SinkError>> + Send;

    /// Commit the accepted photos and return their results.
    ///
    /// # Errors
    ///
    /// [`SinkError`] when the batch cannot be committed; the request is refused.
    fn finish(&self, end: UploadEnd) -> impl Future<Output = Result<SinkBatch, SinkError>> + Send;

    /// The request is refused or gone: drop what is held for `upload_id`.
    fn abort(&self, upload_id: Uuid) -> impl Future<Output = ()> + Send;
}
