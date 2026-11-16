//! What a request leaves behind when it does not succeed, and who removes it.
//!
//! A [`Scope`] records the landing files of one request. A request that ends in a refusal calls
//! [`Scope::abort`], which cleans up before the answer is sent. A request whose future is dropped
//! (the client is gone, or a layer above gave up) never reaches that call, so the scope's `Drop`
//! hands the same cleanup to a [`TaskTracker`] task: tokio documents the tracker as the way to
//! wait for spawned tasks at graceful shutdown
//! (<https://docs.rs/tokio-util/0.7.20/tokio_util/task/task_tracker/struct.TaskTracker.html>), so
//! the server can wait for these cleanups before it exits. A process that is killed leaves its
//! files for the reconcile tick.

use std::path::PathBuf;
use std::sync::Arc;

use op_storage::landing::{LandedPhoto, Landing};
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use super::sink::PartSink;

/// The cleanup duty of one request.
pub(super) struct Scope<S: PartSink> {
    landing: Landing,
    sink: Arc<S>,
    tracker: TaskTracker,
    upload_id: Uuid,
    photos: Vec<LandedPhoto>,
    handed: bool,
    armed: bool,
}

impl<S: PartSink> Scope<S> {
    pub(super) fn new(
        landing: Landing,
        sink: Arc<S>,
        tracker: TaskTracker,
        upload_id: Uuid,
    ) -> Self {
        Self {
            landing,
            sink,
            tracker,
            upload_id,
            photos: Vec::new(),
            handed: false,
            armed: true,
        }
    }

    /// A photo of this request now has a content name.
    pub(super) fn track(&mut self, photo: LandedPhoto) {
        self.photos.push(photo);
    }

    /// The sink holds (or may hold) a photo of this request, so a failure must tell it.
    pub(super) fn mark_handed(&mut self) {
        self.handed = true;
    }

    /// The request succeeded: the sink owns the photos. Remove the upload folder, which is empty
    /// once every part was renamed or discarded.
    pub(super) async fn complete(mut self) {
        self.armed = false;
        let folder = self.folder();
        // NotFound and NotEmpty are both fine: either nothing was landed, or a part is still there
        // for the reconcile tick.
        let _ = tokio::fs::remove_dir(folder).await;
    }

    /// The request failed: tell the sink, remove every file of the request.
    pub(super) async fn abort(mut self) {
        self.armed = false;
        clean_up(self.take()).await;
    }

    fn folder(&self) -> PathBuf {
        self.landing.root().join(self.upload_id.to_string())
    }

    fn take(&mut self) -> Cleanup<S> {
        Cleanup {
            landing: self.landing.clone(),
            sink: Arc::clone(&self.sink),
            upload_id: self.upload_id,
            photos: std::mem::take(&mut self.photos),
            handed: self.handed,
        }
    }
}

impl<S: PartSink> Drop for Scope<S> {
    fn drop(&mut self) {
        if self.armed {
            let cleanup = self.take();
            self.tracker.spawn(clean_up(cleanup));
        }
    }
}

struct Cleanup<S: PartSink> {
    landing: Landing,
    sink: Arc<S>,
    upload_id: Uuid,
    photos: Vec<LandedPhoto>,
    handed: bool,
}

async fn clean_up<S: PartSink>(cleanup: Cleanup<S>) {
    if cleanup.handed {
        let sink = Arc::clone(&cleanup.sink);
        let id = cleanup.upload_id;
        if let Err(e) = tokio::spawn(async move { sink.abort(id).await }).await {
            tracing::error!(upload_id = %id, error = %e, "the sink's abort panicked");
        }
    }
    for photo in &cleanup.photos {
        if let Err(e) = cleanup.landing.remove(photo).await {
            tracing::warn!(error = %e, "a landing file of a refused upload was not removed");
        }
    }
    let folder = cleanup.landing.root().join(cleanup.upload_id.to_string());
    match tokio::fs::remove_dir_all(&folder).await {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            tracing::warn!(error = %e, folder = %folder.display(), "an upload folder was not removed")
        }
    }
}
