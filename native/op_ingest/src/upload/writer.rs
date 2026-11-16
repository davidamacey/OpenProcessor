//! The task that owns one part's file.
//!
//! A `tokio::fs` call keeps running on the blocking pool when the future that started it is
//! dropped. If the receiver wrote the file itself, a request dropped in the middle of a write, an
//! open or the rename to the content name could therefore race with the cleanup and leave a file
//! behind. So the file belongs to a task that is never dropped half way: the receiver sends it
//! chunks through a channel of two, and a part is complete only when the receiver says so. A
//! channel that closes without that message means the receiver is gone, and the task discards the
//! part after the call it was in. The same task renames the part to its content name and returns
//! the photo through a oneshot channel; when nobody is left to take it, the task removes it. The
//! bounded channel is also the back pressure between the socket and the disk: the receiver
//! cannot read more than two chunks ahead of the file.

use bytes::Bytes;
use op_storage::error::StorageError;
use op_storage::keys::ProjectId;
use op_storage::landing::{LandedPhoto, Landing, PartWriter};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use super::UploadState;
use super::refusal::Refusal;
use super::sink::PartSink;

/// Chunks the receiver may be ahead of the file.
const AHEAD: usize = 2;

enum Msg {
    Chunk(Bytes),
    End,
}

/// What stopped the pump.
pub(super) enum Stop {
    /// The request is refused for this reason; the part is discarded.
    Refused(Refusal),
    /// The writer task ended early; [`PartPipe::end`] says why.
    WriterGone,
}

/// The receiver's end of one part's writer task.
pub(super) struct PartPipe {
    tx: mpsc::Sender<Msg>,
    done: oneshot::Receiver<Result<Option<LandedPhoto>, StorageError>>,
}

impl PartPipe {
    /// Start the task for `writer` on the state's task tracker, so shutdown can wait for it.
    pub(super) fn start<S: PartSink>(
        state: &UploadState<S>,
        project: ProjectId,
        upload_id: Uuid,
        writer: PartWriter,
    ) -> Self {
        let (tx, rx) = mpsc::channel(AHEAD);
        let (done_tx, done) = oneshot::channel();
        state.cleanup.spawn(run(
            state.landing.clone(),
            project,
            upload_id,
            writer,
            rx,
            done_tx,
        ));
        Self { tx, done }
    }

    /// Queue a chunk; waits while the file is two chunks behind.
    pub(super) async fn send(&self, chunk: Bytes) -> Result<(), Stop> {
        self.tx
            .send(Msg::Chunk(chunk))
            .await
            .map_err(|_| Stop::WriterGone)
    }

    /// The part is complete: wait for the file to be flushed and named. `None` for a part that
    /// had no bytes.
    pub(super) async fn end(self) -> Result<Option<LandedPhoto>, StorageError> {
        // a task that already ended has the answer waiting in `done`
        let _ = self.tx.send(Msg::End).await;
        match self.done.await {
            Ok(result) => result,
            Err(_) => Err(StorageError::OutsideRoot {
                path: std::path::PathBuf::new(),
            }),
        }
    }
}

enum Collected {
    /// The receiver said the part is complete; `wrote` tells whether it had any bytes.
    Ended { wrote: bool },
    /// The channel closed without that message: the receiver is gone.
    Abandoned,
    /// A write failed.
    Failed(StorageError),
}

async fn collect(writer: &mut PartWriter, rx: &mut mpsc::Receiver<Msg>) -> Collected {
    let mut wrote = false;
    loop {
        match rx.recv().await {
            Some(Msg::Chunk(chunk)) => {
                wrote = true;
                if let Err(e) = writer.write(&chunk).await {
                    return Collected::Failed(e);
                }
            }
            Some(Msg::End) => return Collected::Ended { wrote },
            None => return Collected::Abandoned,
        }
    }
}

type Done = oneshot::Sender<Result<Option<LandedPhoto>, StorageError>>;

async fn run(
    landing: Landing,
    project: ProjectId,
    upload_id: Uuid,
    mut writer: PartWriter,
    mut rx: mpsc::Receiver<Msg>,
    done: Done,
) {
    let folder = landing.root().join(upload_id.to_string());
    match collect(&mut writer, &mut rx).await {
        Collected::Ended { wrote: true } => name(&landing, project, writer, &folder, done).await,
        Collected::Ended { wrote: false } => {
            discard(writer, &folder).await;
            let _ = done.send(Ok(None));
        }
        Collected::Abandoned => discard(writer, &folder).await,
        Collected::Failed(e) => {
            discard(writer, &folder).await;
            let _ = done.send(Err(e));
        }
    }
}

/// Flush the part, give it its content name and hand the photo to the receiver; when the
/// receiver is gone, remove the photo here.
async fn name(
    landing: &Landing,
    project: ProjectId,
    writer: PartWriter,
    folder: &std::path::Path,
    done: Done,
) {
    let named = match writer.finish().await {
        Ok(part) => landing.name_by_content(project, part).await,
        Err(e) => Err(e),
    };
    match named {
        Ok(photo) => {
            if let Err(Ok(Some(unclaimed))) = done.send(Ok(Some(photo))) {
                if let Err(e) = landing.remove(&unclaimed).await {
                    tracing::warn!(error = %e, "a landed photo of a dropped request was not removed");
                }
                let _ = tokio::fs::remove_dir(folder).await;
            }
        }
        Err(e) => {
            let _ = done.send(Err(e));
        }
    }
}

async fn discard(writer: PartWriter, folder: &std::path::Path) {
    if let Err(e) = writer.discard().await {
        tracing::warn!(error = %e, "a part file of a refused upload was not removed");
    }
    // NotEmpty (other parts are still there) and NotFound are both fine
    let _ = tokio::fs::remove_dir(folder).await;
}
