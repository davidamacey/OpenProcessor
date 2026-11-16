//! The bounded hand-off of landed photos to the sink.
//!
//! A request has one channel of capacity `OP_UPLOAD_HANDOFF_QUEUE` and one task that drains it
//! into [`PartSink::accept`], one photo after the other. The receiver sends with
//! [`mpsc::Sender::send`], which waits while the channel is full; while it waits the receiver
//! does not poll the body, hyper stops reading the socket and TCP's flow control slows the
//! client (<https://docs.rs/tokio/1.54.1/tokio/sync/mpsc/fn.channel.html>: a bounded channel
//! "provides backpressure"). The task is spawned so a panic in the sink is caught as a
//! [`tokio::task::JoinError`] instead of unwinding through the request, and it runs on the state's
//! task tracker, so a request that is dropped does not hide the photo the sink is still in
//! `accept` for from the shutdown wait.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::task::TaskTracker;

use super::refusal::Refusal;
use super::sink::{PartSink, ReceivedPart, SinkError, UploadEnd};
use super::wire::SinkBatch;

/// The sending end of a request's hand-off and the task that drains it.
pub(super) struct Handoff {
    tx: mpsc::Sender<ReceivedPart>,
    drain: JoinHandle<Result<(), SinkError>>,
}

impl Handoff {
    pub(super) fn start<S: PartSink>(sink: Arc<S>, capacity: usize, tracker: &TaskTracker) -> Self {
        let (tx, mut rx) = mpsc::channel::<ReceivedPart>(capacity);
        let drain = tracker.spawn(async move {
            while let Some(part) = rx.recv().await {
                sink.accept(part).await?;
            }
            Ok(())
        });
        Self { tx, drain }
    }

    /// Queue a photo; waits while the queue is full. `false` when the drain task has ended, which
    /// means the sink refused a photo or panicked: [`Handoff::close`] says which.
    pub(super) async fn send(&self, part: ReceivedPart) -> bool {
        self.tx.send(part).await.is_ok()
    }

    /// No more photos: wait for the sink to take the queued ones.
    pub(super) async fn close(self) -> Result<(), Refusal> {
        drop(self.tx);
        match self.drain.await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e.into()),
            Err(join) => {
                tracing::error!(error = %join, "the ingest sink panicked taking a photo");
                Err(Refusal::Internal)
            }
        }
    }

    /// The request is over: stop the drain task. A photo it is inside `accept` for is dropped
    /// there, and every photo still queued is dropped with the channel, which releases its permit.
    pub(super) fn cancel(self) {
        self.drain.abort();
    }
}

/// Commit through the sink in a task of its own, so a panic is an internal error.
pub(super) async fn finish<S: PartSink>(
    sink: &Arc<S>,
    end: UploadEnd,
) -> Result<SinkBatch, Refusal> {
    let sink = Arc::clone(sink);
    match tokio::spawn(async move { sink.finish(end).await }).await {
        Ok(Ok(batch)) => Ok(batch),
        Ok(Err(e)) => Err(e.into()),
        Err(join) => {
            tracing::error!(error = %join, "the ingest sink panicked finishing an upload");
            Err(Refusal::Internal)
        }
    }
}
