//! One chunk of the body per poll.
//!
//! `multer`, which parses the form under axum's `Multipart`, fills an unbounded buffer from the
//! body stream for as long as the stream answers `Ready` (`StreamBuffer::poll_stream` in
//! multer 3.1.0 loops on `poll_next` until it sees `Pending` or the end). Behind hyper that is
//! bounded by what the connection has read so far, because hyper reads the socket only when the
//! body is polled; behind a body that can always answer, a whole photo would be copied into
//! memory before the handler saw its first byte. This wrapper answers `Pending` after every chunk
//! it hands out, with an immediate wake-up, so the parser never holds more than the chunk in hand
//! and the writer's buffer, whatever the transport.

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::{Body, BodyDataStream};
use bytes::Bytes;
use futures_util::Stream;

/// The request body, one chunk per poll.
pub(super) struct OneChunkPerPoll {
    inner: BodyDataStream,
    just_yielded: bool,
}

impl OneChunkPerPoll {
    /// Wrap `body`.
    pub(super) fn wrap(body: Body) -> Body {
        Body::from_stream(Self {
            inner: body.into_data_stream(),
            just_yielded: false,
        })
    }
}

impl Stream for OneChunkPerPoll {
    type Item = Result<Bytes, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.just_yielded {
            self.just_yielded = false;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        let next = Pin::new(&mut self.inner).poll_next(cx);
        if matches!(next, Poll::Ready(Some(Ok(_)))) {
            self.just_yielded = true;
        }
        next
    }
}
