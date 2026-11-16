//! SHA-256 over a stream of chunks (plan 18 section 2.2: `photos.content_sha256` is SHA-256).
//!
//! `sha2` 0.11.1 (<https://docs.rs/sha2/0.11.1/sha2/>) is a software implementation here; the
//! host's CPUs have no SHA instructions, so the cost per byte is measured in M18. An async caller
//! uses [`Sha256Stream::update_cooperative`], which hashes in slices and yields between them.

use sha2::{Digest, Sha256};

use crate::keys::ContentHash;

/// Bytes hashed between two yields to the executor (about 0.2 ms at software SHA-256 speed).
const HASH_SLICE: usize = 64 * 1024;

/// A running SHA-256. Feed every chunk to [`Sha256Stream::update`], then [`Sha256Stream::finish`].
#[derive(Debug, Clone, Default)]
pub struct Sha256Stream {
    inner: Sha256,
    len: u64,
}

impl Sha256Stream {
    /// An empty stream.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the next chunk.
    pub fn update(&mut self, chunk: &[u8]) {
        self.inner.update(chunk);
        self.len += chunk.len() as u64;
    }

    /// Add the next chunk without holding the executor: the chunk is hashed in slices of 64 KiB
    /// with a [`tokio::task::yield_now`] between slices
    /// (<https://docs.rs/tokio/1.54.1/tokio/task/fn.yield_now.html>). Tokio's guidance is that a
    /// task must not run long between `.await`s; a slice takes well under a millisecond, and
    /// yielding keeps the work on the worker thread, where a `spawn_blocking` hand-off would cost
    /// two thread hops per chunk.
    pub async fn update_cooperative(&mut self, chunk: &[u8]) {
        let mut slices = chunk.chunks(HASH_SLICE).peekable();
        while let Some(slice) = slices.next() {
            self.update(slice);
            if slices.peek().is_some() {
                tokio::task::yield_now().await;
            }
        }
    }

    /// The number of bytes seen so far.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether no byte has been seen.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The digest of everything seen.
    #[must_use]
    pub fn finish(self) -> ContentHash {
        ContentHash(self.inner.finalize().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_shot(data: &[u8]) -> ContentHash {
        ContentHash(Sha256::digest(data).into())
    }

    #[test]
    fn the_empty_stream_is_the_known_empty_digest() {
        assert_eq!(
            Sha256Stream::new().finish().to_string(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn chunks_of_any_size_give_the_one_shot_digest() {
        let data: Vec<u8> = (0..100_000_u32).map(|i| (i % 251) as u8).collect();
        for size in [1, 7, 4096, 65_535, 100_000] {
            let mut s = Sha256Stream::new();
            for chunk in data.chunks(size) {
                s.update(chunk);
            }
            assert_eq!(s.len(), data.len() as u64);
            assert_eq!(s.finish(), one_shot(&data), "chunk size {size}");
        }
    }

    #[test]
    fn a_big_chunk_is_hashed_in_slices_with_a_yield_between_them() {
        use std::future::Future;
        use std::pin::pin;
        use std::task::{Context, Poll, Waker};

        let data = vec![7_u8; HASH_SLICE * 3 + 1];
        let mut s = Sha256Stream::new();
        let mut cx = Context::from_waker(Waker::noop());
        let mut pending = 0;
        {
            let mut fut = pin!(s.update_cooperative(&data));
            while fut.as_mut().poll(&mut cx) == Poll::Pending {
                pending += 1;
            }
        }
        assert_eq!(pending, 3, "four slices need three yields");
        assert_eq!(s.finish(), one_shot(&data));
    }

    #[test]
    fn a_small_chunk_is_hashed_without_a_yield() {
        use std::future::Future;
        use std::pin::pin;
        use std::task::{Context, Poll, Waker};

        let mut s = Sha256Stream::new();
        let mut cx = Context::from_waker(Waker::noop());
        let mut fut = pin!(s.update_cooperative(b"abc"));
        assert_eq!(fut.as_mut().poll(&mut cx), Poll::Ready(()));
    }

    #[test]
    fn abc_is_the_known_digest() {
        let mut s = Sha256Stream::new();
        s.update(b"abc");
        assert_eq!(
            s.finish().to_string(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
