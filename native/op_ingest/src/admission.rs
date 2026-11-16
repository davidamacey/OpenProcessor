//! Admission: how much photo work the process takes on at once.
//!
//! Two [`Semaphore`]s bound it, process wide, so many requests share one limit. `photos` counts
//! photos being received or worked on. `bytes` counts the bytes those photos hold on the landing
//! disk and in the sink, in KiB units so a `u32` request covers 4 TiB. A photo takes its photo
//! permit when its part starts and its bytes permit when its first chunk arrives, so a client
//! that is slow to start a part holds neither a byte reservation nor, once its last part is in,
//! anything at all. The permits live in a [`WorkPermit`] that travels with the photo to the sink
//! and is released when the sink drops it.
//!
//! The bound is on waiting, not on refusing: a part that cannot be admitted within `wait` is
//! [`Busy`], which the route answers 503 with `Retry-After`. tokio's semaphore is fair (permits
//! are granted in the order they were asked for, so a large reservation is not starved by small
//! ones) and `acquire_many_owned` is cancel safe: a waiter that is dropped loses its place and
//! takes no permits
//! (<https://docs.rs/tokio/1.54.1/tokio/sync/struct.Semaphore.html>).
//!
//! No deadlock between requests: a request waits for permits while holding only permits of photos
//! that are already in the sink, and the sink's work on a photo never waits for a later part of
//! the request, so held permits are always released in the end.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::timeout;

/// Bytes per permit of the bytes semaphore.
const UNIT: u64 = 1024;

/// Which bound a part could not get inside the wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bound {
    /// Photos in work.
    Photos,
    /// Bytes in flight.
    Bytes,
}

/// A part could not be admitted inside the admission wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Busy {
    /// The bound that stayed full.
    pub bound: Bound,
}

/// What is free right now, for tests and metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Available {
    /// Free photo permits.
    pub photos: usize,
    /// Free bytes, in whole KiB.
    pub bytes: u64,
}

/// The two bounds, shared by every request of the process.
#[derive(Debug, Clone)]
pub struct Admission {
    photos: Arc<Semaphore>,
    bytes: Arc<Semaphore>,
    wait: Duration,
}

/// One photo's share of the bounds. Dropping it gives both back.
#[derive(Debug)]
pub struct WorkPermit {
    _photo: OwnedSemaphorePermit,
    bytes: Option<OwnedSemaphorePermit>,
}

fn units(bytes: u64) -> u32 {
    u32::try_from(bytes.div_ceil(UNIT).max(1)).unwrap_or(u32::MAX)
}

impl Admission {
    /// Bounds of `photos` photos and `bytes` bytes, each wait capped at `wait`.
    #[must_use]
    pub fn new(photos: usize, bytes: u64, wait: Duration) -> Self {
        let byte_units = usize::try_from(bytes.div_ceil(UNIT)).unwrap_or(usize::MAX >> 4);
        Self {
            photos: Arc::new(Semaphore::new(photos)),
            bytes: Arc::new(Semaphore::new(byte_units)),
            wait,
        }
    }

    /// Take a photo permit for a part that is starting.
    ///
    /// # Errors
    ///
    /// [`Busy`] when no photo permit was free inside the wait.
    pub async fn admit_photo(&self) -> Result<WorkPermit, Busy> {
        let photo = timeout(self.wait, Arc::clone(&self.photos).acquire_owned())
            .await
            .ok()
            .and_then(Result::ok)
            .ok_or(Busy {
                bound: Bound::Photos,
            })?;
        Ok(WorkPermit {
            _photo: photo,
            bytes: None,
        })
    }

    /// Reserve `bytes` (rounded up to whole KiB) for the photo of `permit`, at its first chunk.
    /// The reservation replaces any earlier one.
    ///
    /// # Errors
    ///
    /// [`Busy`] when the bytes were not free inside the wait; the permit keeps what it had.
    pub async fn reserve_bytes(&self, permit: &mut WorkPermit, bytes: u64) -> Result<(), Busy> {
        let got = timeout(
            self.wait,
            Arc::clone(&self.bytes).acquire_many_owned(units(bytes)),
        )
        .await
        .ok()
        .and_then(Result::ok)
        .ok_or(Busy {
            bound: Bound::Bytes,
        })?;
        permit.bytes = Some(got);
        Ok(())
    }

    /// What is free now.
    #[must_use]
    pub fn available(&self) -> Available {
        Available {
            photos: self.photos.available_permits(),
            bytes: self.bytes.available_permits() as u64 * UNIT,
        }
    }
}

impl WorkPermit {
    /// Shrink the reservation to the photo's real size, after the part ended, and give the rest
    /// back at once.
    pub fn settle(&mut self, actual_bytes: u64) {
        let keep = usize::try_from(units(actual_bytes)).unwrap_or(usize::MAX);
        if let Some(held) = self.bytes.as_mut() {
            let extra = held.num_permits().saturating_sub(keep);
            if extra > 0 {
                drop(held.split(extra));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admission(photos: usize, kib: u64) -> Admission {
        Admission::new(photos, kib * UNIT, Duration::from_millis(30))
    }

    #[tokio::test]
    async fn a_photo_permit_is_returned_when_the_work_permit_drops() {
        let a = admission(2, 100);
        let first = a.admit_photo().await.unwrap();
        let _second = a.admit_photo().await.unwrap();
        assert_eq!(a.available().photos, 0);
        drop(first);
        assert_eq!(a.available().photos, 1);
    }

    #[tokio::test]
    async fn a_full_photo_bound_is_busy_after_the_wait() {
        let a = admission(1, 100);
        let _held = a.admit_photo().await.unwrap();
        let busy = a.admit_photo().await.unwrap_err();
        assert_eq!(busy.bound, Bound::Photos);
    }

    #[tokio::test]
    async fn reserve_settle_and_drop_move_the_bytes_exactly() {
        let a = admission(2, 100);
        let mut p = a.admit_photo().await.unwrap();
        a.reserve_bytes(&mut p, 60 * UNIT).await.unwrap();
        assert_eq!(a.available().bytes, 40 * UNIT);
        p.settle(10 * UNIT + 1);
        assert_eq!(a.available().bytes, 89 * UNIT);
        drop(p);
        assert_eq!(a.available().bytes, 100 * UNIT);
    }

    #[tokio::test]
    async fn a_reservation_larger_than_what_is_free_is_busy_and_changes_nothing() {
        let a = admission(2, 100);
        let mut first = a.admit_photo().await.unwrap();
        a.reserve_bytes(&mut first, 80 * UNIT).await.unwrap();
        let mut second = a.admit_photo().await.unwrap();
        let busy = a.reserve_bytes(&mut second, 30 * UNIT).await.unwrap_err();
        assert_eq!(busy.bound, Bound::Bytes);
        assert_eq!(a.available().bytes, 20 * UNIT);
    }

    #[tokio::test]
    async fn a_waiter_that_is_dropped_takes_no_permit() {
        let a = admission(1, 100);
        let held = a.admit_photo().await.unwrap();
        let waiting = a.clone();
        let task = tokio::spawn(async move { waiting.admit_photo().await });
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        drop(held);
        assert_eq!(a.available().photos, 1);
    }
}
