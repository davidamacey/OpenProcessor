//! Byte accounting and admission: the device cap, the pinned host count, and the fair admission
//! by count and bytes (plan 17 sections 2.2 and 2.3).
//!
//! * [`DeviceCap`] is the hard cap on every device allocation the engine makes except the region
//!   (`OP_FRAME_ENGINE_POOL_MIB`): frames, nvJPEG's lane buffers, weight sets, scratch. A charge
//!   past it fails at once as an out-of-memory error; nothing waits on it.
//! * [`HostCount`] counts pinned host bytes (nvJPEG's pinned buffers). They are reported, not
//!   charged to the device cap. Their bound is structural: each lane holds one pinned buffer per
//!   backend state, and nvJPEG grows a buffer only to what one photo's host phase needs (its
//!   encoded bytes for `GPU_HYBRID`; its entropy-decoded coefficients for `HYBRID`, which nvJPEG
//!   does not size in its documentation). So the total stays under lanes x 2 x the largest photo
//!   the route admits (`OP_FRAME_ENGINE_MAX_MEGAPIXELS`, `OP_FRAME_ENGINE_MAX_MESSAGE_MIB`), and
//!   the peak is measured, not assumed ([`HostCount::peak`], reported by the GPU self-test).
//! * [`Admission`] gives out permits by count and by bytes (in KiB, so a 32-bit permit count
//!   covers 4 TiB), first come first served, each wait bounded by a deadline. It is tokio's
//!   semaphore, which is fair also across `acquire_many`: a large request at the head of the
//!   queue holds back later small ones, so a large photo is never starved
//!   (<https://docs.rs/tokio/1.54.1/tokio/sync/struct.Semaphore.html>).

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::time::Instant;

/// The hard cap on device bytes; [`DeviceCap::reserve`] fails past it.
#[derive(Debug)]
pub struct DeviceCap {
    limit: usize,
    in_use: AtomicUsize,
    peak: AtomicUsize,
}

/// A charge refused by the cap: the allocation answers out-of-memory (`gpu_memory`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapExceeded {
    /// Bytes the refused charge asked for.
    pub need: usize,
    /// Bytes already counted against the cap when it was refused.
    pub in_use: usize,
    /// The cap in bytes.
    pub limit: usize,
}

impl fmt::Display for CapExceeded {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} device bytes asked with {} of {} in use (the engine's cap)",
            self.need, self.in_use, self.limit
        )
    }
}

impl std::error::Error for CapExceeded {}

impl DeviceCap {
    /// A cap of `limit` bytes with nothing counted, shared by every holder of the returned `Arc`.
    pub fn new(limit: usize) -> Arc<DeviceCap> {
        Arc::new(DeviceCap {
            limit,
            in_use: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        })
    }

    /// Count `bytes` against the cap, or refuse them without counting.
    ///
    /// # Errors
    ///
    /// [`CapExceeded`] when `bytes` more would pass the cap (or overflow `usize`).
    pub fn reserve(&self, bytes: usize) -> Result<(), CapExceeded> {
        let mut current = self.in_use.load(Ordering::Relaxed);
        loop {
            let next = match current.checked_add(bytes) {
                Some(n) if n <= self.limit => n,
                _ => {
                    return Err(CapExceeded {
                        need: bytes,
                        in_use: current,
                        limit: self.limit,
                    });
                }
            };
            match self.in_use.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::Relaxed);
                    return Ok(());
                }
                Err(seen) => current = seen,
            }
        }
    }

    /// Give back bytes counted by [`DeviceCap::reserve`].
    pub fn release(&self, bytes: usize) {
        self.in_use.fetch_sub(bytes, Ordering::AcqRel);
    }

    /// [`DeviceCap::reserve`] as a value that releases on drop.
    ///
    /// # Errors
    ///
    /// [`CapExceeded`] when `bytes` more would pass the cap; nothing is counted then.
    pub fn charge(self: &Arc<Self>, bytes: usize) -> Result<Charge, CapExceeded> {
        self.reserve(bytes)?;
        Ok(Charge {
            cap: self.clone(),
            bytes,
        })
    }

    /// The cap in bytes.
    pub fn limit(&self) -> usize {
        self.limit
    }

    /// Bytes counted against the cap now.
    pub fn in_use(&self) -> usize {
        self.in_use.load(Ordering::Acquire)
    }

    /// The highest `in_use` value reached since creation, in bytes.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
}

/// Bytes counted against a [`DeviceCap`] until dropped.
#[derive(Debug)]
pub struct Charge {
    cap: Arc<DeviceCap>,
    bytes: usize,
}

impl Charge {
    /// The bytes this charge holds against the cap.
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Charge {
    fn drop(&mut self) {
        self.cap.release(self.bytes);
    }
}

/// Pinned host bytes in use and their peak (reported; see the module documentation for their
/// bound).
#[derive(Debug, Default)]
pub struct HostCount {
    in_use: AtomicUsize,
    peak: AtomicUsize,
}

impl HostCount {
    /// Count `bytes` of pinned host memory as in use, and raise the peak when needed.
    pub fn add(&self, bytes: usize) {
        let now = self.in_use.fetch_add(bytes, Ordering::AcqRel) + bytes;
        self.peak.fetch_max(now, Ordering::Relaxed);
    }

    /// Stop counting `bytes` that were counted with [`HostCount::add`].
    pub fn sub(&self, bytes: usize) {
        self.in_use.fetch_sub(bytes, Ordering::AcqRel);
    }

    /// Pinned host bytes counted now.
    pub fn in_use(&self) -> usize {
        self.in_use.load(Ordering::Acquire)
    }

    /// The highest `in_use` value reached since creation, in bytes.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Acquire)
    }
}

/// Which bound refused an admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverloadCause {
    /// The count bound (photos in flight, or photos in work).
    Count,
    /// The byte budget.
    Bytes,
}

/// Why no permit was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitError {
    /// The bound was not free by the deadline: a retryable overload (`RESOURCE_EXHAUSTED`).
    Overload(OverloadCause),
    /// The request is larger than the whole bound and can never fit (`over_budget`).
    NeverFits(OverloadCause),
}

impl fmt::Display for AdmitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdmitError::Overload(c) => write!(f, "overload: {c:?} not free by the deadline"),
            AdmitError::NeverFits(c) => write!(f, "larger than the whole {c:?} bound"),
        }
    }
}

impl std::error::Error for AdmitError {}

/// Admission by count and by bytes, fair, with bounded waits.
#[derive(Debug)]
pub struct Admission {
    count: Arc<Semaphore>,
    kib: Arc<Semaphore>,
    max_count: u32,
    total_kib: u32,
    /// What granted permits hold. The semaphores' free counts are not used for this: tokio hands
    /// permits to the waiter at the head of the queue as they free up, before its request is
    /// whole, so they read lower than what was granted.
    held: Arc<Held>,
}

#[derive(Debug, Default)]
struct Held {
    count: AtomicU64,
    kib: AtomicU64,
}

/// Count and bytes held until dropped.
#[derive(Debug)]
pub struct Permit {
    _count: Option<OwnedSemaphorePermit>,
    _kib: Option<OwnedSemaphorePermit>,
    bytes: u64,
    units: (u32, u32),
    held: Arc<Held>,
}

impl Permit {
    /// The bytes this permit holds (as asked, before rounding to KiB).
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.held
            .count
            .fetch_sub(u64::from(self.units.0), Ordering::AcqRel);
        self.held
            .kib
            .fetch_sub(u64::from(self.units.1), Ordering::AcqRel);
    }
}

/// Bytes as whole KiB, rounded up; `None` past 4 TiB.
pub fn kib_of(bytes: u64) -> Option<u32> {
    u32::try_from(bytes.div_ceil(1024)).ok()
}

impl Admission {
    /// `max_count` units of count and `budget_bytes` (rounded down to whole KiB) of bytes.
    pub fn new(max_count: u32, budget_bytes: u64) -> Admission {
        let total_kib = u32::try_from(budget_bytes / 1024).unwrap_or(u32::MAX);
        Admission {
            count: Arc::new(Semaphore::new(max_count as usize)),
            kib: Arc::new(Semaphore::new(total_kib as usize)),
            max_count,
            total_kib,
            held: Arc::default(),
        }
    }

    /// Take `count` units and `bytes`, waiting in line until `deadline` at the latest. A
    /// deadline already passed takes them only if they are free at once.
    ///
    /// # Errors
    ///
    /// [`AdmitError::NeverFits`] when `bytes` (as whole KiB) or `count` is larger than the whole
    /// bound; [`AdmitError::Overload`] when the bound was not free by `deadline`.
    pub async fn admit(
        &self,
        count: u32,
        bytes: u64,
        deadline: Instant,
    ) -> Result<Permit, AdmitError> {
        let kib = match kib_of(bytes) {
            Some(k) if k <= self.total_kib => k,
            _ => return Err(AdmitError::NeverFits(OverloadCause::Bytes)),
        };
        if count > self.max_count {
            return Err(AdmitError::NeverFits(OverloadCause::Count));
        }
        let held_count = take(&self.count, count, deadline, OverloadCause::Count).await?;
        let held_kib = take(&self.kib, kib, deadline, OverloadCause::Bytes).await?;
        self.held
            .count
            .fetch_add(u64::from(count), Ordering::AcqRel);
        self.held.kib.fetch_add(u64::from(kib), Ordering::AcqRel);
        Ok(Permit {
            _count: held_count,
            _kib: held_kib,
            bytes,
            units: (count, kib),
            held: self.held.clone(),
        })
    }

    /// Count units held by granted permits.
    pub fn count_in_use(&self) -> u64 {
        self.held.count.load(Ordering::Acquire)
    }

    /// Bytes held by granted permits (whole KiB times 1024).
    pub fn bytes_in_use(&self) -> u64 {
        self.held.kib.load(Ordering::Acquire) * 1024
    }

    /// The count bound given at creation.
    pub fn max_count(&self) -> u32 {
        self.max_count
    }

    /// The byte bound in bytes, rounded down to whole KiB.
    pub fn budget_bytes(&self) -> u64 {
        u64::from(self.total_kib) * 1024
    }
}

async fn take(
    sem: &Arc<Semaphore>,
    n: u32,
    deadline: Instant,
    cause: OverloadCause,
) -> Result<Option<OwnedSemaphorePermit>, AdmitError> {
    if n == 0 {
        return Ok(None);
    }
    match sem.clone().try_acquire_many_owned(n) {
        Ok(p) => return Ok(Some(p)),
        Err(TryAcquireError::Closed) => return Err(AdmitError::Overload(cause)),
        Err(TryAcquireError::NoPermits) => {}
    }
    if deadline <= Instant::now() {
        return Err(AdmitError::Overload(cause));
    }
    match tokio::time::timeout_at(deadline, sem.clone().acquire_many_owned(n)).await {
        Ok(Ok(p)) => Ok(Some(p)),
        Ok(Err(_)) | Err(_) => Err(AdmitError::Overload(cause)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cap_refuses_past_its_limit_and_releases() {
        let cap = DeviceCap::new(1000);
        let a = cap.charge(600).unwrap();
        assert_eq!(
            cap.reserve(401),
            Err(CapExceeded {
                need: 401,
                in_use: 600,
                limit: 1000
            })
        );
        let b = cap.charge(400).unwrap();
        assert_eq!(cap.in_use(), 1000);
        drop(a);
        drop(b);
        assert_eq!(cap.in_use(), 0);
        assert_eq!(cap.peak(), 1000);
        assert!(
            cap.reserve(usize::MAX).is_err(),
            "an overflowing ask is refused"
        );
    }

    #[test]
    fn cap_holds_under_threads() {
        let cap = DeviceCap::new(10_000);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let cap = cap.clone();
                std::thread::spawn(move || {
                    for _ in 0..10_000 {
                        if let Ok(c) = cap.charge(1_500) {
                            assert!(cap.in_use() <= 10_000);
                            drop(c);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(cap.in_use(), 0);
        assert!(cap.peak() <= 10_000);
    }

    #[test]
    fn kib_rounds_up() {
        assert_eq!(kib_of(0), Some(0));
        assert_eq!(kib_of(1), Some(1));
        assert_eq!(kib_of(1024), Some(1));
        assert_eq!(kib_of(1025), Some(2));
        assert_eq!(kib_of(u64::MAX), None);
    }

    fn at(ms: u64) -> Instant {
        Instant::now() + Duration::from_millis(ms)
    }

    #[tokio::test(start_paused = true)]
    async fn admission_answers_at_once_when_the_deadline_passed() {
        let a = Admission::new(1, 1 << 20);
        let held = a.admit(1, 10, Instant::now()).await.unwrap();
        assert_eq!(
            a.admit(1, 10, Instant::now()).await.unwrap_err(),
            AdmitError::Overload(OverloadCause::Count)
        );
        drop(held);
        assert!(a.admit(1, 10, Instant::now()).await.is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn admission_refuses_what_never_fits() {
        let a = Admission::new(4, 8 * 1024);
        assert_eq!(
            a.admit(1, 8 * 1024 + 1, at(10)).await.unwrap_err(),
            AdmitError::NeverFits(OverloadCause::Bytes)
        );
        assert_eq!(
            a.admit(5, 0, at(10)).await.unwrap_err(),
            AdmitError::NeverFits(OverloadCause::Count)
        );
        assert!(a.admit(4, 8 * 1024, at(10)).await.is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn bytes_wait_is_bounded_by_the_deadline() {
        let a = Admission::new(8, 100 * 1024);
        let _held = a.admit(1, 90 * 1024, at(0)).await.unwrap();
        let started = Instant::now();
        let err = a.admit(1, 20 * 1024, at(250)).await.unwrap_err();
        assert_eq!(err, AdmitError::Overload(OverloadCause::Bytes));
        assert_eq!(Instant::now() - started, Duration::from_millis(250));
    }

    #[tokio::test(start_paused = true)]
    async fn a_large_waiter_is_served_before_later_small_ones() {
        let a = Arc::new(Admission::new(8, 100 * 1024));
        let first = a.admit(1, 60 * 1024, at(0)).await.unwrap();
        // The large one queues first: 80 KiB cannot fit while 60 are held.
        let large = {
            let a = a.clone();
            tokio::spawn(async move { a.admit(1, 80 * 1024, at(10_000)).await })
        };
        tokio::task::yield_now().await;
        // 30 KiB would fit now (60 + 30 <= 100) but must wait behind the large request.
        let small = {
            let a = a.clone();
            tokio::spawn(async move { a.admit(1, 30 * 1024, at(10_000)).await })
        };
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(
            !small.is_finished(),
            "a later small request does not overtake the large one"
        );
        drop(first);
        let large = large.await.unwrap().unwrap();
        assert_eq!(a.bytes_in_use(), 80 * 1024);
        drop(large);
        assert!(small.await.unwrap().is_ok());
    }

    #[tokio::test(start_paused = true)]
    async fn bytes_in_use_never_pass_the_budget() {
        let a = Arc::new(Admission::new(64, 1024 * 1024));
        let tasks: Vec<_> = (0..64u64)
            .map(|i| {
                let a = a.clone();
                tokio::spawn(async move {
                    let need = (i % 7 + 1) * 100 * 1024;
                    if let Ok(p) = a.admit(1, need, at(60_000)).await {
                        assert!(a.bytes_in_use() <= a.budget_bytes());
                        tokio::time::sleep(Duration::from_millis(i % 5)).await;
                        drop(p);
                    }
                })
            })
            .collect();
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(a.bytes_in_use(), 0);
        assert_eq!(a.count_in_use(), 0);
    }
}
