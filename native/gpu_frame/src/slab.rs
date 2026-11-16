//! Offsets inside the one registered region: first fit with coalescing, and a bounded wait
//! (plan 17 section 2.2; the Python engine's `slab.py`).
//!
//! Every model input of every photo in flight is a slice of the region, addressed by offset in
//! the model call. A request that finds no room waits at most its bound and then answers
//! overload (a retryable status): pressure is an answer, never an unbounded wait. First fit can
//! fragment, which is why the wait has its own bound (`OP_FRAME_ENGINE_SLAB_WAIT_SECONDS`).
//!
//! Waits come in two forms over one free list: `take` for async callers (the photo's control task
//! takes the round-1 slices) and `take_blocking` for pool threads (a launch thread stages round-2
//! slices). A release wakes both.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use tokio::sync::Notify;

/// Every slice starts on this boundary (`cuMemAlloc` alignment; kernels may write vectors).
pub const ALIGN: usize = 256;

/// `bytes` rounded up to [`ALIGN`], or `None` on overflow.
pub fn aligned(bytes: usize) -> Option<usize> {
    bytes.checked_add(ALIGN - 1).map(|n| n / ALIGN * ALIGN)
}

/// Why no slice was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceError {
    /// No room within the wait: a retryable overload (`RESOURCE_EXHAUSTED`).
    Overload {
        /// Bytes asked for.
        need: usize,
    },
    /// Larger than the whole region: a size setting, never load (`over_budget`).
    TooLarge {
        /// Bytes asked for.
        need: usize,
        /// The whole region's size in bytes.
        region: usize,
    },
}

impl fmt::Display for SliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SliceError::Overload { need } => {
                write!(f, "{need} bytes of the region not free in time")
            }
            SliceError::TooLarge { need, region } => {
                write!(f, "{need} bytes exceed the region of {region}")
            }
        }
    }
}

impl std::error::Error for SliceError {}

/// The free list: `[offset, length)` blocks sorted by offset, never adjacent (coalesced).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirstFit {
    size: usize,
    free: Vec<(usize, usize)>,
    in_use: usize,
    peak: usize,
}

impl FirstFit {
    /// A free list over `size` bytes; `size` must be a positive multiple of [`ALIGN`].
    pub fn new(size: usize) -> Option<FirstFit> {
        (size > 0 && size.is_multiple_of(ALIGN)).then(|| FirstFit {
            size,
            free: vec![(0, size)],
            in_use: 0,
            peak: 0,
        })
    }

    /// The lowest offset with `n` free bytes (`n` already aligned), taken.
    pub fn take(&mut self, n: usize) -> Option<usize> {
        let k = self.free.iter().position(|&(_, len)| len >= n)?;
        let (off, len) = self.free[k];
        if len == n {
            self.free.remove(k);
        } else {
            self.free[k] = (off + n, len - n);
        }
        self.in_use += n;
        self.peak = self.peak.max(self.in_use);
        Some(off)
    }

    /// Give back `[off, off + n)` and merge it with free neighbours.
    pub fn give(&mut self, off: usize, n: usize) {
        self.in_use -= n;
        let k = self.free.partition_point(|&(o, _)| o < off);
        self.free.insert(k, (off, n));
        if k + 1 < self.free.len() && self.free[k].0 + self.free[k].1 == self.free[k + 1].0 {
            self.free[k].1 += self.free[k + 1].1;
            self.free.remove(k + 1);
        }
        if k > 0 && self.free[k - 1].0 + self.free[k - 1].1 == self.free[k].0 {
            self.free[k - 1].1 += self.free[k].1;
            self.free.remove(k);
        }
    }

    /// The slab's total size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Bytes handed out now, after rounding up to the alignment.
    pub fn in_use(&self) -> usize {
        self.in_use
    }

    /// The highest `in_use` value reached since creation, in bytes.
    pub fn peak(&self) -> usize {
        self.peak
    }

    /// The free blocks, for inspection.
    pub fn free_blocks(&self) -> &[(usize, usize)] {
        &self.free
    }
}

/// A [`FirstFit`] shared by threads and tasks, with bounded waits.
#[derive(Debug)]
pub struct Slab {
    list: Mutex<FirstFit>,
    cond: Condvar,
    notify: Notify,
    waits: AtomicU64,
}

/// What the slab holds now, for metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabUsage {
    /// The slab's total size in bytes.
    pub size: usize,
    /// Bytes handed out now.
    pub in_use: usize,
    /// The highest bytes handed out at once since creation.
    pub peak: usize,
    /// Requests that had to wait at least once.
    pub waits: u64,
}

impl Slab {
    /// A slab of `size` bytes, or `None` when `size` is not a positive multiple of [`ALIGN`].
    pub fn new(size: usize) -> Option<Slab> {
        Some(Slab {
            list: Mutex::new(FirstFit::new(size)?),
            cond: Condvar::new(),
            notify: Notify::new(),
            waits: AtomicU64::new(0),
        })
    }

    fn lock(&self) -> MutexGuard<'_, FirstFit> {
        // The list stays consistent across a panic elsewhere: every update is a few statements
        // without a call that could unwind.
        self.list.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn need(&self, bytes: usize) -> Result<usize, SliceError> {
        let size = self.lock().size();
        match aligned(bytes) {
            Some(n) if n <= size => Ok(n),
            _ => Err(SliceError::TooLarge {
                need: bytes,
                region: size,
            }),
        }
    }

    /// The offset of `bytes` (rounded up to [`ALIGN`]), waiting at most `wait`; for async callers.
    ///
    /// # Errors
    ///
    /// [`SliceError::TooLarge`] when the rounded size is larger than the slab;
    /// [`SliceError::Overload`] when no block was free within `wait`.
    pub async fn take(&self, bytes: usize, wait: Duration) -> Result<usize, SliceError> {
        let n = self.need(bytes)?;
        let deadline = tokio::time::Instant::now() + wait;
        let mut waited = false;
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            // Registered before the check, so a release between the check and the wait wakes it.
            notified.as_mut().enable();
            if let Some(off) = self.lock().take(n) {
                return Ok(off);
            }
            if !waited {
                self.waits.fetch_add(1, Ordering::Relaxed);
                waited = true;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(SliceError::Overload { need: bytes });
            }
        }
    }

    /// The same for a pool thread: blocks the calling thread at most `wait`.
    ///
    /// # Errors
    ///
    /// [`SliceError::TooLarge`] when the rounded size is larger than the slab;
    /// [`SliceError::Overload`] when no block was free within `wait`.
    pub fn take_blocking(&self, bytes: usize, wait: Duration) -> Result<usize, SliceError> {
        let n = self.need(bytes)?;
        let deadline = std::time::Instant::now() + wait;
        let mut list = self.lock();
        let mut waited = false;
        loop {
            if let Some(off) = list.take(n) {
                return Ok(off);
            }
            if !waited {
                self.waits.fetch_add(1, Ordering::Relaxed);
                waited = true;
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                return Err(SliceError::Overload { need: bytes });
            }
            list = self
                .cond
                .wait_timeout(list, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Give back a slice taken with `bytes` (the size asked, not the rounded one).
    pub fn give(&self, off: usize, bytes: usize) {
        if let Some(n) = aligned(bytes) {
            self.lock().give(off, n);
        }
        self.cond.notify_all();
        self.notify.notify_waiters();
    }

    /// A snapshot of the slab's size, use, peak and wait count.
    pub fn usage(&self) -> SlabUsage {
        let list = self.lock();
        SlabUsage {
            size: list.size(),
            in_use: list.in_use(),
            peak: list.peak(),
            waits: self.waits.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn sizes_round_to_the_alignment() {
        assert_eq!(aligned(0), Some(0));
        assert_eq!(aligned(1), Some(256));
        assert_eq!(aligned(256), Some(256));
        assert_eq!(aligned(257), Some(512));
        assert_eq!(aligned(usize::MAX), None);
        assert!(FirstFit::new(1000).is_none());
        assert!(FirstFit::new(0).is_none());
    }

    #[test]
    fn first_fit_takes_the_lowest_block_that_fits() {
        let mut s = FirstFit::new(4096).unwrap();
        let a = s.take(1024).unwrap();
        let b = s.take(1024).unwrap();
        let c = s.take(1024).unwrap();
        assert_eq!((a, b, c), (0, 1024, 2048));
        s.give(b, 1024);
        // 512 fits in the hole at 1024, the lowest block, not at 3072.
        assert_eq!(s.take(512), Some(1024));
        assert_eq!(s.take(1024), Some(3072));
        assert_eq!(s.take(1024), None, "only 512 bytes are free, in one hole");
        assert_eq!(s.in_use(), 3584);
    }

    #[test]
    fn released_neighbours_coalesce() {
        let mut s = FirstFit::new(4096).unwrap();
        let offs: Vec<usize> = (0..4).map(|_| s.take(1024).unwrap()).collect();
        s.give(offs[0], 1024);
        s.give(offs[2], 1024);
        assert_eq!(s.free_blocks(), &[(0, 1024), (2048, 1024)]);
        s.give(offs[1], 1024);
        assert_eq!(
            s.free_blocks(),
            &[(0, 3072)],
            "left and right neighbours merge"
        );
        s.give(offs[3], 1024);
        assert_eq!(s.free_blocks(), &[(0, 4096)]);
        assert_eq!(s.take(4096), Some(0), "the whole region is one block again");
        assert_eq!(s.peak(), 4096);
    }

    #[test]
    fn a_blocking_wait_is_bounded() {
        let s = Slab::new(1024).unwrap();
        let held = s.take_blocking(1024, Duration::ZERO).unwrap();
        let started = std::time::Instant::now();
        assert_eq!(
            s.take_blocking(256, Duration::from_millis(50)),
            Err(SliceError::Overload { need: 256 })
        );
        assert!(started.elapsed() >= Duration::from_millis(50));
        assert_eq!(s.usage().waits, 1);
        s.give(held, 1024);
        assert_eq!(s.take_blocking(256, Duration::ZERO), Ok(0));
        assert_eq!(
            s.take_blocking(2048, Duration::ZERO),
            Err(SliceError::TooLarge {
                need: 2048,
                region: 1024
            })
        );
    }

    #[test]
    fn a_release_wakes_a_blocked_thread() {
        let s = Arc::new(Slab::new(1024).unwrap());
        let held = s.take_blocking(1024, Duration::ZERO).unwrap();
        let waiter = {
            let s = s.clone();
            std::thread::spawn(move || s.take_blocking(512, Duration::from_secs(10)))
        };
        std::thread::sleep(Duration::from_millis(20));
        s.give(held, 1024);
        assert_eq!(waiter.join().unwrap(), Ok(0));
    }

    #[tokio::test(start_paused = true)]
    async fn an_async_wait_is_bounded_and_woken() {
        let s = Arc::new(Slab::new(1024).unwrap());
        let held = s.take(1024, Duration::ZERO).await.unwrap();
        assert_eq!(
            s.take(256, Duration::from_millis(100)).await,
            Err(SliceError::Overload { need: 256 })
        );
        let waiter = {
            let s = s.clone();
            tokio::spawn(async move { s.take(512, Duration::from_secs(10)).await })
        };
        tokio::task::yield_now().await;
        s.give(held, 1024);
        assert_eq!(waiter.await.unwrap(), Ok(0));
    }
}
