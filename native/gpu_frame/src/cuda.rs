//! The CUDA driver and NVRTC through cudarc's raw layers: the context, streams, the
//! stream-ordered memory pool, pinned host memory, copies, modules and kernel launches.
//!
//! Every CUDA call of the crate except the region's allocation and export (`region.rs`) and
//! nvJPEG (`nvjpeg/ffi.rs`) is made here, and each unsafe block carries its SAFETY note. The
//! libraries are loaded at run time (cudarc's `dynamic-loading` feature); a missing library is an
//! error value, never a panic, so the crate builds and tests on a machine without a GPU.
//!
//! Errors are values that keep the call's name and its `CUresult`, and classify themselves:
//! [`Fault::ContextLost`] for the codes after which "the process must be terminated and
//! relaunched", [`Fault::OutOfMemory`] for `CUDA_ERROR_OUT_OF_MEMORY` (plan 17 sections 1.3, 2.3).
//!
//! Driver API reference: https://docs.nvidia.com/cuda/cuda-driver-api/index.html (context
//! management, stream management, memory management, stream-ordered memory allocator, module
//! management, execution control); NVRTC: https://docs.nvidia.com/cuda/nvrtc/index.html;
//! cudarc 0.19.10: https://docs.rs/cudarc/0.19.10/cudarc/driver/result/index.html.

use std::ffi::{c_char, c_int, c_void};
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{CudaContext, result, sys};

mod error;

use error::driver;
pub use error::{CUresult, CudaError, Fault, STICKY, fault_of};

/// One device's primary context, retained for the life of this value, and whether it was lost.
///
/// Every thread that makes a CUDA call binds it first ([`Context::bind`]); nvJPEG reaches the
/// driver through the runtime, which uses the context current on the calling thread.
pub struct Context {
    inner: Arc<CudaContext>,
    lost: AtomicBool,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("ordinal", &self.inner.ordinal())
            .field("lost", &self.is_lost())
            .finish()
    }
}

impl Context {
    /// Retain the primary context of device `ordinal` (as `CUDA_VISIBLE_DEVICES` numbers it)
    /// and bind it to the calling thread.
    pub fn new(ordinal: usize) -> Result<Arc<Context>, CudaError> {
        // SAFETY: probing the loader has no precondition; it only tries to open the library.
        if !unsafe { sys::is_culib_present() } {
            return Err(CudaError::LibraryMissing {
                library: "CUDA driver (libcuda)",
            });
        }
        let inner = driver("cuDevicePrimaryCtxRetain", CudaContext::new(ordinal))?;
        Ok(Arc::new(Context {
            inner,
            lost: AtomicBool::new(false),
        }))
    }

    /// Make this context current on the calling thread (cudarc's `bind_to_thread`).
    pub fn bind(&self) -> Result<(), CudaError> {
        self.check("cuCtxSetCurrent", self.inner.bind_to_thread())
    }

    /// The device ordinal.
    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    /// The device's compute capability (major, minor).
    pub fn compute_capability(&self) -> Result<(i32, i32), CudaError> {
        self.check("cuDeviceGetAttribute", self.inner.compute_capability())
    }

    /// The device's name.
    pub fn device_name(&self) -> Result<String, CudaError> {
        self.check("cuDeviceGetName", self.inner.name())
    }

    /// The device's PCI bus id (`0000:65:00.0`), which names the card whatever the ordinal.
    pub fn pci_bus_id(&self) -> Result<String, CudaError> {
        let mut buf = [0 as c_char; 32];
        // SAFETY: the buffer holds `len` bytes, the documented minimum is 13; the device handle
        // comes from the retained context.
        let rc = unsafe {
            sys::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as c_int, self.inner.cu_device())
        };
        self.check("cuDeviceGetPCIBusId", rc.result())?;
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Free and total device memory in bytes (`cuMemGetInfo`), for the whole device.
    pub fn memory_info(&self) -> Result<(usize, usize), CudaError> {
        self.bind()?;
        self.check("cuMemGetInfo", result::mem_get_info())
    }

    /// Wait for all work of the context (`cuCtxSynchronize`): the probe that tells a photo's
    /// failure from a lost context.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.bind()?;
        self.check("cuCtxSynchronize", result::ctx::synchronize())
    }

    /// Whether a call of this context answered a sticky code.
    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    /// Turn a driver result into a value, marking the context lost on a sticky code.
    pub(crate) fn check<T>(
        &self,
        call: &'static str,
        r: Result<T, result::DriverError>,
    ) -> Result<T, CudaError> {
        let r = driver(call, r);
        if let Err(e) = &r {
            self.note(e);
        }
        r
    }

    /// Record a failure that came back through another library (nvJPEG) or another path.
    pub(crate) fn note(&self, error: &CudaError) {
        if error.fault() == Fault::ContextLost {
            self.lost.store(true, Ordering::Release);
        }
    }
}

/// A non-blocking CUDA stream, destroyed when the last holder drops it.
pub struct Stream {
    ctx: Arc<Context>,
    raw: sys::CUstream,
}

// SAFETY: a CUstream is a driver handle that the driver API accepts from any thread of the
// process ("Stream Management": no thread affinity is documented); this type adds no state.
unsafe impl Send for Stream {}
// SAFETY: as above; enqueueing on one stream from several threads is serialised by the driver.
unsafe impl Sync for Stream {}

impl fmt::Debug for Stream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stream").field("raw", &self.raw).finish()
    }
}

impl Stream {
    /// A new stream that does not synchronise with the legacy default stream
    /// (`CU_STREAM_NON_BLOCKING`).
    pub fn new(ctx: &Arc<Context>) -> Result<Arc<Stream>, CudaError> {
        ctx.bind()?;
        let raw = ctx.check(
            "cuStreamCreate",
            result::stream::create(result::stream::StreamKind::NonBlocking),
        )?;
        Ok(Arc::new(Stream {
            ctx: ctx.clone(),
            raw,
        }))
    }

    /// The context the stream belongs to.
    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }

    /// The driver handle, for libraries that take a `cudaStream_t` (the same handle).
    pub(crate) fn raw(&self) -> sys::CUstream {
        self.raw
    }

    /// Block until every command queued on the stream has completed.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.bind()?;
        // SAFETY: the stream is live (owned by this value).
        self.ctx.check("cuStreamSynchronize", unsafe {
            result::stream::synchronize(self.raw)
        })
    }

    /// Fill `len` bytes of `dst` from `offset` with `value`, in stream order.
    pub fn memset(
        &self,
        dst: &dyn DeviceRange,
        offset: usize,
        len: usize,
        value: u8,
    ) -> Result<(), CudaError> {
        let addr = dst.checked_addr(offset, len)?;
        self.ctx.bind()?;
        // SAFETY: `checked_addr` proved [addr, addr + len) lies inside a live device allocation.
        self.ctx.check("cuMemsetD8Async", unsafe {
            result::memset_d8_async(addr, value, len, self.raw)
        })
    }

    /// Copy `src` from host memory into `dst` at `offset`, in stream order. A pageable source is
    /// staged by the driver before this returns, so `src` may be reused at once; the device
    /// write completes only in stream order ("API synchronization behavior",
    /// https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html).
    pub fn copy_from_host(
        &self,
        dst: &dyn DeviceRange,
        offset: usize,
        src: &[u8],
    ) -> Result<(), CudaError> {
        let addr = dst.checked_addr(offset, src.len())?;
        self.ctx.bind()?;
        // SAFETY: the destination range is inside a live allocation (`checked_addr`); `src` is a
        // live host slice of exactly the copied length, staged before the call returns.
        self.ctx.check("cuMemcpyHtoDAsync", unsafe {
            result::memcpy_htod_async(addr, src, self.raw)
        })
    }

    /// Copy `dst.len()` bytes of `src` from `offset` into pinned host memory. The bytes are
    /// readable through the returned guard once the stream was waited for.
    pub fn copy_to_pinned<'a>(
        &'a self,
        src: &dyn DeviceRange,
        offset: usize,
        dst: &'a mut PinnedBuf,
        len: usize,
    ) -> Result<PendingCopy<'a>, CudaError> {
        if len > dst.len {
            return Err(CudaError::invalid(format!(
                "{len} bytes into a pinned buffer of {}",
                dst.len
            )));
        }
        let addr = src.checked_addr(offset, len)?;
        self.ctx.bind()?;
        // SAFETY: the source range is inside a live device allocation; the destination is `len`
        // bytes of a live pinned allocation, borrowed mutably by the guard until the stream was
        // waited for, so no host code reads it while the copy runs.
        let rc = unsafe {
            sys::cuMemcpyDtoHAsync_v2(dst.ptr.cast::<c_void>(), addr, len, self.raw).result()
        };
        self.ctx.check("cuMemcpyDtoHAsync", rc)?;
        Ok(PendingCopy {
            stream: self,
            buf: Some(dst),
            len,
        })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        // A failed bind leaves the stream to the context's teardown; there is no caller to tell.
        if self.ctx.bind().is_ok() {
            // SAFETY: the stream is live and dropped once; work still queued completes before
            // the driver releases it (cuStreamDestroy).
            let _ = unsafe { result::stream::destroy(self.raw) };
        }
    }
}

/// A device-to-host copy in flight; [`PendingCopy::wait`] returns the bytes.
pub struct PendingCopy<'a> {
    stream: &'a Stream,
    buf: Option<&'a mut PinnedBuf>,
    len: usize,
}

impl<'a> PendingCopy<'a> {
    /// Wait for the stream, then read the copied bytes.
    pub fn wait(mut self) -> Result<&'a [u8], CudaError> {
        self.stream.synchronize()?;
        match self.buf.take() {
            Some(buf) => Ok(&buf.as_slice()[..self.len]),
            None => Err(CudaError::invalid("copy already collected")),
        }
    }
}

impl Drop for PendingCopy<'_> {
    fn drop(&mut self) {
        if self.buf.is_some() {
            // The borrow of the host buffer ends here: the copy must have landed first.
            let _ = self.stream.synchronize();
        }
    }
}

/// A range of device memory a copy or kernel may address.
pub trait DeviceRange {
    /// The device address of the first byte.
    fn device_addr(&self) -> u64;
    /// The length in bytes.
    fn byte_len(&self) -> usize;

    /// The address of `[offset, offset + len)`, refused when it leaves the range.
    fn checked_addr(&self, offset: usize, len: usize) -> Result<u64, CudaError> {
        match offset.checked_add(len) {
            Some(end) if end <= self.byte_len() => Ok(self.device_addr() + offset as u64),
            _ => Err(CudaError::invalid(format!(
                "bytes {offset}..+{len} outside a range of {}",
                self.byte_len()
            ))),
        }
    }
}

/// Page-locked host memory (`cuMemHostAlloc`), freed on drop.
pub struct PinnedBuf {
    ctx: Arc<Context>,
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the buffer is plain host memory owned by this value; access goes through `&`/`&mut`.
unsafe impl Send for PinnedBuf {}
// SAFETY: shared access only reads (`as_slice`).
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    /// `len` bytes of pinned host memory (at least one).
    pub fn new(ctx: &Arc<Context>, len: usize) -> Result<PinnedBuf, CudaError> {
        ctx.bind()?;
        // SAFETY: flags 0 is a plain page-locked allocation; the pointer is owned from here on.
        let ptr = ctx.check("cuMemHostAlloc", unsafe {
            result::malloc_host(len.max(1), 0)
        })?;
        let ptr = ptr.cast::<u8>();
        // SAFETY: `ptr` is a fresh allocation of at least `len` bytes; zero-filling it makes every
        // byte initialised before any slice of it exists.
        unsafe { std::ptr::write_bytes(ptr, 0, len) };
        Ok(PinnedBuf {
            ctx: ctx.clone(),
            ptr,
            len,
        })
    }

    /// The length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes; copies into the buffer are read only through [`PendingCopy::wait`].
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` holds `len` initialised bytes (zero-filled at allocation, then written
        // only by copies that completed before their guard released the mutable borrow).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if self.ctx.bind().is_ok() {
            // SAFETY: allocated by cuMemHostAlloc above, freed once; no copy is in flight because
            // every copy into it is waited for before its guard releases the borrow.
            let _ = unsafe { result::free_host(self.ptr.cast::<c_void>()) };
        }
    }
}

/// A stream-ordered memory pool of one device (`cuMemPoolCreate`).
pub(crate) struct MemPool {
    ctx: Arc<Context>,
    raw: sys::CUmemoryPool,
}

// SAFETY: a memory pool is a driver handle usable from any thread; allocation and free calls on
// it are thread safe (stream-ordered allocator documentation).
unsafe impl Send for MemPool {}
// SAFETY: as above.
unsafe impl Sync for MemPool {}

impl MemPool {
    /// A pool of device memory on the context's device that keeps up to `keep_bytes` of freed
    /// memory cached (`CU_MEMPOOL_ATTR_RELEASE_THRESHOLD`) instead of returning it at each
    /// synchronisation.
    pub(crate) fn new(ctx: &Arc<Context>, keep_bytes: u64) -> Result<MemPool, CudaError> {
        ctx.bind()?;
        // SAFETY: all-zero is a valid `CUmemPoolProps` (every enum has a zero variant, the
        // pointer is null, the sizes are 0); the fields the pool needs are set below.
        let mut props: sys::CUmemPoolProps = unsafe { MaybeUninit::zeroed().assume_init() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        props.location.__bindgen_anon_1.id = ctx.ordinal() as c_int;
        // SAFETY: `props` is a valid, fully initialised description.
        let raw = ctx.check("cuMemPoolCreate", unsafe {
            result::mem_pool::create(&props)
        })?;
        let pool = MemPool {
            ctx: ctx.clone(),
            raw,
        };
        let mut keep = keep_bytes;
        // SAFETY: the release threshold attribute takes a `cuuint64_t`; `keep` lives for the call.
        ctx.check("cuMemPoolSetAttribute", unsafe {
            result::mem_pool::set_attribute(
                pool.raw,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut keep as *mut u64).cast::<c_void>(),
            )
        })?;
        Ok(pool)
    }

    /// Allocate `bytes` in stream order on `stream`.
    ///
    /// # Safety
    /// The memory is uninitialised and usable in order after this call on `stream`; on another
    /// stream only after `stream` was waited for. The caller frees it once with
    /// [`MemPool::free_raw`].
    pub(crate) unsafe fn alloc_raw(
        &self,
        bytes: usize,
        stream: sys::CUstream,
    ) -> Result<u64, CudaError> {
        self.ctx.bind()?;
        // SAFETY: the pool is live; the stream is the caller's live stream (this fn's contract).
        self.ctx.check("cuMemAllocFromPoolAsync", unsafe {
            result::mem_pool::alloc_async(self.raw, bytes, stream)
        })
    }

    /// Free an allocation of this pool in stream order on `stream`.
    ///
    /// # Safety
    /// `ptr` came from [`MemPool::alloc_raw`] of this pool and is freed once; work on other
    /// streams that uses it has been waited for.
    pub(crate) unsafe fn free_raw(&self, ptr: u64, stream: sys::CUstream) -> Result<(), CudaError> {
        self.ctx.bind()?;
        // SAFETY: the caller's contract above.
        self.ctx
            .check("cuMemFreeAsync", unsafe { result::free_async(ptr, stream) })
    }

    /// `bytes` from the pool in order on `stream`, freed in order on the same stream on drop.
    pub(crate) fn alloc(
        self: &Arc<Self>,
        bytes: usize,
        stream: &Arc<Stream>,
    ) -> Result<PoolAlloc, CudaError> {
        // SAFETY: the allocation is owned by the returned value, which keeps its stream and frees
        // it once, on that stream, when dropped.
        let ptr = unsafe { self.alloc_raw(bytes, stream.raw()) }?;
        Ok(PoolAlloc {
            pool: self.clone(),
            stream: stream.clone(),
            ptr,
        })
    }

    pub(crate) fn context(&self) -> &Arc<Context> {
        &self.ctx
    }
}

/// One allocation of a [`MemPool`], owned: freed on its stream when dropped.
pub(crate) struct PoolAlloc {
    pool: Arc<MemPool>,
    stream: Arc<Stream>,
    ptr: u64,
}

impl PoolAlloc {
    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }

    pub(crate) fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

impl Drop for PoolAlloc {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from this pool and is freed once, in order on the stream it was
        // allocated on; work on other streams was waited for by the owner (frame_pool.rs).
        let _ = unsafe { self.pool.free_raw(self.ptr, self.stream.raw()) };
    }
}

impl Drop for MemPool {
    fn drop(&mut self) {
        if self.ctx.bind().is_ok() {
            // SAFETY: the pool is live and destroyed once; outstanding allocations keep the
            // driver's resources until they are freed (cuMemPoolDestroy).
            let _ = unsafe { result::mem_pool::destroy(self.raw) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(u64, usize);
    impl DeviceRange for Fake {
        fn device_addr(&self) -> u64 {
            self.0
        }
        fn byte_len(&self) -> usize {
            self.1
        }
    }

    #[test]
    fn device_ranges_refuse_bytes_outside_them() {
        let r = Fake(4096, 100);
        assert_eq!(r.checked_addr(0, 100), Ok(4096));
        assert_eq!(r.checked_addr(60, 40), Ok(4156));
        assert!(r.checked_addr(60, 41).is_err());
        assert!(r.checked_addr(usize::MAX, 2).is_err());
    }
}
