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
//! Driver API reference: <https://docs.nvidia.com/cuda/cuda-driver-api/index.html> (context
//! management, stream management, memory management, stream-ordered memory allocator, module
//! management, execution control); NVRTC: <https://docs.nvidia.com/cuda/nvrtc/index.html>;
//! cudarc 0.19.10: <https://docs.rs/cudarc/0.19.10/cudarc/driver/result/index.html>.

use std::ffi::{CString, c_char, c_int, c_uint, c_void};
use std::fmt;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::driver::{CudaContext, result, sys};
use cudarc::nvrtc::{result as nvrtc_result, sys as nvrtc_sys};

use crate::kernels::{KernelArg, LaunchConfig};

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
    ///
    /// # Errors
    ///
    /// [`CudaError::LibraryMissing`] when libcuda cannot be loaded, [`CudaError::Driver`] when the
    /// driver cannot retain the device's primary context (a bad ordinal, a lost card).
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
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the driver refuses the call (cuCtxSetCurrent); a sticky code
    /// marks the context lost.
    pub fn bind(&self) -> Result<(), CudaError> {
        self.check("cuCtxSetCurrent", self.inner.bind_to_thread())
    }

    /// The device ordinal.
    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    /// The device's compute capability (major, minor).
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the driver refuses the call (cuDeviceGetAttribute); a sticky code
    /// marks the context lost.
    pub fn compute_capability(&self) -> Result<(i32, i32), CudaError> {
        self.check("cuDeviceGetAttribute", self.inner.compute_capability())
    }

    /// The device's name.
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the driver refuses the call (cuDeviceGetName); a sticky code
    /// marks the context lost.
    pub fn device_name(&self) -> Result<String, CudaError> {
        self.check("cuDeviceGetName", self.inner.name())
    }

    /// The device's PCI bus id (`0000:65:00.0`), which names the card whatever the ordinal.
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the driver refuses the call (the PCI bus id query); a sticky code
    /// marks the context lost.
    pub fn pci_bus_id(&self) -> Result<String, CudaError> {
        bus_id_of(self.inner.cu_device()).inspect_err(|e| self.note(e))
    }

    /// Free and total device memory in bytes (`cuMemGetInfo`), for the whole device.
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the driver refuses the call (cuMemGetInfo or the binding of the
    /// context); a sticky code marks the context lost.
    pub fn memory_info(&self) -> Result<(usize, usize), CudaError> {
        self.bind()?;
        self.check("cuMemGetInfo", result::mem_get_info())
    }

    /// Wait for all work of the context (`cuCtxSynchronize`): the probe that tells a photo's
    /// failure from a lost context.
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the driver refuses the call (cuCtxSynchronize or the binding of
    /// the context); a sticky code marks the context lost.
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

fn bus_id_of(device: sys::CUdevice) -> Result<String, CudaError> {
    let mut buf = [0 as c_char; 32];
    // SAFETY: the buffer holds `len` bytes, the documented minimum is 13; the device handle comes
    // from cuDeviceGet or a retained context.
    let rc = unsafe { sys::cuDeviceGetPCIBusId(buf.as_mut_ptr(), buf.len() as c_int, device) };
    driver("cuDeviceGetPCIBusId", rc.result())?;
    let bytes: Vec<u8> = buf
        .iter()
        .take_while(|c| **c != 0)
        .map(|c| *c as u8)
        .collect();
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// The name and PCI bus id of device `ordinal` (as `CUDA_VISIBLE_DEVICES` numbers it), read
/// without creating a context: nothing is allocated on the card, so a process can check it was
/// given the card it expects before it takes any memory there (`cuInit`, `cuDeviceGet`,
/// `cuDeviceGetName`, `cuDeviceGetPCIBusId` need no context:
/// <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html>).
///
/// # Errors
///
/// [`CudaError::LibraryMissing`] when libcuda cannot be loaded, [`CudaError::Driver`] when `cuInit`
/// or a device query fails, [`CudaError::Invalid`] when `ordinal` does not fit a C int.
pub fn device_identity(ordinal: usize) -> Result<(String, String), CudaError> {
    // SAFETY: probing the loader has no precondition; it only tries to open the library.
    if !unsafe { sys::is_culib_present() } {
        return Err(CudaError::LibraryMissing {
            library: "CUDA driver (libcuda)",
        });
    }
    driver("cuInit", result::init())?;
    let ordinal = c_int::try_from(ordinal)
        .map_err(|_| CudaError::invalid(format!("device ordinal {ordinal}")))?;
    let device = driver("cuDeviceGet", result::device::get(ordinal))?;
    let name = driver("cuDeviceGetName", result::device::get_name(device))?;
    Ok((name, bus_id_of(device)?))
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
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the context cannot be bound or the driver refuses the stream.
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
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the context cannot be bound or a queued command failed.
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.ctx.bind()?;
        // SAFETY: the stream is live (owned by this value).
        self.ctx.check("cuStreamSynchronize", unsafe {
            result::stream::synchronize(self.raw)
        })
    }

    /// Fill `len` bytes of `dst` from `offset` with `value`, in stream order.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] when `offset + len` leaves `dst`; [`CudaError::Driver`] when the
    /// driver refuses the memset.
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
    /// <https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html>).
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] when `src` does not fit `dst` at `offset`; [`CudaError::Driver`] when
    /// the driver refuses the copy.
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
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] when `len` is larger than `dst` or the range leaves `src`;
    /// [`CudaError::Driver`] when the driver refuses the copy.
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

    /// Copy a rectangle of `height` rows of `width_bytes` each, the first at `offset` of `src`
    /// and the next ones `src_pitch` bytes apart, into the front of pinned host memory with the
    /// rows packed (`width_bytes` apart): one `cuMemcpy2DAsync` for a crop of a frame, where a
    /// copy per row or of the whole row band would move many more bytes over the bus
    /// (<https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html>, `cuMemcpy2DAsync`).
    /// The bytes are readable through the returned guard once the stream was waited for.
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] when the rectangle does not fit the pitch, overflows, or is larger
    /// than `dst`, or leaves `src`; [`CudaError::Driver`] when the driver refuses the copy.
    pub fn copy_rect_to_pinned<'a>(
        &'a self,
        src: &dyn DeviceRange,
        offset: usize,
        src_pitch: usize,
        width_bytes: usize,
        height: usize,
        dst: &'a mut PinnedBuf,
    ) -> Result<PendingCopy<'a>, CudaError> {
        let len = rect_bytes(src_pitch, width_bytes, height)?;
        if len > dst.len {
            return Err(CudaError::invalid(format!(
                "{len} bytes into a pinned buffer of {}",
                dst.len
            )));
        }
        let addr = src.checked_addr(offset, rect_span(src_pitch, width_bytes, height)?)?;
        // SAFETY: all-zero is a valid `CUDA_MEMCPY2D` (null pointers, zero sizes, the zero memory
        // type); every field the copy reads is set below.
        let mut copy: sys::CUDA_MEMCPY2D = unsafe { MaybeUninit::zeroed().assume_init() };
        copy.srcMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = addr;
        copy.srcPitch = src_pitch;
        copy.dstMemoryType = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
        copy.dstHost = dst.ptr.cast::<c_void>();
        copy.dstPitch = width_bytes;
        copy.WidthInBytes = width_bytes;
        copy.Height = height;
        self.ctx.bind()?;
        // SAFETY: every source row lies inside a live device allocation (`checked_addr` over the
        // whole span); the destination is `len` bytes of a live pinned allocation, borrowed
        // mutably by the guard until the stream was waited for; `copy` lives for the call (the
        // driver reads the description before it returns).
        let rc = unsafe { sys::cuMemcpy2DAsync_v2(&copy, self.raw).result() };
        self.ctx.check("cuMemcpy2DAsync", rc)?;
        Ok(PendingCopy {
            stream: self,
            buf: Some(dst),
            len,
        })
    }
}

/// The packed bytes of a rectangle; refused when a row is wider than the pitch or it overflows.
fn rect_bytes(pitch: usize, width_bytes: usize, height: usize) -> Result<usize, CudaError> {
    if width_bytes > pitch {
        return Err(CudaError::invalid(format!(
            "rows of {width_bytes} bytes do not fit a pitch of {pitch}"
        )));
    }
    width_bytes
        .checked_mul(height)
        .ok_or_else(|| CudaError::invalid("rectangle size overflows"))
}

/// The source bytes a rectangle reaches from its first byte: `height - 1` pitches and one row.
fn rect_span(pitch: usize, width_bytes: usize, height: usize) -> Result<usize, CudaError> {
    match height {
        0 => Ok(0),
        h => pitch
            .checked_mul(h - 1)
            .and_then(|n| n.checked_add(width_bytes))
            .ok_or_else(|| CudaError::invalid("rectangle span overflows")),
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
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the stream wait fails; [`CudaError::Invalid`] when the copy was
    /// already collected.
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
    ///
    /// # Errors
    ///
    /// [`CudaError::Invalid`] when `offset + len` overflows or passes [`DeviceRange::byte_len`].
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
    ///
    /// # Errors
    ///
    /// [`CudaError::Driver`] when the context cannot be bound or the page-locked allocation fails.
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

/// NVRTC's version (major, minor), or `LibraryMissing`.
pub(crate) fn nvrtc_version() -> Result<(i32, i32), CudaError> {
    // SAFETY: probing the loader has no precondition.
    if !unsafe { nvrtc_sys::is_culib_present() } {
        return Err(CudaError::LibraryMissing {
            library: "NVRTC (libnvrtc)",
        });
    }
    let (mut major, mut minor) = (0 as c_int, 0 as c_int);
    // SAFETY: both out-parameters are valid for a write of their type.
    let rc = unsafe { nvrtc_sys::nvrtcVersion(&mut major, &mut minor) };
    nvrtc_status("nvrtcVersion", rc, String::new())?;
    Ok((major, minor))
}

fn nvrtc_status(
    call: &'static str,
    rc: nvrtc_sys::nvrtcResult,
    log: String,
) -> Result<(), CudaError> {
    match rc {
        nvrtc_sys::nvrtcResult::NVRTC_SUCCESS => Ok(()),
        other => Err(CudaError::Nvrtc {
            call,
            code: other as i32,
            log,
        }),
    }
}

/// Compile CUDA C `source` (program `name`) with `options` into a cubin
/// (`nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetCUBINSize`, `nvrtcGetCUBIN`).
pub(crate) fn nvrtc_compile(
    source: &str,
    name: &str,
    options: &[String],
) -> Result<Vec<u8>, CudaError> {
    nvrtc_version()?;
    let src = CString::new(source).map_err(|_| CudaError::invalid("NUL byte in kernel source"))?;
    let cname = CString::new(name).map_err(|_| CudaError::invalid("NUL byte in program name"))?;
    let prog = nvrtc_result::create_program(&src, Some(&cname)).map_err(|e| CudaError::Nvrtc {
        call: "nvrtcCreateProgram",
        code: e.0 as i32,
        log: String::new(),
    })?;
    let out = compile_program(prog, options);
    // SAFETY: `prog` was created above and is destroyed once, after its last use.
    let _ = unsafe { nvrtc_result::destroy_program(prog) };
    out
}

fn compile_program(
    prog: nvrtc_sys::nvrtcProgram,
    options: &[String],
) -> Result<Vec<u8>, CudaError> {
    // SAFETY: `prog` is a live program; the options are owned strings converted to C strings by
    // cudarc for the duration of the call.
    if let Err(e) = unsafe { nvrtc_result::compile_program(prog, options) } {
        // SAFETY: the program is live; the log exists after a failed compile.
        let log = unsafe { nvrtc_result::get_program_log(prog) }.unwrap_or_default();
        let log: Vec<u8> = log
            .iter()
            .take_while(|c| **c != 0)
            .map(|c| *c as u8)
            .collect();
        return Err(CudaError::Nvrtc {
            call: "nvrtcCompileProgram",
            code: e.0 as i32,
            log: String::from_utf8_lossy(&log).into_owned(),
        });
    }
    let mut size = 0usize;
    // SAFETY: the program compiled; `size` is a valid out-parameter.
    let rc = unsafe { nvrtc_sys::nvrtcGetCUBINSize(prog, &mut size) };
    nvrtc_status("nvrtcGetCUBINSize", rc, String::new())?;
    let mut cubin = vec![0u8; size];
    // SAFETY: `cubin` holds exactly the size NVRTC reported for this program's cubin.
    let rc = unsafe { nvrtc_sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr().cast::<c_char>()) };
    nvrtc_status("nvrtcGetCUBIN", rc, String::new())?;
    Ok(cubin)
}

/// A module loaded into the context (`cuModuleLoadData`), unloaded on drop.
pub(crate) struct LoadedModule {
    ctx: Arc<Context>,
    raw: sys::CUmodule,
}

// SAFETY: a module handle is usable from any thread of the context's process.
unsafe impl Send for LoadedModule {}
// SAFETY: as above; function lookups and launches on it are thread safe in the driver.
unsafe impl Sync for LoadedModule {}

impl LoadedModule {
    /// Load a cubin.
    pub(crate) fn load(ctx: &Arc<Context>, cubin: &[u8]) -> Result<Self, CudaError> {
        if cubin.is_empty() {
            return Err(CudaError::invalid("an empty cubin"));
        }
        ctx.bind()?;
        // SAFETY: `cubin` is a complete image from nvrtcGetCUBIN, alive for the call; the driver
        // copies what it keeps.
        let raw = ctx.check("cuModuleLoadData", unsafe {
            result::module::load_data(cubin.as_ptr().cast::<c_void>())
        })?;
        Ok(LoadedModule {
            ctx: ctx.clone(),
            raw,
        })
    }

    /// The kernel `name` of this module; it keeps the module loaded.
    pub(crate) fn function(self: &Arc<Self>, name: &str) -> Result<LoadedFunction, CudaError> {
        let cname =
            CString::new(name).map_err(|_| CudaError::invalid("NUL byte in kernel name"))?;
        self.ctx.bind()?;
        // SAFETY: the module is live; the returned handle is valid while it stays loaded, which
        // the `LoadedFunction` guarantees by holding an `Arc` of this module.
        let raw = self.ctx.check("cuModuleGetFunction", unsafe {
            result::module::get_function(self.raw, cname)
        })?;
        Ok(LoadedFunction {
            module: self.clone(),
            raw,
        })
    }
}

/// One kernel of a loaded module; keeps the module loaded.
pub(crate) struct LoadedFunction {
    module: Arc<LoadedModule>,
    raw: sys::CUfunction,
}

// SAFETY: a CUfunction handle is valid from any thread while its module is loaded, which the
// `Arc` guarantees; launches are thread safe in the driver.
unsafe impl Send for LoadedFunction {}
// SAFETY: as above.
unsafe impl Sync for LoadedFunction {}

impl LoadedFunction {
    /// Launch on `stream` with `args` passed by value.
    pub(crate) fn launch(
        &self,
        cfg: &LaunchConfig,
        stream: &Stream,
        args: &[KernelArg],
    ) -> Result<(), CudaError> {
        let ctx = &self.module.ctx;
        let mut values: Vec<KernelArg> = args.to_vec();
        let mut params: Vec<*mut c_void> = values.iter_mut().map(KernelArg::as_param).collect();
        ctx.bind()?;
        // SAFETY: the function belongs to a live module (held by `self`); each parameter points
        // at one value of `values`, alive and unmoved for the call (the driver copies them at
        // launch); the argument list is the caller's statement of the kernel's signature
        // (`Module::launch`).
        ctx.check("cuLaunchKernel", unsafe {
            result::launch_kernel(
                self.raw,
                cfg.grid,
                cfg.block,
                cfg.shared_bytes as c_uint,
                stream.raw(),
                &mut params,
            )
        })
    }
}

impl Drop for LoadedModule {
    fn drop(&mut self) {
        if self.ctx.bind().is_ok() {
            // SAFETY: the module is live and unloaded once; no holder of its functions remains
            // (they keep the module alive).
            let _ = unsafe { result::module::unload(self.raw) };
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

    #[test]
    fn a_rectangle_reaches_its_last_row_only_and_packs_its_rows() {
        // three rows of 6 bytes in a frame of 30-byte rows: 2 pitches and one row from the start
        assert_eq!(rect_span(30, 6, 3), Ok(66));
        assert_eq!(rect_bytes(30, 6, 3), Ok(18));
        assert_eq!(rect_span(30, 6, 0), Ok(0));
        assert!(rect_bytes(5, 6, 1).is_err(), "a row wider than its pitch");
        assert!(rect_span(usize::MAX, 6, 3).is_err());
        assert!(rect_bytes(usize::MAX, usize::MAX, 2).is_err());
        // a crop at offset 60 of a 126-byte range fits exactly; one row more does not
        let frame = Fake(4096, 126);
        assert_eq!(
            frame.checked_addr(60, rect_span(30, 6, 3).expect("span")),
            Ok(4156)
        );
        assert!(
            frame
                .checked_addr(60, rect_span(30, 6, 4).expect("span"))
                .is_err()
        );
    }
}
