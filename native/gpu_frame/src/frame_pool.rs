//! Device buffers from one stream-ordered memory pool, each charged to the engine's cap
//! (plan 17 section 2.3, items 2 and the table's rows charged to the cap).
//!
//! A buffer is allocated on a stream and freed on that same stream when dropped
//! (`cuMemAllocFromPoolAsync`, `cuMemFreeAsync`): work queued on that stream before the drop
//! completes before the memory is reused. Work on any other stream that touched the buffer must
//! have been waited for before the drop; that is the owner's rule, as in the Python engine
//! (`engine.py`: "every path that drops the frame has waited for this stream"). The buffer keeps
//! its stream alive, so the stream cannot be destroyed first.
//!
//! The pool keeps up to the cap of freed memory cached (its release threshold), so the device
//! memory it holds stays within the cap while the count of live bytes is the cap itself
//! (stream-ordered allocator: https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MALLOC__ASYNC.html).

use std::fmt;
use std::sync::Arc;

use crate::budget::{CapExceeded, Charge, DeviceCap};
use crate::cuda::{Context, CudaError, DeviceRange, Fault, MemPool, PoolAlloc, Stream};

/// A device allocation that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocError {
    /// The engine's own cap refused it.
    Cap(CapExceeded),
    /// The driver refused it (`CUDA_ERROR_OUT_OF_MEMORY`, or another code).
    Cuda(CudaError),
}

impl AllocError {
    /// Whether the photo answers `gpu_memory`.
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            AllocError::Cap(_) => true,
            AllocError::Cuda(e) => e.fault() == Fault::OutOfMemory,
        }
    }
}

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AllocError::Cap(e) => e.fmt(f),
            AllocError::Cuda(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for AllocError {}

impl From<CudaError> for AllocError {
    fn from(e: CudaError) -> Self {
        AllocError::Cuda(e)
    }
}

/// The pool every frame, scratch buffer and weight set comes from, and the cap they count
/// against. nvJPEG's lane buffers come from it too (`nvjpeg/ffi.rs`).
#[derive(Clone)]
pub struct FramePool {
    pub(crate) pool: Arc<MemPool>,
    cap: Arc<DeviceCap>,
}

impl fmt::Debug for FramePool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FramePool").field("cap", &self.cap).finish()
    }
}

impl FramePool {
    /// A pool on the context's device whose live bytes are capped by `cap`.
    pub fn new(ctx: &Arc<Context>, cap: Arc<DeviceCap>) -> Result<FramePool, CudaError> {
        let pool = MemPool::new(ctx, cap.limit() as u64)?;
        Ok(FramePool {
            pool: Arc::new(pool),
            cap,
        })
    }

    /// The cap this pool charges.
    pub fn cap(&self) -> &Arc<DeviceCap> {
        &self.cap
    }

    /// The context of the pool's device.
    pub fn context(&self) -> &Arc<Context> {
        self.pool.context()
    }

    /// `bytes` of uninitialised device memory (at least one byte is allocated), usable in order
    /// on `stream`, charged to the cap until dropped.
    pub fn alloc(&self, bytes: usize, stream: &Arc<Stream>) -> Result<DeviceBuf, AllocError> {
        let size = bytes.max(1);
        let charge = self.cap.charge(size).map_err(AllocError::Cap)?;
        let alloc = self.pool.alloc(size, stream)?;
        Ok(DeviceBuf {
            alloc,
            len: bytes,
            _charge: charge,
        })
    }

    /// A buffer holding `data`, readable from any stream once this returns: the copy is queued
    /// on `stream` and the stream is waited for before the buffer is handed out. Kernels on other
    /// streams would otherwise read it before the copy lands (the throwaway pipeline's
    /// illegal-address crash with weight sets shared across lanes; "API synchronization
    /// behavior", https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html).
    pub fn upload(&self, data: &[u8], stream: &Arc<Stream>) -> Result<DeviceBuf, AllocError> {
        let buf = self.alloc(data.len(), stream)?;
        stream.copy_from_host(&buf, 0, data)?;
        stream.synchronize()?;
        Ok(buf)
    }
}

/// Device memory from the [`FramePool`], charged to the cap, freed on its stream when dropped.
pub struct DeviceBuf {
    alloc: PoolAlloc,
    len: usize,
    // Declared after `alloc`: the bytes are freed before the cap is credited.
    _charge: Charge,
}

impl fmt::Debug for DeviceBuf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuf")
            .field("ptr", &self.alloc.ptr())
            .field("len", &self.len)
            .finish()
    }
}

impl DeviceBuf {
    /// The stream the buffer was allocated on and is freed on.
    pub fn stream(&self) -> &Arc<Stream> {
        self.alloc.stream()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl DeviceRange for DeviceBuf {
    fn device_addr(&self) -> u64 {
        self.alloc.ptr()
    }

    fn byte_len(&self) -> usize {
        self.len
    }
}
