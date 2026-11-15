//! The region: one device allocation exported to the model server, whose slices carry every model
//! input (plan 17 section 2.3 item 3; the Python engine's `transport.py`).
//!
//! * It is one synchronous `cuMemAlloc`, never pool memory: memory from a stream-ordered pool
//!   cannot be exported with `cuIpcGetMemHandle`, which needs the base of a `cuMemAlloc`
//!   allocation (https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html,
//!   `cuIpcGetMemHandle`). cudarc's `result::malloc_sync` is that call.
//! * Its 64-byte IPC handle is what the model server's `CudaSharedMemoryRegister` takes as
//!   `raw_handle` over gRPC.
//! * It is freed only when nothing may still import it: the memory is released when the last
//!   holder (the region or a slice) is dropped AND the model server holds no registration of it.
//!   A region still registered when dropped is left to process exit: freeing exported memory
//!   before the importer closed it is undefined behaviour (CUDA documentation of
//!   `cuIpcCloseMemHandle`), so "unregister, then free" is the only order this type allows.
//! * A slice goes back to the slab when dropped. Its holder drops it only after its call ended
//!   and the stream that wrote it was waited for.

use std::fmt;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cudarc::driver::{result, sys};

use crate::cuda::{Context, CudaError, DeviceRange};
use crate::slab::{ALIGN, Slab, SlabUsage, SliceError};

/// The size of a CUDA IPC memory handle (`CU_IPC_HANDLE_SIZE`).
pub const IPC_HANDLE_BYTES: usize = 64;

struct RegionMemory {
    ctx: Arc<Context>,
    ptr: u64,
    len: usize,
    handle: [u8; IPC_HANDLE_BYTES],
    slab: Slab,
    /// The model server may hold an import: set by a successful register, cleared by a
    /// successful unregister (`model_server.rs`).
    registered: Arc<AtomicBool>,
}

impl Drop for RegionMemory {
    fn drop(&mut self) {
        if self.registered.load(Ordering::Acquire) {
            // The model server may still map it: left to process exit (see the module docs).
            return;
        }
        if self.ctx.bind().is_ok() {
            // SAFETY: `ptr` is the base of this region's own cuMemAlloc allocation, freed once;
            // no slice remains (each holds this value), and no importer holds it (checked above).
            let _ = unsafe { result::free_sync(self.ptr) };
        }
    }
}

/// The exported allocation and its slab.
#[derive(Clone)]
pub struct Region {
    mem: Arc<RegionMemory>,
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Region")
            .field("len", &self.mem.len)
            .field("registered", &self.is_registered())
            .finish()
    }
}

impl Region {
    /// Allocate `bytes` (a positive multiple of 256) and export it.
    pub fn new(ctx: &Arc<Context>, bytes: usize) -> Result<Region, CudaError> {
        let slab = Slab::new(bytes).ok_or_else(|| {
            CudaError::invalid(format!(
                "region of {bytes} bytes: not a positive multiple of {ALIGN}"
            ))
        })?;
        ctx.bind()?;
        // SAFETY: a plain synchronous allocation; the pointer is owned by `RegionMemory` from the
        // next statement on, which frees it.
        let ptr = ctx.check("cuMemAlloc", unsafe { result::malloc_sync(bytes) })?;
        let mut mem = RegionMemory {
            ctx: ctx.clone(),
            ptr,
            len: bytes,
            handle: [0; IPC_HANDLE_BYTES],
            slab,
            registered: Arc::new(AtomicBool::new(false)),
        };
        let mut handle = MaybeUninit::<sys::CUipcMemHandle>::zeroed();
        // SAFETY: `ptr` is the base of a live cuMemAlloc allocation of this context (the only
        // kind cuIpcGetMemHandle exports); the out-parameter is valid for one handle.
        let rc = unsafe { sys::cuIpcGetMemHandle(handle.as_mut_ptr(), ptr) };
        ctx.check("cuIpcGetMemHandle", rc.result())?;
        // SAFETY: the call succeeded, so the handle was written; it was zeroed before in any case.
        let handle = unsafe { handle.assume_init() };
        for (dst, src) in mem.handle.iter_mut().zip(handle.reserved.iter()) {
            *dst = *src as u8;
        }
        Ok(Region { mem: Arc::new(mem) })
    }

    /// The serialized IPC handle the model server's register call takes as `raw_handle`.
    pub fn raw_handle(&self) -> &[u8; IPC_HANDLE_BYTES] {
        &self.mem.handle
    }

    /// The region's size in bytes.
    pub fn len(&self) -> usize {
        self.mem.len
    }

    /// Whether the region has no bytes (never: the size is positive).
    pub fn is_empty(&self) -> bool {
        self.mem.len == 0
    }

    /// The context the region belongs to.
    pub fn context(&self) -> &Arc<Context> {
        &self.mem.ctx
    }

    /// A slice of `bytes`, waiting at most `wait` for room; for async callers.
    pub async fn slice(&self, bytes: usize, wait: Duration) -> Result<Slice, SliceError> {
        let offset = self.mem.slab.take(bytes, wait).await?;
        Ok(Slice {
            mem: self.mem.clone(),
            offset,
            len: bytes,
        })
    }

    /// The same for a pool thread.
    pub fn slice_blocking(&self, bytes: usize, wait: Duration) -> Result<Slice, SliceError> {
        let offset = self.mem.slab.take_blocking(bytes, wait)?;
        Ok(Slice {
            mem: self.mem.clone(),
            offset,
            len: bytes,
        })
    }

    /// What the slab holds now.
    pub fn usage(&self) -> SlabUsage {
        self.mem.slab.usage()
    }

    /// Whether the model server may hold a registration of the region.
    pub fn is_registered(&self) -> bool {
        self.mem.registered.load(Ordering::Acquire)
    }

    /// The flag the registration sets and clears (`model_server.rs`).
    pub(crate) fn registration_flag(&self) -> Arc<AtomicBool> {
        self.mem.registered.clone()
    }

    /// Whether `slice` is a slice of this region.
    pub fn owns(&self, slice: &Slice) -> bool {
        Arc::ptr_eq(&self.mem, &slice.mem)
    }
}

/// `len` bytes of the region at `offset`, given back to the slab when dropped.
pub struct Slice {
    mem: Arc<RegionMemory>,
    offset: usize,
    len: usize,
}

impl fmt::Debug for Slice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Slice")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

impl Slice {
    /// The offset inside the region (the model call's `shared_memory_offset`).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The length asked for (the model call's `shared_memory_byte_size`).
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl DeviceRange for Slice {
    fn device_addr(&self) -> u64 {
        self.mem.ptr + self.offset as u64
    }

    fn byte_len(&self) -> usize {
        self.len
    }
}

impl Drop for Slice {
    fn drop(&mut self) {
        self.mem.slab.give(self.offset, self.len);
    }
}
