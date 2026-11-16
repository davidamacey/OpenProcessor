//! Every unsafe call into nvJPEG: the library loaded at run time, owning types for its objects
//! (each create paired with its destroy in a `Drop`), and the counting allocators.
//!
//! Derived from the nvJPEG part of `src/decode/nvjpeg/ffi.rs` of ortloom, the owner's Rust
//! serving project, at commit 59fbb42, used here under its MIT licence:
//!
//! > Copyright (c) 2026 David Macey
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a copy of this
//! > software and associated documentation files (the "Software"), to deal in the Software
//! > without restriction, including without limitation the rights to use, copy, modify, merge,
//! > publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
//! > to whom the Software is furnished to do so, subject to the following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included in all copies or
//! > substantial portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
//! > INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//! > PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
//! > FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//! > OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//! > DEALINGS IN THE SOFTWARE.
//!
//! Taken: the symbol table resolved through `libloading` (`fn_table!`), the leaked library
//! handle, the status names, the decoupled decode sequence and the drop order of states before
//! their buffers. Changed: the CUDA runtime, NVRTC and driver tables are not taken (streams and
//! memory come from cudarc, `cuda.rs`); handles are created with the V2 allocators so every byte
//! nvJPEG allocates is counted (device bytes from the engine's pool against its cap, pinned host
//! bytes in a [`HostCount`]); each lane keeps a pinned and a device buffer per backend state, as
//! the Python engine's `nvjpeg_lanes.py` does.
//!
//! Signatures are those of `nvjpeg.h` in the `nvidia-nvjpeg` 13.2.3.58 wheel the engine image
//! ships (`NVJPEG_VER` 13.2.3.58); handles are pointers, enums C `int`s. nvJPEG documentation:
//! <https://docs.nvidia.com/cuda/nvjpeg/index.html> (1.3 thread safety: the handle is shared,
//! states, streams and buffers are per thread, user allocators must be thread safe; 2.2.8 and
//! 2.2.11 the V2 allocators; 2.3.1.6 `nvjpegCreateExV2`; decoupled decoding).

use std::ffi::{c_int, c_uint, c_void};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use cudarc::driver::{result, sys};

use crate::budget::{DeviceCap, HostCount};
use crate::cuda::{Context, MemPool};
use crate::nvjpeg::status::{Backend, NvjpegFailure, NvjpegStatus, check};

pub(super) type Status = c_int;
pub(super) type Handle = *mut c_void;
type Decoder = *mut c_void;
type State = *mut c_void;
type JpegStream = *mut c_void;
type DecodeParams = *mut c_void;
type BufferPinned = *mut c_void;
type BufferDevice = *mut c_void;
pub(super) type CudaStream = *mut c_void;
pub(super) type EncoderState = *mut c_void;
pub(super) type EncoderParams = *mut c_void;

/// `nvjpegImage_t`.
#[repr(C)]
pub(super) struct NvjpegImage {
    pub(super) channel: [*mut u8; 4],
    pub(super) pitch: [usize; 4],
}

type DevMalloc = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, usize, CudaStream) -> c_int;
type DevFree = unsafe extern "C" fn(*mut c_void, *mut c_void, usize, CudaStream) -> c_int;

/// `nvjpegDevAllocatorV2_t`.
#[repr(C)]
struct DevAllocatorV2 {
    dev_malloc: DevMalloc,
    dev_free: DevFree,
    dev_ctx: *mut c_void,
}

/// `nvjpegPinnedAllocatorV2_t` (the same callback shapes).
#[repr(C)]
struct PinnedAllocatorV2 {
    pinned_malloc: DevMalloc,
    pinned_free: DevFree,
    pinned_ctx: *mut c_void,
}

const NVJPEG_BACKEND_DEFAULT: c_int = 0;
const NVJPEG_OUTPUT_RGBI: c_int = 5;
const NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION: c_uint = 1 << 5;

/// The shared library's name, as the CUDA 13 toolkit and the `nvidia-nvjpeg` wheel install it.
pub const NVJPEG_SONAME: &str = "libnvjpeg.so.13";

/// Declares a table of C entry points and a loader that resolves every one of them or names the
/// first that is missing.
macro_rules! fn_table {
    ($vis:vis $table:ident { $($name:ident : fn($($arg:ty),*) -> $ret:ty;)* }) => {
        #[expect(non_snake_case, reason = "the fields are the C entry points, named as nvJPEG names them")]
        $vis struct $table {
            $($vis $name: unsafe extern "C" fn($($arg),*) -> $ret,)*
        }

        impl $table {
            /// # Safety
            /// `lib` stays loaded for as long as the table is used (it is leaked), and the
            /// declared signatures match the library's.
            unsafe fn load(lib: &libloading::Library) -> Result<Self, &'static str> {
                Ok($table {
                    // SAFETY: the caller's contract above.
                    $($name: *unsafe { lib.get::<unsafe extern "C" fn($($arg),*) -> $ret>(stringify!($name)) }
                        .map_err(|_| stringify!($name))?,)*
                })
            }
        }
    };
}

fn_table!(Fns {
    nvjpegCreateExV2: fn(c_int, *mut DevAllocatorV2, *mut PinnedAllocatorV2, c_uint, *mut Handle) -> Status;
    nvjpegDestroy: fn(Handle) -> Status;
    nvjpegDecoderCreate: fn(Handle, c_int, *mut Decoder) -> Status;
    nvjpegDecoderDestroy: fn(Decoder) -> Status;
    nvjpegDecoderStateCreate: fn(Handle, Decoder, *mut State) -> Status;
    nvjpegJpegStateDestroy: fn(State) -> Status;
    nvjpegJpegStreamCreate: fn(Handle, *mut JpegStream) -> Status;
    nvjpegJpegStreamDestroy: fn(JpegStream) -> Status;
    nvjpegJpegStreamParse: fn(Handle, *const u8, usize, c_int, c_int, JpegStream) -> Status;
    nvjpegJpegStreamGetFrameDimensions: fn(JpegStream, *mut c_uint, *mut c_uint) -> Status;
    nvjpegJpegStreamGetComponentsNum: fn(JpegStream, *mut c_uint) -> Status;
    nvjpegDecodeParamsCreate: fn(Handle, *mut DecodeParams) -> Status;
    nvjpegDecodeParamsDestroy: fn(DecodeParams) -> Status;
    nvjpegDecodeParamsSetOutputFormat: fn(DecodeParams, c_int) -> Status;
    nvjpegDecoderJpegSupported: fn(Decoder, JpegStream, DecodeParams, *mut c_int) -> Status;
    nvjpegBufferPinnedCreateV2: fn(Handle, *mut PinnedAllocatorV2, *mut BufferPinned) -> Status;
    nvjpegBufferPinnedDestroy: fn(BufferPinned) -> Status;
    nvjpegBufferDeviceCreateV2: fn(Handle, *mut DevAllocatorV2, *mut BufferDevice) -> Status;
    nvjpegBufferDeviceDestroy: fn(BufferDevice) -> Status;
    nvjpegStateAttachPinnedBuffer: fn(State, BufferPinned) -> Status;
    nvjpegStateAttachDeviceBuffer: fn(State, BufferDevice) -> Status;
    nvjpegDecodeJpegHost: fn(Handle, Decoder, State, DecodeParams, JpegStream) -> Status;
    nvjpegDecodeJpegTransferToDevice: fn(Handle, Decoder, State, JpegStream, CudaStream) -> Status;
    nvjpegDecodeJpegDevice: fn(Handle, Decoder, State, *mut NvjpegImage, CudaStream) -> Status;
});

// The encoder's entry points (nvJPEG documentation 2.3.5, "JPEG Encoding"), a table of their own
// so a test can record the calls without filling the decoder's entries; loaded from the same
// library handle as `Fns`.
fn_table!(pub(super) EncFns {
    nvjpegEncoderStateCreate: fn(Handle, *mut EncoderState, CudaStream) -> Status;
    nvjpegEncoderStateDestroy: fn(EncoderState) -> Status;
    nvjpegEncoderParamsCreate: fn(Handle, *mut EncoderParams, CudaStream) -> Status;
    nvjpegEncoderParamsDestroy: fn(EncoderParams) -> Status;
    nvjpegEncoderParamsSetQuality: fn(EncoderParams, c_int, CudaStream) -> Status;
    nvjpegEncoderParamsSetEncoding: fn(EncoderParams, c_int, CudaStream) -> Status;
    nvjpegEncoderParamsSetOptimizedHuffman: fn(EncoderParams, c_int, CudaStream) -> Status;
    nvjpegEncoderParamsSetSamplingFactors: fn(EncoderParams, c_int, CudaStream) -> Status;
    nvjpegEncodeImage: fn(Handle, EncoderState, EncoderParams, *const NvjpegImage, c_int, c_int, c_int, CudaStream) -> Status;
    nvjpegEncodeRetrieveBitstream: fn(Handle, EncoderState, *mut u8, *mut usize, CudaStream) -> Status;
});

/// Load `soname` from `dir` when given and present there, else from the loader path. The handle
/// is leaked: every function pointer taken from it must stay valid for the life of the process,
/// and unloading a library that ran CUDA initialisers is worse than a leaked handle.
fn open(soname: &str, dir: Option<&Path>) -> Result<&'static libloading::Library, String> {
    let candidate = dir.map(|d| d.join(soname)).filter(|p| p.exists());
    let lib = match &candidate {
        // SAFETY: loading a shared library runs its initialisers; this is NVIDIA's nvJPEG.
        Some(path) => unsafe { libloading::Library::new(path) },
        // SAFETY: as above.
        None => unsafe { libloading::Library::new(soname) },
    }
    .map_err(|e| format!("{soname}: {e}"))?;
    Ok(Box::leak(Box::new(lib)))
}

/// The library's version (`nvjpegGetProperty` MAJOR, MINOR, PATCH), read before the rest of the
/// table so a missing symbol can name the version that lacks it.
fn version_of(lib: &libloading::Library) -> Result<(i32, i32, i32), String> {
    type GetProperty = unsafe extern "C" fn(c_int, *mut c_int) -> Status;
    // SAFETY: the library is leaked (`open`); the signature is the header's.
    let get = *unsafe { lib.get::<GetProperty>("nvjpegGetProperty") }
        .map_err(|e| format!("{NVJPEG_SONAME}: nvjpegGetProperty: {e}"))?;
    let mut v = [0 as c_int; 3];
    for (kind, slot) in v.iter_mut().enumerate() {
        // SAFETY: `libraryPropertyType` MAJOR/MINOR/PATCH are 0/1/2; `slot` is a valid out-param.
        check("nvjpegGetProperty", unsafe { get(kind as c_int, slot) })
            .map_err(|e| e.to_string())?;
    }
    Ok((v[0], v[1], v[2]))
}

/// What the allocator callbacks reach through their context pointer.
struct Counters {
    ctx: Arc<Context>,
    pool: Arc<MemPool>,
    cap: Arc<DeviceCap>,
    host: Arc<HostCount>,
}

/// Device bytes for nvJPEG: charged to the cap, taken from the engine's pool on nvJPEG's stream,
/// usable on any stream once this returns (the allocating stream is waited for). Answers
/// non-zero on failure, which nvJPEG reports as `NVJPEG_STATUS_ALLOCATOR_FAILURE`.
unsafe extern "C" fn dev_malloc(
    ctx: *mut c_void,
    out: *mut *mut c_void,
    size: usize,
    stream: CudaStream,
) -> c_int {
    // SAFETY: `ctx` points at the `Counters` held by the `Shared` that created the handle, which
    // outlives every object of the handle (they hold the `Shared`); access is shared, read only.
    let c = unsafe { &*ctx.cast::<Counters>() };
    if c.ctx.bind().is_err() || c.cap.reserve(size).is_err() {
        return 1;
    }
    let raw_stream = stream.cast::<sys::CUstream_st>();
    // SAFETY: the pool is live; nvJPEG passes its stream or the legacy default stream; the
    // allocation is handed to nvJPEG, which frees it once through `dev_free`.
    let ptr = match unsafe { c.pool.alloc_raw(size.max(1), raw_stream) } {
        Ok(p) => p,
        Err(_) => {
            c.cap.release(size);
            return 1;
        }
    };
    // SAFETY: the stream is the one the allocation was ordered on; waiting makes the memory
    // usable from the lane's own stream too.
    let waited = unsafe { result::stream::synchronize(raw_stream) };
    if c.ctx.check("cuStreamSynchronize", waited).is_err() {
        // Freed in order on the same stream; nvJPEG never saw the pointer.
        // SAFETY: `ptr` came from this pool just above and is freed once.
        let _ = unsafe { c.pool.free_raw(ptr, raw_stream) };
        c.cap.release(size);
        return 1;
    }
    // SAFETY: `out` is nvJPEG's valid out-parameter.
    unsafe { *out = ptr as *mut c_void };
    0
}

/// Gives back what `dev_malloc` allocated. The whole context is waited for first, so no stream
/// still reads the buffer whatever stream nvJPEG passes, as the default `cudaFree` would ensure.
unsafe extern "C" fn dev_free(
    ctx: *mut c_void,
    ptr: *mut c_void,
    size: usize,
    stream: CudaStream,
) -> c_int {
    // SAFETY: as in `dev_malloc`.
    let c = unsafe { &*ctx.cast::<Counters>() };
    if c.ctx.synchronize().is_err() {
        return 1;
    }
    // SAFETY: `ptr` came from `dev_malloc` on this pool and is freed once; nothing uses it (the
    // context was synchronised above).
    let freed = unsafe {
        c.pool
            .free_raw(ptr as u64, stream.cast::<sys::CUstream_st>())
    };
    c.cap.release(size);
    if freed.is_ok() { 0 } else { 1 }
}

/// Pinned host bytes for nvJPEG, counted in the [`HostCount`].
unsafe extern "C" fn pinned_malloc(
    ctx: *mut c_void,
    out: *mut *mut c_void,
    size: usize,
    _stream: CudaStream,
) -> c_int {
    // SAFETY: as in `dev_malloc`.
    let c = unsafe { &*ctx.cast::<Counters>() };
    if c.ctx.bind().is_err() {
        return 1;
    }
    // SAFETY: a plain page-locked allocation handed to nvJPEG, freed once by `pinned_free`.
    match c.ctx.check("cuMemHostAlloc", unsafe {
        result::malloc_host(size.max(1), 0)
    }) {
        Ok(p) => {
            c.host.add(size);
            // SAFETY: `out` is nvJPEG's valid out-parameter.
            unsafe { *out = p };
            0
        }
        Err(_) => 1,
    }
}

/// Gives back what `pinned_malloc` allocated, after the context finished any copy from it.
unsafe extern "C" fn pinned_free(
    ctx: *mut c_void,
    ptr: *mut c_void,
    size: usize,
    _stream: CudaStream,
) -> c_int {
    // SAFETY: as in `dev_malloc`.
    let c = unsafe { &*ctx.cast::<Counters>() };
    if c.ctx.synchronize().is_err() {
        return 1;
    }
    // SAFETY: `ptr` came from `pinned_malloc` and is freed once, with no copy in flight.
    let freed = c
        .ctx
        .check("cuMemFreeHost", unsafe { result::free_host(ptr) });
    c.host.sub(size);
    if freed.is_ok() { 0 } else { 1 }
}

/// The library, one handle created with the counting allocators, and one decoder per backend.
/// The handle and the decoders are shared by every lane: the handle is documented thread safe;
/// sharing the decoders is what the Python lanes and the throwaway pipeline do (practice, not a
/// documented guarantee; plan 17 section 2.1).
pub(crate) struct Shared {
    fns: Fns,
    enc: EncFns,
    handle: Handle,
    decoders: [(Backend, Decoder); 2],
    version: (i32, i32, i32),
    // The callbacks' context and allocator descriptions, at fixed addresses for the handle's life.
    dev_alloc: Box<DevAllocatorV2>,
    pinned_alloc: Box<PinnedAllocatorV2>,
    _counters: Arc<Counters>,
}

// SAFETY: the handle and decoders are shared across threads as documented (handle) and practised
// (decoders), see above; the boxes are never mutated after creation; the counters are atomics.
unsafe impl Send for Shared {}
// SAFETY: as above.
unsafe impl Sync for Shared {}

impl Shared {
    /// Load nvJPEG (from `dir` first when given) and create the handle and decoders. The calling
    /// thread must have `ctx` bound.
    pub(crate) fn new(
        dir: Option<&Path>,
        ctx: &Arc<Context>,
        pool: &Arc<MemPool>,
        cap: &Arc<DeviceCap>,
        host: &Arc<HostCount>,
    ) -> Result<Shared, String> {
        let lib = open(NVJPEG_SONAME, dir)?;
        let version = version_of(lib)?;
        let lacks = |missing: &str| {
            format!(
                "{NVJPEG_SONAME} version {}.{}.{} lacks {missing} (the engine needs the V2 allocator and encoder API)",
                version.0, version.1, version.2
            )
        };
        // SAFETY: the library is leaked; the signatures are those of the shipped header.
        let fns = unsafe { Fns::load(lib) }.map_err(lacks)?;
        // SAFETY: as above.
        let enc = unsafe { EncFns::load(lib) }.map_err(lacks)?;
        let counters = Arc::new(Counters {
            ctx: ctx.clone(),
            pool: pool.clone(),
            cap: cap.clone(),
            host: host.clone(),
        });
        // The callbacks only read through this pointer (atomics and shared handles).
        let cptr: *mut c_void = Arc::as_ptr(&counters).cast_mut().cast();
        let mut dev_alloc = Box::new(DevAllocatorV2 {
            dev_malloc,
            dev_free,
            dev_ctx: cptr,
        });
        let mut pinned_alloc = Box::new(PinnedAllocatorV2 {
            pinned_malloc,
            pinned_free,
            pinned_ctx: cptr,
        });
        let mut handle: Handle = ptr::null_mut();
        // SAFETY: both allocator descriptions and their context live in boxes owned by the value
        // returned below, which destroys the handle before dropping them; `handle` is a valid
        // out-parameter.
        check("nvjpegCreateExV2", unsafe {
            (fns.nvjpegCreateExV2)(
                NVJPEG_BACKEND_DEFAULT,
                &mut *dev_alloc,
                &mut *pinned_alloc,
                NVJPEG_FLAGS_UPSAMPLING_WITH_INTERPOLATION,
                &mut handle,
            )
        })
        .map_err(|e| e.to_string())?;
        let mut shared = Shared {
            fns,
            enc,
            handle,
            decoders: [
                (Backend::GpuHybrid, ptr::null_mut()),
                (Backend::Hybrid, ptr::null_mut()),
            ],
            version,
            dev_alloc,
            pinned_alloc,
            _counters: counters,
        };
        for i in 0..shared.decoders.len() {
            let backend = shared.decoders[i].0;
            let mut d: Decoder = ptr::null_mut();
            // SAFETY: the handle is live; `d` is a valid out-parameter. On failure `shared` is
            // dropped, destroying exactly what was created (null decoders are skipped).
            check("nvjpegDecoderCreate", unsafe {
                (shared.fns.nvjpegDecoderCreate)(shared.handle, backend.raw(), &mut d)
            })
            .map_err(|e| e.to_string())?;
            shared.decoders[i].1 = d;
        }
        Ok(shared)
    }

    pub(crate) fn version(&self) -> (i32, i32, i32) {
        self.version
    }

    /// The encoder's entry points and the handle they take.
    pub(super) fn encoder(&self) -> (&EncFns, Handle) {
        (&self.enc, self.handle)
    }

    fn decoder(&self, backend: Backend) -> Decoder {
        match backend {
            Backend::GpuHybrid => self.decoders[0].1,
            Backend::Hybrid => self.decoders[1].1,
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // The allocator callbacks bind the context themselves; the allocator descriptions and
        // their context are fields, dropped only after this body destroyed the handle.
        // SAFETY: every lane holds an `Arc<Shared>`, so none is alive; decoders go before the
        // handle; null decoders (a failed start) are skipped.
        unsafe {
            for (_, d) in &self.decoders {
                if !d.is_null() {
                    (self.fns.nvjpegDecoderDestroy)(*d);
                }
            }
            if !self.handle.is_null() {
                (self.fns.nvjpegDestroy)(self.handle);
            }
        }
    }
}

/// What nvJPEG read from a JPEG's headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub components: u32,
}

/// One lane's nvJPEG objects: a state per backend, each with its own pinned and device buffer
/// (`nvjpeg_lanes.py`), one parsed-stream object and the decode parameters (RGBI output). Used by
/// one thread at a time: `Send`, not `Sync`.
pub(crate) struct LaneObjects {
    shared: Arc<Shared>,
    states: [State; 2],
    pinned: [BufferPinned; 2],
    device: [BufferDevice; 2],
    stream: JpegStream,
    params: DecodeParams,
}

// SAFETY: nvJPEG states, streams and buffers may move between threads as long as one thread uses
// them at a time (documentation 1.3), which `&mut self` on every method enforces.
unsafe impl Send for LaneObjects {}

impl LaneObjects {
    /// Create a lane's objects; the calling thread must have the context bound.
    pub(crate) fn new(shared: Arc<Shared>) -> Result<LaneObjects, NvjpegFailure> {
        let mut lane = LaneObjects {
            shared,
            states: [ptr::null_mut(); 2],
            pinned: [ptr::null_mut(); 2],
            device: [ptr::null_mut(); 2],
            stream: ptr::null_mut(),
            params: ptr::null_mut(),
        };
        let sh = lane.shared.clone();
        let f = &sh.fns;
        // SAFETY: every call gets the live handle and a valid out-parameter; the allocator
        // descriptions live in `sh`; on failure `lane` is dropped and destroys exactly the
        // objects created so far (null ones are skipped).
        unsafe {
            for (i, backend) in Backend::ORDER.iter().enumerate() {
                check(
                    "nvjpegDecoderStateCreate",
                    (f.nvjpegDecoderStateCreate)(
                        sh.handle,
                        sh.decoder(*backend),
                        &mut lane.states[i],
                    ),
                )?;
                let pinned_alloc = (&*sh.pinned_alloc as *const PinnedAllocatorV2).cast_mut();
                check(
                    "nvjpegBufferPinnedCreateV2",
                    (f.nvjpegBufferPinnedCreateV2)(sh.handle, pinned_alloc, &mut lane.pinned[i]),
                )?;
                let dev_alloc = (&*sh.dev_alloc as *const DevAllocatorV2).cast_mut();
                check(
                    "nvjpegBufferDeviceCreateV2",
                    (f.nvjpegBufferDeviceCreateV2)(sh.handle, dev_alloc, &mut lane.device[i]),
                )?;
                check(
                    "nvjpegStateAttachPinnedBuffer",
                    (f.nvjpegStateAttachPinnedBuffer)(lane.states[i], lane.pinned[i]),
                )?;
                check(
                    "nvjpegStateAttachDeviceBuffer",
                    (f.nvjpegStateAttachDeviceBuffer)(lane.states[i], lane.device[i]),
                )?;
            }
            check(
                "nvjpegJpegStreamCreate",
                (f.nvjpegJpegStreamCreate)(sh.handle, &mut lane.stream),
            )?;
            check(
                "nvjpegDecodeParamsCreate",
                (f.nvjpegDecodeParamsCreate)(sh.handle, &mut lane.params),
            )?;
            check(
                "nvjpegDecodeParamsSetOutputFormat",
                (f.nvjpegDecodeParamsSetOutputFormat)(lane.params, NVJPEG_OUTPUT_RGBI),
            )?;
        }
        Ok(lane)
    }

    /// Parse `data`'s headers into the lane's stream object. With `save_stream = 0` the object
    /// keeps pointers into `data`: the caller keeps `data` alive and unchanged until the lane's
    /// CUDA stream was synchronised after the decode (`nvjpeg.rs` holds it).
    pub(crate) fn parse(&mut self, data: &[u8]) -> Result<StreamInfo, NvjpegFailure> {
        let f = &self.shared.fns;
        let (mut w, mut h, mut n) = (0 as c_uint, 0 as c_uint, 0 as c_uint);
        // SAFETY: `data` is live for the call and, by this method's contract, until the decode's
        // stream wait; the stream object and handle are this lane's live objects.
        unsafe {
            check(
                "nvjpegJpegStreamParse",
                (f.nvjpegJpegStreamParse)(
                    self.shared.handle,
                    data.as_ptr(),
                    data.len(),
                    0,
                    0,
                    self.stream,
                ),
            )?;
            check(
                "nvjpegJpegStreamGetFrameDimensions",
                (f.nvjpegJpegStreamGetFrameDimensions)(self.stream, &mut w, &mut h),
            )?;
            check(
                "nvjpegJpegStreamGetComponentsNum",
                (f.nvjpegJpegStreamGetComponentsNum)(self.stream, &mut n),
            )?;
        }
        Ok(StreamInfo {
            width: w,
            height: h,
            components: n,
        })
    }

    /// Whether `backend` decodes the stream last parsed (nvJPEG answers 0 for supported).
    pub(crate) fn supported(&mut self, backend: Backend) -> Result<bool, NvjpegFailure> {
        let mut answer: c_int = 1;
        // SAFETY: the decoder, stream object and params are live; `answer` is a valid out-param.
        check("nvjpegDecoderJpegSupported", unsafe {
            (self.shared.fns.nvjpegDecoderJpegSupported)(
                self.shared.decoder(backend),
                self.stream,
                self.params,
                &mut answer,
            )
        })?;
        Ok(answer == 0)
    }

    /// Decode the stream last parsed with `backend` into `dst` (interleaved RGB, `pitch` bytes per
    /// row, `pitch x height` bytes) on `cuda_stream`; the device work completes in stream order.
    pub(crate) fn decode(
        &mut self,
        backend: Backend,
        dst: u64,
        dst_len: usize,
        info: StreamInfo,
        cuda_stream: sys::CUstream,
    ) -> Result<(), NvjpegFailure> {
        let pitch = info.width as usize * 3;
        if dst_len < pitch * info.height as usize {
            return Err(NvjpegFailure {
                call: "decode (output smaller than the frame)",
                status: NvjpegStatus(2),
            });
        }
        let index = match backend {
            Backend::GpuHybrid => 0,
            Backend::Hybrid => 1,
        };
        let (f, handle, decoder, state) = (
            &self.shared.fns,
            self.shared.handle,
            self.shared.decoder(backend),
            self.states[index],
        );
        let mut image = NvjpegImage {
            channel: [
                dst as *mut u8,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            ],
            pitch: [pitch, 0, 0, 0],
        };
        let stream: CudaStream = cuda_stream.cast();
        // SAFETY: `image` names a device buffer of at least `pitch x height` bytes (checked
        // above), what RGBI output of the parsed frame needs; state, decoder, params, stream
        // object and CUDA stream belong to this lane; the parsed input stays alive by `parse`'s
        // contract until the stream is synchronised.
        unsafe {
            check(
                "nvjpegDecodeJpegHost",
                (f.nvjpegDecodeJpegHost)(handle, decoder, state, self.params, self.stream),
            )?;
            check(
                "nvjpegDecodeJpegTransferToDevice",
                (f.nvjpegDecodeJpegTransferToDevice)(handle, decoder, state, self.stream, stream),
            )?;
            check(
                "nvjpegDecodeJpegDevice",
                (f.nvjpegDecodeJpegDevice)(handle, decoder, state, &mut image, stream),
            )?;
        }
        Ok(())
    }
}

impl Drop for LaneObjects {
    fn drop(&mut self) {
        let f = &self.shared.fns;
        // SAFETY: each object was created by this lane (null ones are skipped) and the lane's
        // CUDA stream was synchronised by its owner before the drop; states go before the
        // buffers attached to them.
        unsafe {
            for s in self.states {
                if !s.is_null() {
                    (f.nvjpegJpegStateDestroy)(s);
                }
            }
            for b in self.pinned {
                if !b.is_null() {
                    (f.nvjpegBufferPinnedDestroy)(b);
                }
            }
            for b in self.device {
                if !b.is_null() {
                    (f.nvjpegBufferDeviceDestroy)(b);
                }
            }
            if !self.stream.is_null() {
                (f.nvjpegJpegStreamDestroy)(self.stream);
            }
            if !self.params.is_null() {
                (f.nvjpegDecodeParamsDestroy)(self.params);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_library_is_an_error_value() {
        let dir = std::env::temp_dir().join("gpu_frame-no-such-dir");
        assert!(open("libgpu_frame_no_such_library.so.0", Some(&dir)).is_err());
    }
}
