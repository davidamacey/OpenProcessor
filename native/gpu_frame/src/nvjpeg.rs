//! Decode lanes: OS threads, each owning one nvJPEG lane and one CUDA stream, decoding with the
//! decoupled API (plan 17 sections 2.0 and 2.1; the Python engine's `nvjpeg_lanes.py`).
//!
//! A lane parses the file, asks each backend in turn whether it decodes it (`GPU_HYBRID`, then
//! `HYBRID`), takes the frame from the engine's [`FramePool`] on the lane's stream and runs the
//! host phase, the transfer and the device phase, with interpolated chroma upsampling and RGBI
//! output (HWC, `width x 3` bytes per row). nvJPEG ignores EXIF orientation: the caller orients.
//!
//! The encoded bytes are a reference-counted `Bytes` held by the lane until its stream was
//! synchronised: nvJPEG keeps pointers into them (`save_stream = 0`). [`Lane::decode`] does not
//! wait, so the caller can queue its own kernels on the lane's stream and wait once
//! ([`Lane::synchronize`]); [`Lanes::decode`] is decode then wait, in one job.
//!
//! After an nvJPEG failure the lane's stream is waited for (the frame goes back to a pool shared
//! with other lanes, nothing may still write it) and the wait's answer tells a lost context from
//! the photo's own failure, as the Python engine's `_check_context` does.

mod ffi;
pub mod status;

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;

use crate::budget::HostCount;
use crate::cuda::{CudaError, DeviceRange, Fault, Stream};
use crate::frame_pool::{AllocError, DeviceBuf, FramePool};
use crate::pool::{CancelToken, JobError, JobHandle, PoolError, WorkerPool};
use ffi::{LaneObjects, Shared, StreamInfo};
pub use status::{Backend, NvjpegFailure, NvjpegStatus};

/// Why a lane gave no frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// nvJPEG could not be loaded or lacks a symbol (start-up only).
    Library(String),
    /// No backend decodes this stream (the API decodes it on the host).
    Unsupported,
    /// An nvJPEG call failed.
    Nvjpeg(NvjpegFailure),
    /// The frame's allocation failed.
    Alloc(AllocError),
    /// A CUDA call failed, or the stream wait after an nvJPEG failure did.
    Cuda(CudaError),
    /// The lane thread could not run the job (cancelled, panicked, shut down).
    Job(JobError),
}

impl DecodeError {
    /// Whether the photo answers `gpu_memory` (`prep_budget.is_memory_error`).
    pub fn is_out_of_memory(&self) -> bool {
        match self {
            DecodeError::Nvjpeg(f) => f.status == NvjpegStatus::ALLOCATOR_FAILURE,
            DecodeError::Alloc(e) => e.is_out_of_memory(),
            DecodeError::Cuda(e) => e.fault() == Fault::OutOfMemory,
            _ => false,
        }
    }

    /// Whether the context is lost (refuse work, exit for a restart).
    pub fn is_context_lost(&self) -> bool {
        match self {
            DecodeError::Cuda(e) | DecodeError::Alloc(AllocError::Cuda(e)) => {
                e.fault() == Fault::ContextLost
            }
            _ => false,
        }
    }
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Library(m) => write!(f, "nvJPEG: {m}"),
            DecodeError::Unsupported => write!(f, "no nvJPEG backend supports this stream"),
            DecodeError::Nvjpeg(e) => e.fmt(f),
            DecodeError::Alloc(e) => e.fmt(f),
            DecodeError::Cuda(e) => e.fmt(f),
            DecodeError::Job(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for DecodeError {}

/// A decoded frame: interleaved RGB, `width x 3` bytes per row, on the device.
#[derive(Debug)]
pub struct DeviceFrame {
    buf: DeviceBuf,
    width: u32,
    height: u32,
}

impl DeviceFrame {
    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Bytes per row.
    pub fn pitch(&self) -> usize {
        self.width as usize * 3
    }

    /// The buffer (its stream is the lane's).
    pub fn buffer(&self) -> &DeviceBuf {
        &self.buf
    }

    /// The buffer, to keep after the frame's dimensions are no longer needed.
    pub fn into_buffer(self) -> DeviceBuf {
        self.buf
    }
}

impl DeviceRange for DeviceFrame {
    fn device_addr(&self) -> u64 {
        self.buf.device_addr()
    }

    fn byte_len(&self) -> usize {
        self.buf.byte_len()
    }
}

/// What a decode produced.
#[derive(Debug)]
pub struct Decoded {
    pub frame: DeviceFrame,
    /// Colour components in the file (1 for greyscale, 3 for colour); the frame is RGB either way.
    pub components: u32,
    pub backend: Backend,
}

/// One decode lane, owned by its thread.
pub struct Lane {
    index: usize,
    objects: LaneObjects,
    stream: Arc<Stream>,
    frames: FramePool,
    /// Encoded inputs nvJPEG may still read, released at the next stream wait.
    held: Vec<Bytes>,
}

impl fmt::Debug for Lane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lane")
            .field("index", &self.index)
            .field("held", &self.held.len())
            .finish()
    }
}

impl Lane {
    /// The lane's number in its pool.
    pub fn index(&self) -> usize {
        self.index
    }

    /// The lane's stream: the frame is written in order on it; queue further work here.
    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// Queue the decode of `data` on the lane's stream (no wait). `data` is held until
    /// [`Lane::synchronize`].
    pub fn decode(&mut self, data: Bytes) -> Result<Decoded, DecodeError> {
        self.held.push(data.clone());
        let info = self
            .objects
            .parse(&data)
            .map_err(|e| self.after_failure(e))?;
        for backend in Backend::ORDER {
            if !self
                .objects
                .supported(backend)
                .map_err(|e| self.after_failure(e))?
            {
                continue;
            }
            let frame = self.alloc_frame(info)?;
            self.objects
                .decode(
                    backend,
                    frame.device_addr(),
                    frame.byte_len(),
                    info,
                    self.stream.raw(),
                )
                .map_err(|e| self.after_failure(e))?;
            return Ok(Decoded {
                frame,
                components: info.components,
                backend,
            });
        }
        Err(DecodeError::Unsupported)
    }

    /// Wait for the lane's stream, then release the encoded inputs nvJPEG read.
    pub fn synchronize(&mut self) -> Result<(), DecodeError> {
        let waited = self.stream.synchronize();
        if waited.is_ok() {
            self.held.clear();
        }
        waited.map_err(DecodeError::Cuda)
    }

    fn alloc_frame(&self, info: StreamInfo) -> Result<DeviceFrame, DecodeError> {
        let bytes = (info.width as usize)
            .checked_mul(info.height as usize)
            .and_then(|p| p.checked_mul(3))
            .ok_or(DecodeError::Unsupported)?;
        let buf = self
            .frames
            .alloc(bytes, &self.stream)
            .map_err(DecodeError::Alloc)?;
        Ok(DeviceFrame {
            buf,
            width: info.width,
            height: info.height,
        })
    }

    /// An nvJPEG failure is the photo's only while the context still works: wait for the stream
    /// and let its answer decide.
    fn after_failure(&mut self, failure: NvjpegFailure) -> DecodeError {
        match self.synchronize() {
            Err(DecodeError::Cuda(e)) if e.fault() == Fault::ContextLost => DecodeError::Cuda(e),
            _ => DecodeError::Nvjpeg(failure),
        }
    }
}

impl Drop for Lane {
    fn drop(&mut self) {
        // nvJPEG's objects and the held inputs go only after the stream finished with them.
        let _ = self.stream.synchronize();
    }
}

/// The decode lanes of one device: a [`WorkerPool`] of [`Lane`]s sharing one nvJPEG handle.
pub struct Lanes {
    pool: WorkerPool<Lane>,
    version: (i32, i32, i32),
}

impl fmt::Debug for Lanes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lanes")
            .field("lanes", &self.pool.threads())
            .field("nvjpeg", &self.version)
            .finish()
    }
}

impl Lanes {
    /// Load nvJPEG (from `library_dir` first when given) and start `count` lanes whose frames
    /// and nvJPEG device buffers come from `frames` (charged to its cap); nvJPEG's pinned bytes
    /// are counted in `host`.
    pub fn new(
        frames: &FramePool,
        host: &Arc<HostCount>,
        count: usize,
        library_dir: Option<&Path>,
    ) -> Result<Lanes, DecodeError> {
        let ctx = frames.context().clone();
        ctx.bind().map_err(DecodeError::Cuda)?;
        let shared = Arc::new(
            Shared::new(library_dir, &ctx, &frames.pool, frames.cap(), host)
                .map_err(DecodeError::Library)?,
        );
        let version = shared.version();
        let frames = frames.clone();
        let pool = WorkerPool::new("gpu-frame-lane", count, move |index| {
            ctx.bind().map_err(DecodeError::Cuda)?;
            let stream = Stream::new(&ctx).map_err(DecodeError::Cuda)?;
            let objects = LaneObjects::new(shared.clone()).map_err(DecodeError::Nvjpeg)?;
            Ok(Lane {
                index,
                objects,
                stream,
                frames: frames.clone(),
                held: Vec::new(),
            })
        })
        .map_err(|e| match e {
            PoolError::Init(e) => e,
            PoolError::Panicked(m) => {
                DecodeError::Library(format!("a lane panicked at start: {m}"))
            }
            PoolError::Spawn(e) => {
                DecodeError::Library(format!("a lane thread could not start: {e}"))
            }
        })?;
        Ok(Lanes { pool, version })
    }

    /// Decode `data` on the next free lane and wait for its stream.
    pub fn decode(
        &self,
        data: Bytes,
        cancel: &CancelToken,
    ) -> Result<JobHandle<Result<Decoded, DecodeError>>, JobError> {
        self.run(cancel, move |lane| {
            let decoded = lane.decode(data)?;
            lane.synchronize()?;
            Ok(decoded)
        })
    }

    /// Run `job` on the next free lane (decode, then the caller's own work on the lane's
    /// stream, then one wait).
    pub fn run<T, F>(&self, cancel: &CancelToken, job: F) -> Result<JobHandle<T>, JobError>
    where
        T: Send + 'static,
        F: FnOnce(&mut Lane) -> T + Send + 'static,
    {
        self.pool.submit(cancel, job)
    }

    /// The number of lanes.
    pub fn count(&self) -> usize {
        self.pool.threads()
    }

    /// nvJPEG's version (major, minor, patch).
    pub fn library_version(&self) -> (i32, i32, i32) {
        self.version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_and_lost_context_are_told_apart() {
        let oom = DecodeError::Nvjpeg(NvjpegFailure {
            call: "nvjpegDecodeJpegHost",
            status: NvjpegStatus(5),
        });
        assert!(oom.is_out_of_memory());
        let bad = DecodeError::Nvjpeg(NvjpegFailure {
            call: "nvjpegJpegStreamParse",
            status: NvjpegStatus(3),
        });
        assert!(!bad.is_out_of_memory() && !bad.is_context_lost());
        let cap = DecodeError::Alloc(AllocError::Cap(crate::budget::CapExceeded {
            need: 9,
            in_use: 1,
            limit: 4,
        }));
        assert!(cap.is_out_of_memory());
        let lost = DecodeError::Cuda(CudaError::Driver {
            call: "cuStreamSynchronize",
            code: crate::cuda::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS,
        });
        assert!(lost.is_context_lost() && !lost.is_out_of_memory());
    }
}
