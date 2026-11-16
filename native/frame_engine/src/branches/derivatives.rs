//! `derivatives` (`frame_engine/branches/derivatives.py`): stored renditions (thumbnails) of the
//! oriented sRGB frame, so the gallery shows a photo without reading the original.
//!
//! The geometry is `PIL.Image.thumbnail`'s ([`fit_size`]: the long side fits `side`, never
//! enlarged). A rendition the size of the frame is the frame itself; any other is one Lanczos
//! resample of the whole frame into planar RGB, made interleaved by `kernels/planes_to_rgb.cu`
//! (the copy CuPy makes of the transposed view), then encoded by nvJPEG on the same launch thread
//! and stream (`gpu_frame::Encoder`, the Python engine's settings). The job starts after the
//! round-2 model calls went out so the encodes overlap them; `derivative_jpegs` holds one JPEG
//! per rendition in plan order, and one failed encode fails the photo, not a model call.

use std::sync::Arc;

use bytes::Bytes;
use gpu_frame::{
    DeviceBuf, DeviceRange, EncodeError, Encoder, FramePool, Kernel, KernelArg, Kernels,
    LaunchConfig, Stream,
};

use crate::fault::GpuFault;
use crate::geometry::fit_size;
use crate::kernels::{Filter, FrameRef, PrepKernels, Resample, WeightCache};
use crate::photo::clock::{DERIVATIVE_JPEG, DERIVATIVE_RENDER, JobClock};
use crate::photo::device::Stages;
use crate::pipeline::{RequestState, Tensor};
use crate::wire::plan::RenditionSpec;

/// The output holding the encoded renditions (`wire.DERIVATIVE_JPEGS`).
pub const DERIVATIVE_JPEGS: &str = "derivative_jpegs";
/// The interleave kernel's CUDA source (this engine's own; the Python engine uses CuPy's copy).
pub const INTERLEAVE_SOURCE: &str = include_str!("../../kernels/planes_to_rgb.cu");
/// Threads per block of the interleave.
const THREADS: u32 = 128;

/// One rendition to make: its size, the resample (none when it is the frame) and its quality.
#[derive(Debug, Clone, PartialEq)]
pub struct Rendition {
    /// Width and height of the rendition.
    pub size: (u32, u32),
    /// The resample into planar RGB, or `None` when the frame is encoded as it is.
    pub resample: Option<Resample>,
    /// JPEG quality, 1 to 100.
    pub quality: u32,
}

/// `Derivatives._render`'s geometry for every rendition of a `w x h` frame, in plan order.
///
/// # Errors
/// A size past the kernels' range.
pub fn plan(w: u32, h: u32, specs: &[RenditionSpec]) -> Result<Vec<Rendition>, String> {
    specs
        .iter()
        .map(|spec| {
            let (ow, oh) = fit_size(i64::from(w), i64::from(h), i64::from(spec.side));
            let size = (
                u32::try_from(ow).map_err(|e| e.to_string())?,
                u32::try_from(oh).map_err(|e| e.to_string())?,
            );
            let resample = (size != (w, h)).then(|| Resample {
                roi: [0, 0, i64::from(w), i64::from(h)],
                size: [ow, oh],
                window: [0, 0, ow, oh],
                dst_offset: 0,
                plane: ow * oh,
                row_stride: ow,
                filter: Filter::Lanczos,
            });
            Ok(Rendition {
                size,
                resample,
                quality: spec.quality,
            })
        })
        .collect()
}

/// The rendition sides of the plan, for the frame budget (`need_bytes(header, sides)`).
pub fn sides(specs: &[RenditionSpec]) -> Vec<u32> {
    specs.iter().map(|s| s.side).collect()
}

/// The output tensor: the JPEGs in plan order.
pub fn output(jpegs: Vec<Bytes>) -> (&'static str, Tensor) {
    (DERIVATIVE_JPEGS, Tensor::Bytes(jpegs))
}

/// `ceil(pixels / 128)` blocks of 128 threads.
///
/// # Errors
/// No pixel, or a grid past the launch limits.
pub fn interleave_launch(pixels: usize) -> Result<LaunchConfig, GpuFault> {
    u32::try_from(pixels.div_ceil(THREADS as usize))
        .ok()
        .filter(|g| *g > 0)
        .map(|gx| LaunchConfig {
            grid: (gx, 1, 1),
            block: (THREADS, 1, 1),
            shared_bytes: 0,
        })
        .ok_or_else(|| GpuFault::Failed(format!("no interleave launch for {pixels} pixels")))
}

/// What the kernel's thread `i` reads for output bytes `3i, 3i + 1, 3i + 2`.
pub fn interleave_sources(i: usize, pixels: usize) -> [usize; 3] {
    [i, i + pixels, i + 2 * pixels]
}

/// The kernel on the host, thread by thread: planar `[3, h, w]` to interleaved `[h, w, 3]`.
pub fn interleave_host(planar: &[u8]) -> Vec<u8> {
    let pixels = planar.len() / 3;
    (0..pixels)
        .flat_map(|i| interleave_sources(i, pixels).map(|k| planar[k]))
        .collect()
}

/// The loaded interleave kernel; one instance serves every launch thread.
pub struct InterleaveKernel {
    kernel: Kernel,
}

impl InterleaveKernel {
    /// Compile the kernel for the current device.
    ///
    /// # Errors
    /// NVRTC or the module load fails.
    pub fn load(kernels: &Kernels) -> Result<Self, GpuFault> {
        Ok(Self {
            kernel: kernels
                .load("planes_to_rgb.cu", INTERLEAVE_SOURCE)?
                .function("planes_to_rgb")?,
        })
    }

    /// Queue the interleave of `pixels` from `src` into `dst` (`3 x pixels` bytes each) on `stream`.
    ///
    /// # Errors
    /// A buffer is too small, or the launch fails.
    pub fn launch(
        &self,
        stream: &Arc<Stream>,
        src: &dyn DeviceRange,
        dst: &dyn DeviceRange,
        pixels: usize,
    ) -> Result<(), GpuFault> {
        let args = [
            KernelArg::U64(src.checked_addr(0, 3 * pixels)?),
            KernelArg::U64(dst.checked_addr(0, 3 * pixels)?),
            KernelArg::I64(i64::try_from(pixels).map_err(|e| GpuFault::Failed(e.to_string()))?),
        ];
        self.kernel
            .launch(&interleave_launch(pixels)?, stream, &args)?;
        Ok(())
    }
}

/// The renditions of one photo on the launch pool.
pub struct RenditionJob<F> {
    /// The frame they are made from.
    pub frame: Arc<F>,
    /// What to make, in plan order.
    pub renditions: Vec<Rendition>,
    /// The request the job belongs to, for its stage and cancel state.
    pub request: Arc<RequestState>,
}

/// What a rendition job gave: the JPEGs in plan order and the seconds per stage.
pub struct Rendered {
    /// One JPEG per rendition, in plan order.
    pub jpegs: Vec<Bytes>,
    /// Seconds per stage measured inside the job.
    pub stages: Stages,
}

/// What a launch thread lends a rendition job.
pub struct RenderTools<'a> {
    /// The resample kernels.
    pub kernels: &'a PrepKernels,
    /// The interleave kernel.
    pub interleave: &'a InterleaveKernel,
    /// Where the planar and interleaved pixels and the scratch come from.
    pub frames: &'a FramePool,
    /// The thread's weight sets.
    pub weights: &'a mut WeightCache<DeviceBuf>,
    /// The thread's encoder, on the thread's stream.
    pub encoder: &'a mut Encoder,
}

/// A failed encode under the fault rule: memory answers `gpu_memory`, a lost context fails the
/// engine, anything else fails the request.
fn encode_fault(e: &EncodeError) -> GpuFault {
    if e.is_context_lost() {
        GpuFault::ContextLost(e.to_string())
    } else if e.is_out_of_memory() {
        GpuFault::OutOfMemory(e.to_string())
    } else {
        GpuFault::Failed(e.to_string())
    }
}

/// The planar resample and its interleave, queued on the encoder's stream.
fn render_one(
    t: &mut RenderTools<'_>,
    frame: &dyn FrameRef,
    r: &Resample,
    pixels: usize,
) -> Result<DeviceBuf, GpuFault> {
    let stream = Arc::clone(t.encoder.stream());
    let sets = t.kernels.weights(t.weights, t.frames, &stream, r)?;
    let planar = t.frames.alloc(3 * pixels, &stream)?;
    t.kernels
        .resample(t.frames, &stream, frame, &planar, r, &sets)?;
    let rgb = t.frames.alloc(3 * pixels, &stream)?;
    t.interleave.launch(&stream, &planar, &rgb, pixels)?;
    Ok(rgb)
}

/// Every rendition of the job, rendered and encoded one after the other on the encoder's
/// stream; each encode returns after its stream wait.
///
/// # Errors
/// An allocation, launch or encode fails.
pub fn render(
    mut t: RenderTools<'_>,
    frame: &dyn FrameRef,
    renditions: &[Rendition],
    clock: &mut JobClock,
) -> Result<Vec<Bytes>, GpuFault> {
    let mut jpegs = Vec::with_capacity(renditions.len());
    for r in renditions {
        let pixels = r.size.0 as usize * r.size.1 as usize;
        let rendered = match &r.resample {
            Some(resample) => Some(clock.time(DERIVATIVE_RENDER, || {
                render_one(&mut t, frame, resample, pixels)
            })?),
            None => None,
        };
        let source: &dyn DeviceRange = match &rendered {
            Some(buf) => buf,
            None => frame,
        };
        let jpeg = clock
            .time(DERIVATIVE_JPEG, || {
                t.encoder.encode(source, r.size, r.quality)
            })
            .map_err(|e| encode_fault(&e))?;
        jpegs.push(Bytes::from(jpeg));
    }
    Ok(jpegs)
}

#[cfg(test)]
#[path = "derivatives_tests.rs"]
mod tests;
