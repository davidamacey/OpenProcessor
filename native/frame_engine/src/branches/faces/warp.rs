//! The face warp (`kernels/faces_warp.cu`, the file `frame_engine/faces/kernels.py` reads):
//! aligned 112 x 112 crops sampled from the full-resolution frame straight into the staged
//! recogniser batch, one launch per batch.
//!
//! Each thread is one output pixel of one face: `taps x taps` points spread over the pixel's
//! footprint through the face's inverse similarity, each read bilinearly (outside the frame is
//! 0), averaged, rounded, written as uint8 CHW in the frame's RGB order. The launch arithmetic and
//! the argument order are `WarpKernel.warp_into`'s, written here as pure functions.

use std::sync::Arc;

use gpu_frame::{DeviceRange, FramePool, Kernel, KernelArg, Kernels, LaunchConfig, Stream};

use crate::fault::GpuFault;
use crate::kernels::FrameRef;

/// The kernel's CUDA source, the file the Python engine compiles.
pub const WARP_SOURCE: &str = include_str!("../../../kernels/faces_warp.cu");
/// Threads per block (`warp_into`).
const THREADS: u32 = 128;

/// One recogniser batch's warp: per face its output-to-source map and taps, the output side.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceWarps {
    /// Per face, the output-to-source map.
    pub inv: Vec<[f32; 6]>,
    /// Per face, the taps per axis.
    pub taps: Vec<i32>,
    /// The output side.
    pub size: u32,
}

impl FaceWarps {
    /// Bytes of the batch the warp writes: `faces x 3 x size x size`.
    pub fn bytes(&self) -> usize {
        let side = self.size as usize;
        self.inv.len() * 3 * side * side
    }
}

/// `((size * size + 127) / 128, faces)` blocks of 128 threads.
///
/// # Errors
///
/// No face, or a grid past the launch limits.
pub fn launch(faces: usize, size: u32) -> Result<LaunchConfig, GpuFault> {
    let pixels = u64::from(size) * u64::from(size);
    let gx = u32::try_from(pixels.div_ceil(u64::from(THREADS)))
        .ok()
        .filter(|g| *g > 0);
    let gy = u32::try_from(faces).ok().filter(|g| *g > 0);
    match (gx, gy) {
        (Some(gx), Some(gy)) => Ok(LaunchConfig {
            grid: (gx, gy, 1),
            block: (THREADS, 1, 1),
            shared_bytes: 0,
        }),
        _ => Err(GpuFault::Failed(format!(
            "no warp launch for {faces} faces of side {size}"
        ))),
    }
}

/// The maps (`float32 [faces, 6]`) and taps (`int32 [faces]`) as the kernel reads them.
pub fn blobs(w: &FaceWarps) -> (Vec<u8>, Vec<u8>) {
    let inv = w
        .inv
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let taps = w.taps.iter().flat_map(|v| v.to_le_bytes()).collect();
    (inv, taps)
}

fn int(v: u64, what: &str) -> Result<i32, GpuFault> {
    i32::try_from(v)
        .map_err(|_| GpuFault::Failed(format!("{what} {v} is out of the kernel's range")))
}

/// `warp_faces`'s arguments: `src, H, W, inv, taps, n_faces, S, dst, slot` (`slot` the bytes of
/// one face's crop).
///
/// # Errors
///
/// A dimension past the kernel's `int` arguments.
pub fn args(
    frame: (u64, u32, u32),
    inv: u64,
    taps: u64,
    faces: usize,
    size: u32,
    dst: u64,
) -> Result<Vec<KernelArg>, GpuFault> {
    let (src, w, h) = frame;
    let slot = 3 * i64::from(size) * i64::from(size);
    Ok(vec![
        KernelArg::U64(src),
        KernelArg::I32(int(u64::from(h), "frame height")?),
        KernelArg::I32(int(u64::from(w), "frame width")?),
        KernelArg::U64(inv),
        KernelArg::U64(taps),
        KernelArg::I32(int(faces as u64, "faces")?),
        KernelArg::I32(int(u64::from(size), "output side")?),
        KernelArg::U64(dst),
        KernelArg::I64(slot),
    ])
}

/// The loaded warp kernel; one instance serves every launch thread.
pub struct WarpKernel {
    warp: Kernel,
}

impl WarpKernel {
    /// Compile the kernel for the current device.
    ///
    /// # Errors
    ///
    /// NVRTC or the module load fails.
    pub fn load(kernels: &Kernels) -> Result<Self, GpuFault> {
        Ok(Self {
            warp: kernels
                .load("faces_warp.cu", WARP_SOURCE)?
                .function("warp_faces")?,
        })
    }

    /// Queue the batch's warp on `stream` into `dst` (a region slice of exactly its bytes); the
    /// maps go up on the same stream and are freed in order behind the launch.
    ///
    /// # Errors
    ///
    /// The maps do not match the faces, `dst` is too small, an allocation or copy fails, or
    /// the launch fails.
    pub fn launch(
        &self,
        pool: &FramePool,
        stream: &Arc<Stream>,
        frame: &dyn FrameRef,
        dst: &dyn DeviceRange,
        w: &FaceWarps,
    ) -> Result<(), GpuFault> {
        let n = w.inv.len();
        if n == 0 {
            return Ok(());
        }
        if w.taps.len() != n {
            return Err(GpuFault::Failed(format!(
                "{} taps for {n} faces",
                w.taps.len()
            )));
        }
        let dst_addr = dst.checked_addr(0, w.bytes())?;
        let (inv, taps) = blobs(w);
        let d_inv = pool.alloc(inv.len(), stream)?;
        stream.copy_from_host(&d_inv, 0, &inv)?;
        let d_taps = pool.alloc(taps.len(), stream)?;
        stream.copy_from_host(&d_taps, 0, &taps)?;
        let src = (frame.device_addr(), frame.width(), frame.height());
        let a = args(
            src,
            d_inv.device_addr(),
            d_taps.device_addr(),
            n,
            w.size,
            dst_addr,
        )?;
        self.warp.launch(&launch(n, w.size)?, stream, &a)?;
        Ok(())
    }
}
