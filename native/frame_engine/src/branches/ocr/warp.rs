//! The text-line warp (`kernels/ocr_warp.cu`, the file `frame_engine/ocr/kernels.py` reads): each
//! line of one recogniser call sampled from the full-resolution frame through its perspective map
//! straight into its `[3, 48, width]` slot, padded right of its resized width, one launch per call.
//! The launch arithmetic and the argument order are `LineWarp.warp_into`'s, written here as pure
//! functions.

use std::sync::Arc;

use gpu_frame::{DeviceRange, FramePool, Kernel, KernelArg, Kernels, LaunchConfig, Stream};

use super::LineWarp;
use super::lines::REC_HEIGHT;
use crate::fault::GpuFault;
use crate::kernels::FrameRef;

/// The kernel's CUDA source, the file the Python engine compiles.
pub const WARP_SOURCE: &str = include_str!("../../../kernels/ocr_warp.cu");
/// Threads per block (`warp_into`).
const THREADS: i64 = 128;

/// Bytes of the call's input the warp writes: `lines x 3 x 48 x batch_width`.
pub fn bytes(w: &LineWarp) -> usize {
    let n = w.lines() as i64;
    usize::try_from(n * 3 * REC_HEIGHT * w.batch_width).unwrap_or(0)
}

/// `((batch_width * 48 + 127) / 128, lines)` blocks of 128 threads.
///
/// # Errors
///
/// No line, no width, or a grid past the launch limits.
pub fn launch(lines: usize, batch_width: i64) -> Result<LaunchConfig, GpuFault> {
    let gx = batch_width
        .checked_mul(REC_HEIGHT)
        .map(|p| (p + THREADS - 1) / THREADS)
        .and_then(|g| u32::try_from(g).ok())
        .filter(|g| *g > 0);
    let gy = u32::try_from(lines).ok().filter(|g| *g > 0);
    match (gx, gy) {
        (Some(gx), Some(gy)) => Ok(LaunchConfig {
            grid: (gx, gy, 1),
            block: (THREADS as u32, 1, 1),
            shared_bytes: 0,
        }),
        _ => Err(GpuFault::Failed(format!(
            "no text warp launch for {lines} lines of width {batch_width}"
        ))),
    }
}

/// The checks of `warp_into`: one map row of 9, one width and two taps per line, no line wider
/// than its call.
///
/// # Errors
///
/// The arrays do not match the lines, or a line is wider than its call.
pub fn check(w: &LineWarp) -> Result<(), GpuFault> {
    let n = w.lines();
    if w.maps.len() != 9 * n || w.taps.len() != 2 * n {
        return Err(GpuFault::Failed(format!(
            "{} map values and {} taps for {n} text lines",
            w.maps.len(),
            w.taps.len()
        )));
    }
    if w.widths.iter().any(|v| i64::from(*v) > w.batch_width) {
        return Err(GpuFault::Failed(
            "a text line is wider than its batch".into(),
        ));
    }
    Ok(())
}

/// The maps (`float32 [lines, 9]`), widths (`int32 [lines]`) and taps (`int32 [lines, 2]`) as the
/// kernel reads them.
pub fn blobs(w: &LineWarp) -> [Vec<u8>; 3] {
    [
        w.maps.iter().flat_map(|v| v.to_le_bytes()).collect(),
        w.widths.iter().flat_map(|v| v.to_le_bytes()).collect(),
        w.taps.iter().flat_map(|v| v.to_le_bytes()).collect(),
    ]
}

fn int(v: i64, what: &str) -> Result<i32, GpuFault> {
    i32::try_from(v)
        .map_err(|_| GpuFault::Failed(format!("{what} {v} is out of the kernel's range")))
}

/// `warp_lines`'s arguments: `src, H, W, maps, widths, taps, n_lines, batch_w, height, pad, dst`.
///
/// # Errors
///
/// A dimension past the kernel's `int` arguments.
pub fn args(
    frame: (u64, u32, u32),
    tables: [u64; 3],
    w: &LineWarp,
    dst: u64,
) -> Result<Vec<KernelArg>, GpuFault> {
    let (src, fw, fh) = frame;
    let [maps, widths, taps] = tables;
    Ok(vec![
        KernelArg::U64(src),
        KernelArg::I32(int(i64::from(fh), "frame height")?),
        KernelArg::I32(int(i64::from(fw), "frame width")?),
        KernelArg::U64(maps),
        KernelArg::U64(widths),
        KernelArg::U64(taps),
        KernelArg::I32(int(w.lines() as i64, "lines")?),
        KernelArg::I32(int(w.batch_width, "batch width")?),
        KernelArg::I32(int(REC_HEIGHT, "line height")?),
        KernelArg::U8(w.pad),
        KernelArg::U64(dst),
    ])
}

/// The loaded warp kernel; one instance serves every launch thread.
pub struct TextWarpKernel {
    warp: Kernel,
}

impl TextWarpKernel {
    /// Compile the kernel for the current device.
    ///
    /// # Errors
    ///
    /// NVRTC or the module load fails.
    pub fn load(kernels: &Kernels) -> Result<Self, GpuFault> {
        Ok(Self {
            warp: kernels
                .load("ocr_warp.cu", WARP_SOURCE)?
                .function("warp_lines")?,
        })
    }

    /// Queue the call's warp on `stream` into `dst` (a region slice of at least its bytes); the
    /// tables go up on the same stream and are freed in order behind the launch.
    ///
    /// # Errors
    ///
    /// The tables do not match the lines, `dst` is too small, an allocation or copy fails, or
    /// the launch fails.
    pub fn launch(
        &self,
        pool: &FramePool,
        stream: &Arc<Stream>,
        frame: &dyn FrameRef,
        dst: &dyn DeviceRange,
        w: &LineWarp,
    ) -> Result<(), GpuFault> {
        if w.lines() == 0 {
            return Ok(());
        }
        check(w)?;
        let dst_addr = dst.checked_addr(0, bytes(w))?;
        let up = |blob: &[u8]| -> Result<gpu_frame::DeviceBuf, GpuFault> {
            let d = pool.alloc(blob.len(), stream)?;
            stream.copy_from_host(&d, 0, blob)?;
            Ok(d)
        };
        let [maps, widths, taps] = blobs(w);
        let (maps, widths, taps) = (up(&maps)?, up(&widths)?, up(&taps)?);
        let tables = [maps.device_addr(), widths.device_addr(), taps.device_addr()];
        let src = (frame.device_addr(), frame.width(), frame.height());
        let a = args(src, tables, w, dst_addr)?;
        self.warp
            .launch(&launch(w.lines(), w.batch_width)?, stream, &a)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "warp_tests.rs"]
mod tests;
