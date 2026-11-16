//! The engine's kernels and how each is launched: `resample_h`, `resample_v` and `lap_sums` of
//! `kernels/prep.cu` (the Python engine's `prep_kernels.py` reads the same file) and `orient` of
//! `kernels/orient.cu`, compiled by `gpu_frame`'s loader with CuPy's options.
//!
//! Every model input is resampled from the full-resolution frame by one separable two-pass
//! resampler: a horizontal pass into a float32 scratch holding only the rows the output needs,
//! then a vertical pass writing uint8 CHW into its place in a region slice. The launch arithmetic
//! (grids, blocks, the argument order, the scratch rows) is `PrepKernels.resample_into`'s and
//! `lap_sums`'s, written here once as pure functions so it is tested without a card.
//!
//! Weights: `geometry::weights` makes a set on the host; each pool thread keeps the sets it
//! uploaded in its own [`WeightCache`], least recently used first out past
//! [`WEIGHT_CACHE_BYTES`] (the Python engine stages each set in a per-thread 2 MiB arena). A set
//! is uploaded on the thread's stream with a wait before it is used (`FramePool::upload`), queued
//! before the photo's own work so the wait finds the stream idle; a set is read only by kernels
//! of the thread that uploaded it, and an evicted set is freed in order on that stream.
//!
//! Orientation (`engine._orient`, after `ImageOps.exif_transpose`): a pure permutation of the
//! pixels; [`orient_source`] is the kernel's index arithmetic line for line, and the host
//! reference [`orient_host`] is built on it.

use std::collections::HashMap;
use std::sync::Arc;

use gpu_frame::{
    Context, DeviceBuf, DeviceRange, FramePool, Kernel, KernelArg, Kernels, LaunchConfig,
    PinnedBuf, Stream,
};

use crate::fault::GpuFault;
use crate::geometry::{self, GeometryError};

/// CUDA source of the resample and Laplacian-sum kernels, compiled at start.
pub const PREP_SOURCE: &str = include_str!("../kernels/prep.cu");
/// CUDA source of the EXIF re-orientation kernel, compiled at start.
pub const ORIENT_SOURCE: &str = include_str!("../kernels/orient.cu");

/// Threads per block of both resample passes (`prep_kernels.py`).
const RESAMPLE_THREADS: u32 = 128;
/// `lap_sums`: blocks per rectangle and threads per block (`prep_kernels.py`; the kernel's shared
/// arrays hold 256 entries).
const LAP_BLOCKS: u32 = 64;
const LAP_THREADS: u32 = 256;
const ORIENT_THREADS: u32 = 256;
/// Device bytes of weight sets one pool thread keeps (the Python engine's `ARENA_BYTES`).
pub const WEIGHT_CACHE_BYTES: usize = 2 << 20;

/// The filter of a resample (`resample_into`'s `antialias` and `lanczos`): the triangle widened
/// with the downscale (Pillow `BILINEAR`), the plain triangle (OpenCV `INTER_LINEAR`, the OCR
/// detector's input without `det_antialias`), Pillow's `LANCZOS` (the renditions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Filter {
    /// The triangle widened with the downscale (Pillow `BILINEAR`).
    Antialias,
    /// The plain triangle (OpenCV `INTER_LINEAR`).
    Bilinear,
    /// Pillow's `LANCZOS`, `a = 3`, antialiased.
    Lanczos,
}

/// One resample (`PrepKernels.resample_into`): `roi` of the frame scaled to `size`, of which
/// `window` is written into the destination, a CHW uint8 tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resample {
    /// `x0, y0, w, h` of the frame; a rectangle past the frame reads 0 outside it.
    pub roi: [i64; 4],
    /// `w, h` the rectangle is resampled to.
    pub size: [i64; 2],
    /// `left, top, w, h` of that output that is written.
    pub window: [i64; 4],
    /// Where the window's first pixel of the first plane goes, in bytes from the input's start.
    pub dst_offset: usize,
    /// Bytes from one plane to the next (negative writes the planes in reverse order, as the
    /// OCR detector's BGR input), and from one row to the next.
    pub plane: i64,
    /// Bytes from one row to the next.
    pub row_stride: i64,
    /// Which filter the weights are.
    pub filter: Filter,
}

impl Resample {
    /// The horizontal filter's key.
    pub fn x_key(&self) -> WeightKey {
        WeightKey {
            n_in: self.roi[2],
            n_out: self.size[0],
            left: self.window[0],
            take: self.window[2],
            filter: self.filter,
        }
    }

    /// The vertical filter's key.
    pub fn y_key(&self) -> WeightKey {
        WeightKey {
            n_in: self.roi[3],
            n_out: self.size[1],
            left: self.window[1],
            take: self.window[3],
            filter: self.filter,
        }
    }

    /// One past the last destination byte the vertical pass writes, from the input's start;
    /// `None` for an empty window or one that would start before the input.
    pub fn extent(&self) -> Option<usize> {
        let (ww, wh) = (self.window[2], self.window[3]);
        if ww <= 0 || wh <= 0 {
            return None;
        }
        let planes = self.plane.checked_mul(2)?;
        let first = i64::try_from(self.dst_offset)
            .ok()?
            .checked_add(planes.min(0))?;
        if first < 0 {
            return None;
        }
        let last = planes
            .max(0)
            .checked_add((wh - 1).checked_mul(self.row_stride)?)?
            .checked_add(ww)?;
        usize::try_from(last).ok()?.checked_add(self.dst_offset)
    }

    /// The first destination byte the vertical pass writes, from the input's start.
    pub fn first_byte(&self) -> usize {
        let planes = self.plane.saturating_mul(2).min(0);
        self.dst_offset
            .saturating_sub(usize::try_from(-planes).unwrap_or(usize::MAX))
    }
}

/// The filter of one axis: `n_in` source pixels to `n_out`, the `take` outputs from `left`
/// (`prep_kernels._weights`; the antialiased triangle turns into plain bilinear when the output
/// is larger than the input).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightKey {
    /// Source pixels along the axis (the rectangle's width or height).
    pub n_in: i64,
    /// Output pixels along the axis after scaling.
    pub n_out: i64,
    /// First output index written.
    pub left: i64,
    /// Number of output indexes written from `left`.
    pub take: i64,
    /// Which filter the weights are.
    pub filter: Filter,
}

impl WeightKey {
    /// The packed filter for this key.
    ///
    /// # Errors
    ///
    /// The window `left .. left + take` holds no output index, or `n_out` is not positive.
    pub fn weights(&self) -> Result<geometry::Weights, GeometryError> {
        let (n_in, n_out, left, take) = (self.n_in, self.n_out, self.left, self.take);
        match self.filter {
            Filter::Antialias => geometry::weights(n_in, n_out, left, take, true),
            Filter::Bilinear => geometry::weights(n_in, n_out, left, take, false),
            Filter::Lanczos => geometry::lanczos_weights(n_in, n_out, left, take),
        }
    }
}

/// A weight set where the kernels read it: `first int32[n] | count int32[n] | weights
/// float32[n * kmax]` from the buffer's start, and the source rows `lo..hi` it reads.
#[derive(Debug)]
pub struct WeightSet<B> {
    /// Output indexes in the set.
    pub n: usize,
    /// Weights per output index.
    pub kmax: usize,
    /// First source row (or column) any output of the set reads.
    pub lo: i64,
    /// One past the last source row (or column) any output of the set reads.
    pub hi: i64,
    /// Bytes the set occupies in `buf`, counted against the cache's capacity.
    pub bytes: usize,
    /// The buffer holding the packed set; freed when the last `Arc` of the set is dropped.
    pub buf: B,
}

struct Slot<B> {
    set: Arc<WeightSet<B>>,
    used: u64,
}

/// One thread's weight sets, least recently used out past `cap` bytes. A set in use by queued
/// work stays alive through its `Arc` after it leaves the cache.
pub struct WeightCache<B> {
    cap: usize,
    held: usize,
    tick: u64,
    slots: HashMap<WeightKey, Slot<B>>,
}

impl<B> WeightCache<B> {
    /// An empty cache that keeps at most `cap` bytes of weight sets.
    pub fn new(cap: usize) -> Self {
        Self {
            cap,
            held: 0,
            tick: 0,
            slots: HashMap::new(),
        }
    }

    /// The set of `key`, made and stored with `upload` (the packed blob in, its buffer out) when
    /// it is not cached.
    ///
    /// # Errors
    ///
    /// The weights cannot be made for the key or `upload` fails.
    pub fn get<E: From<GeometryError>>(
        &mut self,
        key: WeightKey,
        upload: impl FnOnce(&[u8]) -> Result<B, E>,
    ) -> Result<Arc<WeightSet<B>>, E> {
        self.tick += 1;
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.used = self.tick;
            return Ok(Arc::clone(&slot.set));
        }
        let w = key.weights()?;
        let set = Arc::new(WeightSet {
            n: w.n,
            kmax: w.kmax,
            lo: w.lo,
            hi: w.hi,
            bytes: w.blob.len(),
            buf: upload(&w.blob)?,
        });
        if set.bytes <= self.cap {
            while self.held + set.bytes > self.cap {
                let Some(oldest) = self
                    .slots
                    .iter()
                    .min_by_key(|(_, s)| s.used)
                    .map(|(k, _)| *k)
                else {
                    break;
                };
                if let Some(gone) = self.slots.remove(&oldest) {
                    self.held -= gone.set.bytes;
                }
            }
            self.held += set.bytes;
            self.slots.insert(
                key,
                Slot {
                    set: Arc::clone(&set),
                    used: self.tick,
                },
            );
        }
        Ok(set)
    }

    /// Bytes of the sets the cache holds.
    pub fn held(&self) -> usize {
        self.held
    }

    /// Number of cached sets.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether no set is cached.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

fn int(v: i64, what: &str) -> Result<i32, GpuFault> {
    i32::try_from(v)
        .map_err(|_| GpuFault::Failed(format!("{what} {v} is out of the kernel's range")))
}

fn grid(v: i64, what: &str) -> Result<u32, GpuFault> {
    u32::try_from(v)
        .ok()
        .filter(|n| *n > 0)
        .ok_or_else(|| GpuFault::Failed(format!("{what} {v} is no launch dimension")))
}

/// The scratch rows and the two launches of one resample, from its vertical filter's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResampleShape {
    /// Scratch rows the horizontal pass writes and the vertical pass reads.
    pub rows: i64,
    /// Device bytes of that scratch (`rows` times the window width times 3 floats).
    pub scratch_bytes: usize,
    /// Launch of the horizontal pass.
    pub h: LaunchConfig,
    /// Launch of the vertical pass.
    pub v: LaunchConfig,
}

/// `resample_into`'s launches: `rows = max(1, hi - lo)` scratch rows of `window w x 3` floats;
/// grids of `ceil(w / 128)` by the scratch rows, then by the window's height; 128 threads.
///
/// # Errors
///
/// The window has no positive size or the scratch size overflows.
pub fn resample_shape(r: &Resample, y_lo: i64, y_hi: i64) -> Result<ResampleShape, GpuFault> {
    let (ww, wh) = (r.window[2], r.window[3]);
    let rows = (y_hi - y_lo).max(1);
    let gx = grid(
        (ww + i64::from(RESAMPLE_THREADS) - 1) / i64::from(RESAMPLE_THREADS),
        "grid",
    )?;
    let block = (RESAMPLE_THREADS, 1, 1);
    let scratch = rows
        .checked_mul(ww)
        .and_then(|n| n.checked_mul(12))
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| GpuFault::Failed("resample scratch size overflows".into()))?;
    Ok(ResampleShape {
        rows,
        scratch_bytes: scratch,
        h: LaunchConfig {
            grid: (gx, grid(rows, "scratch rows")?, 1),
            block,
            shared_bytes: 0,
        },
        v: LaunchConfig {
            grid: (gx, grid(wh, "window height")?, 1),
            block,
            shared_bytes: 0,
        },
    })
}

/// Where one weight set is: its device address and its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetAt {
    /// Device address of the packed set.
    pub addr: u64,
    /// Output indexes in the set.
    pub n: usize,
    /// Weights per output index.
    pub kmax: usize,
    /// First source row (or column) the set reads.
    pub lo: i64,
}

impl<B: DeviceRange> From<&WeightSet<B>> for SetAt {
    fn from(s: &WeightSet<B>) -> Self {
        Self {
            addr: s.buf.device_addr(),
            n: s.n,
            kmax: s.kmax,
            lo: s.lo,
        }
    }
}

/// `resample_h`'s arguments: `src, H, W, x0, y0 + lo, rows, xmin, xcnt, wx, kmax, out_w, tmp`.
///
/// # Errors
///
/// A value does not fit the kernel's 32-bit argument.
pub fn resample_h_args(
    frame: (u64, u32, u32),
    r: &Resample,
    wx: SetAt,
    y_lo: i64,
    rows: i64,
    tmp: u64,
) -> Result<Vec<KernelArg>, GpuFault> {
    let (src, w, h) = frame;
    let n = wx.n as u64;
    Ok(vec![
        KernelArg::U64(src),
        KernelArg::I32(int(i64::from(h), "frame height")?),
        KernelArg::I32(int(i64::from(w), "frame width")?),
        KernelArg::I32(int(r.roi[0], "x0")?),
        KernelArg::I32(int(r.roi[1] + y_lo, "y0")?),
        KernelArg::I32(int(rows, "rows")?),
        KernelArg::U64(wx.addr),
        KernelArg::U64(wx.addr + 4 * n),
        KernelArg::U64(wx.addr + 8 * n),
        KernelArg::I32(int(wx.kmax as i64, "kmax")?),
        KernelArg::I32(int(r.window[2], "window width")?),
        KernelArg::U64(tmp),
    ])
}

/// `resample_v`'s arguments: `tmp, out_w, ymin, ycnt, wy, kmax, out_h, ybase, dst, plane,
/// row_stride` (the scratch starts at source row `lo`).
///
/// # Errors
///
/// A value does not fit the kernel's 32-bit argument.
pub fn resample_v_args(
    tmp: u64,
    r: &Resample,
    wy: SetAt,
    dst: u64,
) -> Result<Vec<KernelArg>, GpuFault> {
    let n = wy.n as u64;
    Ok(vec![
        KernelArg::U64(tmp),
        KernelArg::I32(int(r.window[2], "window width")?),
        KernelArg::U64(wy.addr),
        KernelArg::U64(wy.addr + 4 * n),
        KernelArg::U64(wy.addr + 8 * n),
        KernelArg::I32(int(wy.kmax as i64, "kmax")?),
        KernelArg::I32(int(r.window[3], "window height")?),
        KernelArg::I32(int(wy.lo, "ybase")?),
        KernelArg::U64(dst),
        KernelArg::I64(r.plane),
        KernelArg::I32(int(r.row_stride, "row stride")?),
    ])
}

/// `lap_sums` over `n` rectangles: 64 blocks of 256 threads per rectangle.
///
/// # Errors
///
/// `n` is zero or too large for a launch dimension.
pub fn lap_launch(n: usize) -> Result<LaunchConfig, GpuFault> {
    Ok(LaunchConfig {
        grid: (LAP_BLOCKS, grid(n as i64, "rectangles")?, 1),
        block: (LAP_THREADS, 1, 1),
        shared_bytes: 0,
    })
}

/// The rectangles as the kernel reads them: `int32 [n, 4]`, little endian.
///
/// # Errors
///
/// A coordinate does not fit 32 bits.
pub fn rect_blob(rects: &[[i64; 4]]) -> Result<Vec<u8>, GpuFault> {
    let mut out = Vec::with_capacity(rects.len() * 16);
    for v in rects.iter().flatten() {
        out.extend_from_slice(&int(*v, "rectangle")?.to_le_bytes());
    }
    Ok(out)
}

/// `int64 [n, 2]` (sum, sum of squares) as the kernel wrote them.
pub fn lap_values(bytes: &[u8]) -> Vec<[i64; 2]> {
    let (pairs, _) = bytes.as_chunks::<16>();
    pairs
        .iter()
        .map(|c| {
            let (sum, squares) = c.split_at(8);
            let v = |half: &[u8]| {
                let mut b = [0u8; 8];
                b.copy_from_slice(half);
                i64::from_le_bytes(b)
            };
            [v(sum), v(squares)]
        })
        .collect()
}

/// Whether the EXIF tag moves pixels (`engine._orient` leaves 1 and unknown tags alone).
pub fn reorients(orientation: u32) -> bool {
    (2..=8).contains(&orientation)
}

/// `(width, height)` of a `w x h` frame after the tag (`prep_geometry.oriented_size`).
pub fn oriented_dims(w: u32, h: u32, orientation: u32) -> (u32, u32) {
    if (5..=8).contains(&orientation) {
        (h, w)
    } else {
        (w, h)
    }
}

/// The source pixel of output pixel `(x, y)`: the kernel's arithmetic. `engine._orient` applies a
/// transpose (tags 5 to 8), then a horizontal flip (2, 3, 6, 7), then a vertical flip (3, 4, 7,
/// 8); this undoes them in the opposite order.
pub fn orient_source(x: u32, y: u32, orientation: u32, out_w: u32, out_h: u32) -> (u32, u32) {
    let (mut xx, mut yy) = (x, y);
    if matches!(orientation, 3 | 4 | 7 | 8) {
        yy = out_h - 1 - yy;
    }
    if matches!(orientation, 2 | 3 | 6 | 7) {
        xx = out_w - 1 - xx;
    }
    if (5..=8).contains(&orientation) {
        (yy, xx)
    } else {
        (xx, yy)
    }
}

/// The oriented copy of an interleaved RGB frame on the host (the device's reference).
pub fn orient_host(src: &[u8], w: u32, h: u32, orientation: u32) -> (Vec<u8>, u32, u32) {
    let (ow, oh) = oriented_dims(w, h, orientation);
    let mut out = vec![0u8; src.len()];
    for y in 0..oh {
        for x in 0..ow {
            let (sx, sy) = orient_source(x, y, orientation, ow, oh);
            let s = (sy as usize * w as usize + sx as usize) * 3;
            let d = (y as usize * ow as usize + x as usize) * 3;
            out[d..d + 3].copy_from_slice(&src[s..s + 3]);
        }
    }
    (out, ow, oh)
}

/// `orient`'s launch: 256 threads per block across an output row, a block row per output row.
///
/// # Errors
///
/// The output width or height is zero or too large for a launch dimension.
pub fn orient_launch(out_w: u32, out_h: u32) -> Result<LaunchConfig, GpuFault> {
    Ok(LaunchConfig {
        grid: (
            grid(i64::from(out_w.div_ceil(ORIENT_THREADS)), "grid")?,
            grid(i64::from(out_h), "rows")?,
            1,
        ),
        block: (ORIENT_THREADS, 1, 1),
        shared_bytes: 0,
    })
}

/// A pinned host buffer that grows to the largest copy asked of it (crop pixels, sums).
pub struct Staging {
    ctx: Arc<Context>,
    buf: Option<PinnedBuf>,
}

impl Staging {
    /// An empty buffer on `ctx`'s card; the pinned memory is allocated on first use.
    pub fn new(ctx: &Arc<Context>) -> Self {
        Self {
            ctx: Arc::clone(ctx),
            buf: None,
        }
    }

    /// At least `len` bytes (reallocated only to grow).
    ///
    /// # Errors
    ///
    /// The pinned allocation fails.
    pub fn at_least(&mut self, len: usize) -> Result<&mut PinnedBuf, GpuFault> {
        if self.buf.as_ref().is_none_or(|b| b.len() < len) {
            self.buf = None;
            self.buf = Some(PinnedBuf::new(&self.ctx, len)?);
        }
        self.buf
            .as_mut()
            .ok_or_else(|| GpuFault::Failed("no staging buffer".into()))
    }
}

/// A frame on the card as the kernels read it: interleaved RGB, `width x 3` bytes per row.
pub trait FrameRef: DeviceRange {
    /// Width in pixels.
    fn width(&self) -> u32;
    /// Height in pixels.
    fn height(&self) -> u32;
}

/// The loaded kernels; one instance serves every pool thread.
pub struct PrepKernels {
    resample_h: Kernel,
    resample_v: Kernel,
    lap_sums: Kernel,
    orient: Kernel,
}

impl PrepKernels {
    /// Compiles the kernel sources and looks up the entry points.
    ///
    /// # Errors
    ///
    /// A source does not compile or an entry point is missing.
    pub fn load(kernels: &Kernels) -> Result<Self, GpuFault> {
        let prep = kernels.load("prep.cu", PREP_SOURCE)?;
        let orient = kernels.load("orient.cu", ORIENT_SOURCE)?;
        Ok(Self {
            resample_h: prep.function("resample_h")?,
            resample_v: prep.function("resample_v")?,
            lap_sums: prep.function("lap_sums")?,
            orient: orient.function("orient")?,
        })
    }

    /// The two weight sets of `r` from this thread's cache, uploaded on `stream` when missing.
    ///
    /// # Errors
    ///
    /// A weight set cannot be made or uploaded.
    pub fn weights(
        &self,
        cache: &mut WeightCache<DeviceBuf>,
        pool: &FramePool,
        stream: &Arc<Stream>,
        r: &Resample,
    ) -> Result<[Arc<WeightSet<DeviceBuf>>; 2], GpuFault> {
        let upload = |blob: &[u8]| pool.upload(blob, stream).map_err(GpuFault::from);
        Ok([cache.get(r.x_key(), upload)?, cache.get(r.y_key(), upload)?])
    }

    /// Queue `r` on `stream`: frame to `dst` (a region slice), through a scratch freed in order
    /// on the same stream. The destination window is checked to lie inside `dst`.
    ///
    /// # Errors
    ///
    /// The window is empty, `dst` does not hold it, or an allocation or launch fails.
    pub fn resample(
        &self,
        pool: &FramePool,
        stream: &Arc<Stream>,
        frame: &dyn FrameRef,
        dst: &dyn DeviceRange,
        r: &Resample,
        sets: &[Arc<WeightSet<DeviceBuf>>; 2],
    ) -> Result<(), GpuFault> {
        let extent = r
            .extent()
            .ok_or_else(|| GpuFault::Failed(format!("an empty resample window {r:?}")))?;
        let lo = r.first_byte();
        let dst_addr = dst.checked_addr(lo, extent - lo)? + (r.dst_offset - lo) as u64;
        let (wx, wy) = (SetAt::from(&*sets[0]), SetAt::from(&*sets[1]));
        let shape = resample_shape(r, sets[1].lo, sets[1].hi)?;
        let scratch = pool.alloc(shape.scratch_bytes, stream)?;
        let src = (frame.device_addr(), frame.width(), frame.height());
        let h_args = resample_h_args(src, r, wx, wy.lo, shape.rows, scratch.device_addr())?;
        self.resample_h.launch(&shape.h, stream, &h_args)?;
        let v_args = resample_v_args(scratch.device_addr(), r, wy, dst_addr)?;
        self.resample_v.launch(&shape.v, stream, &v_args)?;
        Ok(())
    }

    /// The Laplacian sums of each rectangle (`x1, y1, x2, y2`, clamped and not empty), waited for.
    ///
    /// # Errors
    ///
    /// A rectangle does not fit 32 bits, or an allocation, copy or launch fails.
    pub fn lap_sums(
        &self,
        pool: &FramePool,
        stream: &Arc<Stream>,
        staging: &mut Staging,
        frame: &dyn FrameRef,
        rects: &[[i64; 4]],
    ) -> Result<Vec<[i64; 2]>, GpuFault> {
        if rects.is_empty() {
            return Ok(Vec::new());
        }
        let blob = rect_blob(rects)?;
        let on_card = pool.alloc(blob.len(), stream)?;
        stream.copy_from_host(&on_card, 0, &blob)?;
        let out = pool.alloc(rects.len() * 16, stream)?;
        stream.memset(&out, 0, out.len(), 0)?;
        let args = [
            KernelArg::U64(frame.device_addr()),
            KernelArg::I32(int(i64::from(frame.height()), "frame height")?),
            KernelArg::I32(int(i64::from(frame.width()), "frame width")?),
            KernelArg::U64(on_card.device_addr()),
            KernelArg::I32(int(rects.len() as i64, "rectangles")?),
            KernelArg::U64(out.device_addr()),
        ];
        self.lap_sums
            .launch(&lap_launch(rects.len())?, stream, &args)?;
        let pinned = staging.at_least(out.len())?;
        let bytes = stream.copy_to_pinned(&out, 0, pinned, out.len())?.wait()?;
        Ok(lap_values(bytes))
    }

    /// The frame `src` (`w x h`) re-oriented by `orientation` into a new buffer, queued on
    /// `stream`; `None` when the tag moves no pixel.
    ///
    /// # Errors
    ///
    /// The buffer does not hold the frame, a size does not fit 32 bits, or an allocation or launch fails.
    pub fn orient(
        &self,
        pool: &FramePool,
        stream: &Arc<Stream>,
        src: &dyn DeviceRange,
        (w, h): (u32, u32),
        orientation: u32,
    ) -> Result<Option<(DeviceBuf, u32, u32)>, GpuFault> {
        if !reorients(orientation) {
            return Ok(None);
        }
        let (ow, oh) = oriented_dims(w, h, orientation);
        let out = pool.alloc(src.byte_len(), stream)?;
        let args = [
            KernelArg::U64(src.checked_addr(0, w as usize * h as usize * 3)?),
            KernelArg::I32(int(i64::from(h), "frame height")?),
            KernelArg::I32(int(i64::from(w), "frame width")?),
            KernelArg::I32(int(i64::from(orientation), "orientation")?),
            KernelArg::U64(out.checked_addr(0, ow as usize * oh as usize * 3)?),
            KernelArg::I32(int(i64::from(oh), "frame height")?),
            KernelArg::I32(int(i64::from(ow), "frame width")?),
        ];
        self.orient.launch(&orient_launch(ow, oh)?, stream, &args)?;
        Ok(Some((out, ow, oh)))
    }
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;
