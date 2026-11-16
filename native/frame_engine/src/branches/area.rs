//! The crop side cap (`OP_FRAME_ENGINE_CROP_MAX_SIDE`, `frame_engine/branches/crops.py`): a crop
//! whose long side passes the cap is shrunk before its JPEG encode with OpenCV's area filter
//! (`cv2.resize(..., interpolation=cv2.INTER_AREA)`), never enlarged. A host step on the finish
//! thread, after the crop's pixels reached pinned memory; it is not a resample kernel.
//!
//! A port of OpenCV 5.0.0's `hal::resize` for 8-bit, 3-channel input and a ratio of at least 1 on
//! both sides (`modules/imgproc/src/resize.cpp` at tag 5.0.0, the build opencv-python 5.0.0.93
//! ships; its IPP path declines `INTER_AREA` unless inexact IPP results are allowed, which they
//! are not by default, so OpenCV's own code runs). OpenCV takes one of two paths:
//!
//! * whole ratios (`scale - round(scale)` under `DBL_EPSILON` on both sides, with `scale` the
//!   reciprocal of `dst / src` as OpenCV computes it): each output pixel is the mean of its
//!   `kx x ky` block. At 2 x 2 in integers, `(a + b + c + d + 2) >> 2`; otherwise the integer sum
//!   times `1.0f32 / area`, rounded half to even (`resizeAreaFast_`, `ResizeAreaFastVec`);
//! * other ratios: per axis a table of source taps and `f32` weights (`computeResizeAreaTab`),
//!   each source row spread over a row buffer in `f32`, rows of one output row summed with their
//!   weights, the sum rounded half to even and saturated (`ResizeArea_Invoker`, `inter_area`).
//!
//! The arithmetic order is OpenCV's, in `f32` without fused multiply-adds, as its SIMD and scalar
//! loops both compute.

/// Channels of a crop's pixels (interleaved RGB).
const CN: usize = 3;

/// The size a `w x h` crop is encoded at under a cap of `max_side` (0: no cap), or `None` when
/// the crop stays as it is: the long side scaled to the cap, each side rounded half to even and at
/// least 1 (`CropJpegs._encode`).
pub fn capped_size(w: usize, h: usize, max_side: u32) -> Option<(usize, usize)> {
    let side = w.max(h);
    let cap = usize::try_from(max_side).ok()?;
    if cap == 0 || cap >= side {
        return None;
    }
    let scale = f64::from(max_side) / side as f64;
    let dim = |v: usize| ((v as f64 * scale).round_ties_even() as usize).max(1);
    Some((dim(w), dim(h)))
}

/// `cvRound`: to the nearest integer, ties to even (`lrint` in the default rounding mode).
fn cv_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// `saturate_cast<uchar>(float)`: rounded half to even, then cut to `0..=255`.
fn saturate_u8(v: f32) -> u8 {
    v.round_ties_even().clamp(0.0, 255.0) as u8
}

/// One axis's ratio as OpenCV derives it: `1 / (dst / src)`, its nearest integer and whether the
/// two are within `DBL_EPSILON`.
fn ratio(src: usize, dst: usize) -> (f64, i64, bool) {
    let inv = dst as f64 / src as f64;
    let scale = 1.0 / inv;
    let whole = cv_round(scale);
    (scale, whole, (scale - whole as f64).abs() < f64::EPSILON)
}

/// The `src_w x src_h` interleaved RGB pixels resized to `dst_w x dst_h` with OpenCV's area
/// filter. Only shrinking is ported (both ratios at least 1), which is all the cap asks.
///
/// # Errors
///
/// A byte count that does not match the source size, an empty or larger destination, or a
/// whole ratio whose blocks do not tile the source.
pub fn resize_area(
    src: &[u8],
    (src_w, src_h): (usize, usize),
    (dst_w, dst_h): (usize, usize),
) -> Result<Vec<u8>, String> {
    if src.len() != src_w * src_h * CN || dst_w == 0 || dst_h == 0 {
        return Err(format!(
            "area resize of {} bytes as {src_w} x {src_h} to {dst_w} x {dst_h}",
            src.len()
        ));
    }
    if dst_w > src_w || dst_h > src_h {
        return Err(format!(
            "area resize enlarges {src_w} x {src_h} to {dst_w} x {dst_h}"
        ));
    }
    let (scale_x, kx, whole_x) = ratio(src_w, dst_w);
    let (scale_y, ky, whole_y) = ratio(src_h, dst_h);
    if whole_x && whole_y {
        let kx = usize::try_from(kx).map_err(|e| e.to_string())?;
        let ky = usize::try_from(ky).map_err(|e| e.to_string())?;
        // a whole ratio of two sizes is exact: every block lies inside the source
        if kx * dst_w != src_w || ky * dst_h != src_h {
            return Err(format!(
                "area resize: {src_w} x {src_h} is not {kx} x {ky} blocks of {dst_w} x {dst_h}"
            ));
        }
        return Ok(blocks(src, src_w, (dst_w, dst_h), (kx, ky)));
    }
    let xtab = area_tab(src_w, dst_w, CN, scale_x);
    let ytab = area_tab(src_h, dst_h, 1, scale_y);
    Ok(weighted(src, src_w, (dst_w, dst_h), &xtab, &ytab))
}

/// `resizeAreaFast_` on whole blocks: the mean of each `kx x ky` block.
fn blocks(
    src: &[u8],
    src_w: usize,
    (dst_w, dst_h): (usize, usize),
    (kx, ky): (usize, usize),
) -> Vec<u8> {
    let pitch = src_w * CN;
    let area = kx * ky;
    let scale = 1.0f32 / area as f32;
    let mut out = Vec::with_capacity(dst_w * dst_h * CN);
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            for c in 0..CN {
                let first = dy * ky * pitch + dx * kx * CN + c;
                let sum: u32 = (0..ky)
                    .flat_map(|sy| (0..kx).map(move |sx| first + sy * pitch + sx * CN))
                    .map(|i| u32::from(src[i]))
                    .sum();
                out.push(if kx == 2 && ky == 2 {
                    ((sum + 2) >> 2) as u8
                } else {
                    saturate_u8(sum as f32 * scale)
                });
            }
        }
    }
    out
}

/// One tap of an area table (`DecimateAlpha`): output offset, source offset, weight.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Tap {
    di: usize,
    si: usize,
    alpha: f32,
}

/// `computeResizeAreaTab`: per output index the source cells it covers, the partial cells at
/// either end weighted by the part covered, every weight divided by the cell's width.
fn area_tab(ssize: usize, dsize: usize, cn: usize, scale: f64) -> Vec<Tap> {
    let mut tab = Vec::with_capacity(ssize * 2);
    let last = ssize as i64 - 1;
    for dx in 0..dsize {
        let fsx1 = dx as f64 * scale;
        let fsx2 = fsx1 + scale;
        let cell = scale.min(ssize as f64 - fsx1);
        let mut sx1 = fsx1.ceil() as i64;
        let sx2 = (fsx2.floor() as i64).min(last);
        sx1 = sx1.min(sx2);
        let at = |sx: i64, alpha: f32| Tap {
            di: dx * cn,
            si: usize::try_from(sx).unwrap_or(0) * cn,
            alpha,
        };
        if sx1 as f64 - fsx1 > 1e-3 {
            tab.push(at(sx1 - 1, ((sx1 as f64 - fsx1) / cell) as f32));
        }
        for sx in sx1..sx2 {
            tab.push(at(sx, (1.0 / cell) as f32));
        }
        if fsx2 - sx2 as f64 > 1e-3 {
            tab.push(at(
                sx2,
                ((fsx2 - sx2 as f64).min(1.0).min(cell) / cell) as f32,
            ));
        }
    }
    tab
}

/// One source row spread over the row buffer by the x table, channel by channel, in tap order.
fn spread(row: &[u8], xtab: &[Tap], buf: &mut [f32]) {
    buf.fill(0.0);
    for x in xtab {
        for c in 0..CN {
            buf[x.di + c] += f32::from(row[x.si + c]) * x.alpha;
        }
    }
}

/// `ResizeArea_Invoker` over every output row: each source row of the y table spread over a row
/// buffer with the x table, the buffers of one output row summed with their y weights.
fn weighted(
    src: &[u8],
    src_w: usize,
    (dst_w, dst_h): (usize, usize),
    xtab: &[Tap],
    ytab: &[Tap],
) -> Vec<u8> {
    let width = dst_w * CN;
    let pitch = src_w * CN;
    let mut out = vec![0u8; width * dst_h];
    let mut buf = vec![0f32; width];
    let mut sum = vec![0f32; width];
    let mut prev_dy = ytab.first().map_or(0, |t| t.di);
    let store = |out: &mut [u8], dy: usize, sum: &[f32]| {
        for (d, s) in out[dy * width..(dy + 1) * width].iter_mut().zip(sum) {
            *d = saturate_u8(*s);
        }
    };
    for y in ytab {
        spread(&src[y.si * pitch..(y.si + 1) * pitch], xtab, &mut buf);
        if y.di == prev_dy {
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s += y.alpha * b;
            }
        } else {
            store(&mut out, prev_dy, &sum);
            for (s, b) in sum.iter_mut().zip(&buf) {
                *s = y.alpha * b;
            }
            prev_dy = y.di;
        }
    }
    store(&mut out, prev_dy, &sum);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_scales_the_long_side_rounds_half_to_even_and_never_enlarges() {
        assert_eq!(capped_size(2048, 1001, 1024), Some((1024, 500)));
        assert_eq!(capped_size(5000, 1, 640), Some((640, 1)));
        assert_eq!(capped_size(500, 500, 500), None);
        assert_eq!(capped_size(500, 400, 0), None);
        assert_eq!(capped_size(100, 900, 1000), None);
    }

    #[test]
    fn a_fractional_tab_weights_partial_cells_by_the_part_covered() {
        // 5 source cells onto 2: 2.5 each; the middle cell is split half and half
        let tab = area_tab(5, 2, 1, 2.5);
        let alphas: Vec<(usize, usize, f32)> = tab.iter().map(|t| (t.di, t.si, t.alpha)).collect();
        assert_eq!(
            alphas,
            vec![
                (0, 0, 0.4),
                (0, 1, 0.4),
                (0, 2, 0.2),
                (1, 2, 0.2),
                (1, 3, 0.4),
                (1, 4, 0.4)
            ]
        );
    }

    #[test]
    fn enlarging_and_a_wrong_byte_count_are_refused() {
        assert!(resize_area(&[0; 12], (2, 2), (3, 2)).is_err());
        assert!(resize_area(&[0; 11], (2, 2), (1, 1)).is_err());
        assert_eq!(
            resize_area(&[1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10], (2, 2), (1, 1)),
            Ok(vec![4, 5, 6])
        );
    }
}
