//! Pixel geometry and resampling weights, numerically identical to the Python engine's:
//! `src/domain/image_geometry.py` (letterbox, whole square, un-map, crop rectangle) and
//! `frame_engine/prep_geometry.py` (`blur_rect`, `filter_weights`, `stored_box_rects`) with the
//! packing of `frame_engine/prep_kernels.py` (`_weights`). Goldens generated from those
//! functions (`tests/golden/`, `scripts/codegen/frame_engine_goldens.py`) hold every function
//! here to them bit for bit.
//!
//! What makes the numbers equal: Python's `round` is round half to even
//! ([`f64::round_ties_even`]); `//` is floor division (`div_euclid` by a positive number); the
//! weights are float64 arithmetic with numpy's pairwise row sum and a cast to float32 at the end.

use std::fmt;

/// An image of no area, or a window outside the filter: the Python functions raise here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryError {
    /// A side that is not positive.
    DegenerateSize { w: i64, h: i64 },
    /// No output index in the window asked of a filter.
    EmptyWindow,
}

impl fmt::Display for GeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DegenerateSize { w, h } => write!(f, "degenerate image size: ({w}, {h})"),
            Self::EmptyWindow => f.write_str("the window holds no output index"),
        }
    }
}

impl std::error::Error for GeometryError {}

/// The aspect-preserving fit of a frame into a square, pasted at an integer offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub new_w: i64,
    pub new_h: i64,
    pub left: i64,
    pub top: i64,
    pub scale: f64,
    pub pad_w: f64,
    pub pad_h: f64,
}

/// `letterbox_geometry`: the frame of `w` x `h` fitted into a `target` square. A margin of an odd
/// number of pixels puts the extra one on the right or bottom.
pub fn letterbox_geometry(w: i64, h: i64, target: i64) -> Result<Letterbox, GeometryError> {
    if w <= 0 || h <= 0 {
        return Err(GeometryError::DegenerateSize { w, h });
    }
    let t = target as f64;
    let scale = (t / h as f64).min(t / w as f64);
    let new_w = 1.max((w as f64 * scale).round_ties_even() as i64);
    let new_h = 1.max((h as f64 * scale).round_ties_even() as i64);
    let left = (target - new_w).div_euclid(2);
    let top = (target - new_h).div_euclid(2);
    Ok(Letterbox {
        new_w,
        new_h,
        left,
        top,
        scale,
        pad_w: left as f64,
        pad_h: top as f64,
    })
}

/// The whole-frame embedding input: shorter edge to the target, centre crop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Square {
    pub new_w: i64,
    pub new_h: i64,
    pub left: i64,
    pub top: i64,
}

/// `whole_square_geometry`.
pub fn whole_square_geometry(w: i64, h: i64, target: i64) -> Result<Square, GeometryError> {
    if w <= 0 || h <= 0 {
        return Err(GeometryError::DegenerateSize { w, h });
    }
    let scale = target as f64 / h.min(w) as f64;
    let new_w = target.max((w as f64 * scale).round_ties_even() as i64);
    let new_h = target.max((h as f64 * scale).round_ties_even() as i64);
    Ok(Square {
        new_w,
        new_h,
        left: (new_w - target).div_euclid(2),
        top: (new_h - target).div_euclid(2),
    })
}

/// `undo_letterbox`: a box in network pixels back to the frame's pixels.
pub fn undo_letterbox(bbox: [f64; 4], scale: f64, pad: (f64, f64)) -> [f64; 4] {
    let (pad_w, pad_h) = pad;
    let s = scale.max(1e-6);
    let [x1, y1, x2, y2] = bbox;
    [
        (x1 - pad_w) / s,
        (y1 - pad_h) / s,
        (x2 - pad_w) / s,
        (y2 - pad_h) / s,
    ]
}

/// `crop_rect`: the integer rectangle `(x1, y1, x2, y2)` (the last two exclusive) that holds a
/// box: rounded, cut to the frame, at least one pixel wide and high.
pub fn crop_rect(bbox: [f64; 4], width: i64, height: i64) -> [i64; 4] {
    let [x1, y1, x2, y2] = bbox;
    let w = width.max(1);
    let h = height.max(1);
    let x1i = (x1.round_ties_even() as i64).max(0).min(w - 1);
    let y1i = (y1.round_ties_even() as i64).max(0).min(h - 1);
    [
        x1i,
        y1i,
        (x2.round_ties_even() as i64).max(x1i + 1).min(w),
        (y2.round_ties_even() as i64).max(y1i + 1).min(h),
    ]
}

/// `stored_box_rects`: the rectangle of each stored `bbox_norm` row (`x1, y1, x2, y2` in `[0, 1]`
/// of the oriented frame): scaled to pixels in float64, then [`crop_rect`].
pub fn stored_box_rects(boxes_norm: &[[f64; 4]], w: i64, h: i64) -> Vec<[i64; 4]> {
    boxes_norm
        .iter()
        .map(|[x1, y1, x2, y2]| {
            crop_rect(
                [x1 * w as f64, y1 * h as f64, x2 * w as f64, y2 * h as f64],
                w,
                h,
            )
        })
        .collect()
}

/// `blur_rect`: the clamped rectangle the blur sums read, or `None` when it is empty.
pub fn blur_rect(bbox: [f64; 4], w: i64, h: i64) -> Option<[i64; 4]> {
    let [bx1, by1, bx2, by2] = bbox.map(|v| v.round_ties_even() as i64);
    let (x1, x2) = (bx1.min(w).max(0), bx2.min(w).max(0));
    let (y1, y2) = (by1.min(h).max(0), by2.min(h).max(0));
    (x2 > x1 && y2 > y1).then_some([x1, y1, x2, y2])
}

/// numpy's `DOUBLE_pairwise_sum` as `ndarray.sum(axis=-1)` uses it on a contiguous row: eight
/// running sums for 8 to 128 values, halves (a multiple of 8 apart) above that.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut r = 0.0;
        for x in a {
            r += x;
        }
        r
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (acc, x) in r.iter_mut().zip(&a[i..i + 8]) {
                *acc += x;
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for x in &a[i..] {
            res += x;
        }
        res
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_sum(&a[..half]) + pairwise_sum(&a[half..])
    }
}

/// `filter_weights` for the triangle filter: per output index the first source tap, the tap
/// count, and the normalised weights `[out_size, kmax]` (float32, row after row).
#[derive(Debug, Clone, PartialEq)]
pub struct FilterWeights {
    pub first: Vec<i32>,
    pub count: Vec<i32>,
    pub kmax: usize,
    pub weights: Vec<f32>,
}

/// `filter_weights(in_size, out_size, antialias=...)`: a triangle centred on `(i + 0.5) * in / out`
/// whose support widens with the downscale factor when `antialias` (Pillow `BILINEAR`), plain
/// bilinear otherwise (OpenCV `INTER_LINEAR`).
pub fn filter_weights(
    in_size: i64,
    out_size: i64,
    antialias: bool,
) -> Result<FilterWeights, GeometryError> {
    let n = usize::try_from(out_size).map_err(|_| GeometryError::EmptyWindow)?;
    if n == 0 {
        return Err(GeometryError::EmptyWindow);
    }
    let scale = in_size as f64 / out_size as f64;
    let stretch = if antialias { scale.max(1.0) } else { 1.0 };
    let support = stretch;
    let centers: Vec<f64> = (0..n).map(|i| (i as f64 + 0.5) * scale).collect();
    let first: Vec<i64> = centers
        .iter()
        .map(|c| (c - support + 0.5).floor().max(0.0) as i64)
        .collect();
    let count: Vec<i64> = centers
        .iter()
        .zip(&first)
        .map(|(c, f)| ((c + support + 0.5).floor().min(in_size as f64) as i64 - f).max(0))
        .collect();
    let kmax = usize::try_from(count.iter().copied().max().unwrap_or(0).max(1)).unwrap_or(1);
    let mut w = vec![0f64; n * kmax];
    for i in 0..n {
        let row = &mut w[i * kmax..(i + 1) * kmax];
        for (k, slot) in row.iter_mut().enumerate() {
            let xs = (first[i] + k as i64) as f64;
            let u = (xs - centers[i] + 0.5) / stretch;
            *slot = if (k as i64) >= count[i] {
                0.0
            } else {
                (1.0 - u.abs()).max(0.0)
            };
        }
        let total = pairwise_sum(row);
        for v in row.iter_mut() {
            *v = if total != 0.0 { *v / total } else { 0.0 };
        }
    }
    Ok(FilterWeights {
        first: first.iter().map(|v| *v as i32).collect(),
        count: count.iter().map(|v| *v as i32).collect(),
        kmax,
        weights: w.iter().map(|v| *v as f32).collect(),
    })
}

/// One axis's filter packed as the kernels read it (`prep_kernels._weights`):
/// `first int32[n] | count int32[n] | weights float32[n * kmax]`, little endian, for the `take`
/// outputs from `left`, with what the caller sizes its scratch from.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    pub blob: Vec<u8>,
    pub n: usize,
    pub kmax: usize,
    /// The first source index any output of the slice reads.
    pub lo: i64,
    /// One past the last source index any output of the slice reads.
    pub hi: i64,
}

/// `_weights(n_in, n_out, left, take, antialias, lanczos=False)`.
pub fn weights(
    n_in: i64,
    n_out: i64,
    left: i64,
    take: i64,
    antialias: bool,
) -> Result<Weights, GeometryError> {
    let all = filter_weights(n_in, n_out, antialias)?;
    let total = all.first.len();
    let a = usize::try_from(left).unwrap_or(usize::MAX).min(total);
    let b = usize::try_from(left.saturating_add(take))
        .unwrap_or(total)
        .min(total);
    if a >= b {
        return Err(GeometryError::EmptyWindow);
    }
    let kmax = all.kmax;
    let n = b - a;
    let mut blob = Vec::with_capacity(n * (8 + 4 * kmax));
    for f in &all.first[a..b] {
        blob.extend_from_slice(&f.to_le_bytes());
    }
    for c in &all.count[a..b] {
        blob.extend_from_slice(&c.to_le_bytes());
    }
    for w in &all.weights[a * kmax..b * kmax] {
        blob.extend_from_slice(&w.to_le_bytes());
    }
    let lo = i64::from(all.first[a]);
    let hi = (a..b)
        .map(|i| i64::from(all.first[i]) + i64::from(all.count[i]))
        .max()
        .unwrap_or(lo);
    Ok(Weights {
        blob,
        n,
        kmax,
        lo,
        hi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_wide_photo_is_fitted_and_centred_with_the_odd_pixel_below() {
        let lb = letterbox_geometry(640, 480, 640).expect("geometry");
        assert_eq!((lb.new_w, lb.new_h, lb.left, lb.top), (640, 480, 0, 80));
        assert_eq!((lb.scale, lb.pad_w, lb.pad_h), (1.0, 0.0, 80.0));
        let odd = letterbox_geometry(100, 99, 224).expect("geometry");
        assert_eq!((odd.new_w, odd.new_h, odd.left, odd.top), (224, 222, 0, 1));
        assert_eq!(
            letterbox_geometry(0, 5, 224),
            Err(GeometryError::DegenerateSize { w: 0, h: 5 })
        );
        assert_eq!(
            letterbox_geometry(5, -1, 224),
            Err(GeometryError::DegenerateSize { w: 5, h: -1 })
        );
    }

    #[test]
    fn rounding_is_to_even_as_pythons_round_is() {
        // 2 x 1 into 5: scale 2.5, so the height is 2.5 pixels and Python's round gives 2
        let lb = letterbox_geometry(2, 1, 5).expect("geometry");
        assert_eq!(
            (lb.new_w, lb.new_h, lb.left, lb.top, lb.scale),
            (5, 2, 0, 1, 2.5)
        );
        let tiny = letterbox_geometry(5, 1, 2).expect("geometry");
        assert_eq!(
            (tiny.new_w, tiny.new_h),
            (2, 1),
            "a side never falls below one pixel"
        );
        assert_eq!(2.5f64.round_ties_even(), 2.0);
        assert_eq!(3.5f64.round_ties_even(), 4.0);
        assert_eq!((-0.5f64).round_ties_even(), 0.0);
    }

    #[test]
    fn the_square_takes_the_shorter_edge_and_centres_the_longer() {
        let sq = whole_square_geometry(640, 480, 336).expect("geometry");
        assert_eq!((sq.new_w, sq.new_h, sq.left, sq.top), (448, 336, 56, 0));
        assert_eq!(
            whole_square_geometry(1, 1, 336).expect("geometry").new_w,
            336
        );
        assert!(whole_square_geometry(0, 1, 336).is_err());
    }

    #[test]
    fn a_box_is_unmapped_and_cut_to_the_frame() {
        assert_eq!(
            undo_letterbox([100.0, 180.0, 200.0, 280.0], 0.5, (0.0, 80.0)),
            [200.0, 200.0, 400.0, 400.0]
        );
        assert_eq!(
            undo_letterbox([1.0, 1.0, 2.0, 2.0], 0.0, (0.0, 0.0)),
            [1e6, 1e6, 2e6, 2e6]
        );
        assert_eq!(crop_rect([-3.0, 10.4, 5.5, 10.5], 100, 50), [0, 10, 6, 11]);
        assert_eq!(
            crop_rect([99.6, 49.6, 200.0, 200.0], 100, 50),
            [99, 49, 100, 50]
        );
        assert_eq!(crop_rect([0.0, 0.0, 0.0, 0.0], 0, 0), [0, 0, 1, 1]);
        assert_eq!(
            stored_box_rects(&[[0.1, 0.2, 0.3, 0.4]], 100, 50),
            vec![[10, 10, 30, 20]]
        );
        assert_eq!(blur_rect([-5.0, 2.4, 50.0, 2.4], 10, 10), None);
        assert_eq!(
            blur_rect([-5.0, 2.4, 50.0, 8.6], 10, 10),
            Some([0, 2, 10, 9])
        );
    }

    #[test]
    fn a_two_to_one_downscale_has_the_triangle_filters_taps() {
        let f = filter_weights(4, 2, true).expect("weights");
        assert_eq!(
            (f.kmax, f.first.as_slice(), f.count.as_slice()),
            (3, [0, 1].as_slice(), [3, 3].as_slice())
        );
        assert_eq!(f.weights[..3], [0.42857143f32, 0.42857143, 0.14285715]);
        assert_eq!(f.weights[3..], [0.14285715f32, 0.42857143, 0.42857143]);
        let plain = filter_weights(4, 2, false).expect("weights");
        assert_eq!(
            (plain.kmax, plain.first.as_slice()),
            (2, [0, 2].as_slice()),
            "the support does not widen"
        );
        assert_eq!(filter_weights(4, 0, true), Err(GeometryError::EmptyWindow));
    }

    #[test]
    fn the_packed_set_is_first_count_then_weights_for_the_asked_window() {
        let w = weights(10, 5, 1, 3, true).expect("weights");
        assert_eq!((w.n, w.kmax), (3, 4));
        assert_eq!(w.blob.len(), 3 * 4 + 3 * 4 + 3 * 4 * 4);
        assert_eq!(
            i32::from_le_bytes(w.blob[0..4].try_into().expect("4 bytes")),
            w.lo as i32
        );
        assert!(w.hi > w.lo);
        assert_eq!(weights(10, 5, 5, 3, true), Err(GeometryError::EmptyWindow));
        assert_eq!(weights(10, 5, 4, 100, true).expect("clamped").n, 1);
    }

    #[test]
    fn the_pairwise_sum_matches_a_plain_sum_on_exact_values_and_splits_long_rows() {
        let ones: Vec<f64> = vec![1.0; 300];
        assert_eq!(pairwise_sum(&ones), 300.0);
        assert_eq!(pairwise_sum(&ones[..7]), 7.0);
        assert_eq!(pairwise_sum(&ones[..100]), 100.0);
        assert_eq!(pairwise_sum(&[]), 0.0);
    }
}
