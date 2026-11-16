//! Where each recogniser input pixel comes from, and how lines are batched: ports of
//! `frame_engine/ocr/crop_geometry.py`, `batching.py` and `det_input.py`.
//!
//! A line is cut as PaddleOCR cuts it (`get_rotate_crop_image`, then `resize_norm_img`): the
//! quadrilateral warped to a `cw x ch` rectangle, a quarter turn when it is tall, a resize to 48
//! pixels high. The three maps are composed into one 3 x 3 projective map per line, which the
//! warp kernel (`kernels/ocr_warp.cu`) samples once per recogniser pixel. The perspective solve is
//! `cv2.getPerspectiveTransform`'s (OpenCV 5.0.0 `modules/geometry/src/geometry.cpp`: the 8 x 8
//! system in `double` with `float` products, solved by `hal::LU`'s Gaussian elimination,
//! `modules/core/src/matrix_decomp.cpp`); the inverse is LAPACK's LU with partial pivoting, as
//! `np.linalg.inv` calls it. The map reaches the kernel as float32, which absorbs the last
//! `double` bits of either.

/// `REC_HEIGHT`: the recogniser reads lines 48 pixels high.
pub const REC_HEIGHT: i64 = 48;
/// `MIN_BATCH_WIDTH`: a batch is at least 320 pixels wide (`rec_image_shape [3, 48, 320]`).
const MIN_BATCH_WIDTH: f64 = 320.0;

/// `DetResizeForTest.resize_image_type0` (`det_size`): the detector input's `(height, width)`
/// for a frame of `h x w`, or `None` for an unknown limit.
pub fn det_size(h: i64, w: i64, side: i64, limit: &str, max_side: i64) -> Option<(i64, i64)> {
    let ratio = match limit {
        "max" if h.max(w) > side => side as f64 / h.max(w) as f64,
        "min" if h.min(w) < side => side as f64 / h.min(w) as f64,
        "max" | "min" => 1.0,
        _ => return None,
    };
    let (mut rh, mut rw) = ((h as f64 * ratio) as i64, (w as f64 * ratio) as i64);
    if rh.max(rw) > max_side {
        let shrink = max_side as f64 / rh.max(rw) as f64;
        (rh, rw) = ((rh as f64 * shrink) as i64, (rw as f64 * shrink) as i64);
    }
    let to32 = |v: i64| (((v as f64 / 32.0).round_ties_even() as i64) * 32).max(32);
    Some((to32(rh), to32(rw)))
}

/// `crop_dims`: the longer of each pair of opposite sides, truncated. numpy's norm of a float32
/// pair is `sqrt(x*x + y*y)` added in float32 in that order (OpenBLAS `sdot`).
pub fn crop_dims(q: &[[i32; 2]; 4]) -> (i64, i64) {
    let f = q.map(|p| [p[0] as f32, p[1] as f32]);
    let norm = |a: [f32; 2], b: [f32; 2]| {
        let (x, y) = (a[0] - b[0], a[1] - b[1]);
        (x * x + y * y).sqrt()
    };
    let cw = norm(f[0], f[1]).max(norm(f[2], f[3])) as i64;
    let ch = norm(f[0], f[3]).max(norm(f[1], f[2])) as i64;
    (cw, ch)
}

fn is_tall(cw: i64, ch: i64) -> bool {
    ch as f64 / cw as f64 >= 1.5
}

/// `rec_ratio`: width over height of the line after its quarter turn; `None` for a side under
/// one pixel.
pub fn rec_ratio(q: &[[i32; 2]; 4]) -> Option<f64> {
    let (cw, ch) = crop_dims(q);
    if cw < 1 || ch < 1 {
        return None;
    }
    Some(if is_tall(cw, ch) {
        ch as f64 / cw as f64
    } else {
        cw as f64 / ch as f64
    })
}

type M3 = [[f64; 3]; 3];

fn mul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    out
}

/// `hal::LU` (`LUImpl<double>`) solving `a x = b` in place; `false` when singular.
fn lu_solve<const N: usize>(a: &mut [[f64; N]; N], b: &mut [f64; N]) -> bool {
    for i in 0..N {
        let k = (i..N).fold(i, |k, j| if a[j][i].abs() > a[k][i].abs() { j } else { k });
        if a[k][i].abs() < f64::EPSILON * 100.0 {
            return false;
        }
        if k != i {
            a.swap(i, k);
            b.swap(i, k);
        }
        let d = -1.0 / a[i][i];
        let pivot = a[i];
        for j in i + 1..N {
            let alpha = a[j][i] * d;
            for (v, p) in a[j].iter_mut().zip(&pivot).skip(i + 1) {
                *v += alpha * p;
            }
            b[j] += alpha * b[i];
        }
    }
    for i in (0..N).rev() {
        let mut s = b[i];
        for k in i + 1..N {
            s -= a[i][k] * b[k];
        }
        b[i] = s / a[i][i];
    }
    true
}

/// `cv2.getPerspectiveTransform(src, dst)` (the solve by LU).
fn perspective(src: [[f32; 2]; 4], dst: [[f32; 2]; 4]) -> Option<M3> {
    let mut a = [[0f64; 8]; 8];
    let mut b = [0f64; 8];
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a[i] = [
            x.into(),
            y.into(),
            1.0,
            0.0,
            0.0,
            0.0,
            (-x * u).into(),
            (-y * u).into(),
        ];
        a[i + 4] = [
            0.0,
            0.0,
            0.0,
            x.into(),
            y.into(),
            1.0,
            (-x * v).into(),
            (-y * v).into(),
        ];
        b[i] = u.into();
        b[i + 4] = v.into();
    }
    lu_solve(&mut a, &mut b).then_some([[b[0], b[1], b[2]], [b[3], b[4], b[5]], [b[6], b[7], 1.0]])
}

/// The inverse by LU with partial pivoting (LAPACK `dgesv` on the identity).
fn inverse(m: &M3) -> Option<M3> {
    let mut cols = [[0f64; 3]; 3];
    for (c, col) in cols.iter_mut().enumerate() {
        let mut a = *m;
        let mut e = [0.0; 3];
        e[c] = 1.0;
        if !lu_solve(&mut a, &mut e) {
            return None;
        }
        *col = e;
    }
    Some([0, 1, 2].map(|r| [cols[0][r], cols[1][r], cols[2][r]]))
}

/// One line's sampling: the float32 map the kernel reads and its taps per axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineMap {
    /// Recogniser pixel `(u, v, 1)` to frame `(x, y, w)`, row-major float32.
    pub map: [f32; 9],
    /// Samples per axis inside one recogniser pixel.
    pub taps: [i32; 2],
}

/// `rec_map(quad, width)` and `taps_for(g, width, cap)`: the map from recogniser pixel
/// `(u, v, 1)` to frame `(x, y, w)` for a line resized to `width x 48` (row-major), and the
/// samples per axis inside one recogniser pixel (the frame pixels it spans at the line's centre,
/// rounded up, at most `cap`).
pub fn line_map(q: &[[i32; 2]; 4], width: i64, cap: u32) -> Option<LineMap> {
    let (cw, ch) = crop_dims(q);
    let (fw, fh) = (cw as f32, ch as f32);
    let src = q.map(|p| [p[0] as f32, p[1] as f32]);
    let std = [[0.0, 0.0], [fw, 0.0], [fw, fh], [0.0, fh]];
    let m_inv = inverse(&perspective(src, std)?)?;
    let (cwf, chf) = (cw as f64, ch as f64);
    let (w_rot, h_rot, to_crop) = if is_tall(cw, ch) {
        (
            chf,
            cwf,
            [[0.0, -1.0, cwf - 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        )
    } else {
        (
            cwf,
            chf,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
    };
    let (sx, sy) = (w_rot / width as f64, h_rot / REC_HEIGHT as f64);
    let resize = [
        [sx, 0.0, 0.5 * sx - 0.5],
        [0.0, sy, 0.5 * sy - 0.5],
        [0.0, 0.0, 1.0],
    ];
    let g = mul(&mul(&m_inv, &to_crop), &resize);
    Some(LineMap {
        map: [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|k| g[k / 3][k % 3] as f32),
        taps: taps(&g, width, cap),
    })
}

/// `footprint` and `taps_for` over the float64 map.
fn taps(g: &M3, width: i64, cap: u32) -> [i32; 2] {
    let (u, v) = (width as f64 / 2.0, REC_HEIGHT as f64 / 2.0);
    let at = |du: f64, dv: f64| {
        let p = [0, 1, 2].map(|r| g[r][0] * (u + du) + g[r][1] * (v + dv) + g[r][2]);
        [p[0] / p[2], p[1] / p[2]]
    };
    // numpy's norm of a float64 pair: `x*x + y*y` in that order, then the root
    let dist = |a: [f64; 2], b: [f64; 2]| {
        let (x, y) = (a[0] - b[0], a[1] - b[1]);
        (x * x + y * y).sqrt()
    };
    let su = dist(at(0.5, 0.0), at(-0.5, 0.0));
    let sv = dist(at(0.0, 0.5), at(0.0, -0.5));
    let one = |s: f64| ((s - 1e-3).ceil() as i64).clamp(1, i64::from(cap.max(1))) as i32;
    [one(su), one(sv)]
}

/// One recogniser call: lines (indices into the caller's list, narrowest first) padded to a
/// common width.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    /// Indices into the caller's lines, narrowest first.
    pub indices: Vec<usize>,
    /// The width every line of the call is padded to.
    pub width: i64,
}

/// `plan_batches(ratios, batch, max_width)`: at most `batch` lines by ascending ratio (a stable
/// sort); each batch `int(48 * max(320 / 48, widest ratio))` wide, at most `max_width`.
pub fn plan_batches(ratios: &[f64], batch: usize, max_width: i64) -> Vec<Batch> {
    let mut order: Vec<usize> = (0..ratios.len()).collect();
    order.sort_by(|a, b| ratios[*a].total_cmp(&ratios[*b]));
    order
        .chunks(batch.max(1))
        .map(|idx| {
            let widest = idx
                .iter()
                .map(|i| ratios[*i])
                .fold(MIN_BATCH_WIDTH / REC_HEIGHT as f64, f64::max);
            Batch {
                indices: idx.to_vec(),
                width: ((REC_HEIGHT as f64 * widest) as i64).min(max_width),
            }
        })
        .collect()
}

/// `resized_width`: the width a line of this ratio is resized to inside its batch.
pub fn resized_width(ratio: f64, batch_width: i64) -> i64 {
    ((REC_HEIGHT as f64 * ratio).ceil() as i64).min(batch_width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_detector_size_is_a_multiple_of_32_inside_the_limits() {
        assert_eq!(det_size(480, 640, 736, "min", 4000), Some((736, 992)));
        assert_eq!(det_size(3000, 4000, 960, "max", 4000), Some((704, 960)));
        assert_eq!(det_size(10, 10, 960, "max", 4000), Some((32, 32)));
        assert_eq!(det_size(10, 10, 960, "mid", 4000), None);
    }

    #[test]
    fn lines_are_batched_narrowest_first_and_padded_to_the_widest() {
        let b = plan_batches(&[10.0, 2.0, 5.0, 2.0, 50.0], 2, 2048);
        assert_eq!(
            b,
            vec![
                Batch {
                    indices: vec![1, 3],
                    width: 320
                },
                Batch {
                    indices: vec![2, 0],
                    width: 480
                },
                Batch {
                    indices: vec![4],
                    width: 2048
                },
            ]
        );
        assert_eq!(resized_width(2.0, 320), 96);
        assert_eq!(resized_width(50.0, 2048), 2048);
        assert!(plan_batches(&[], 4, 2048).is_empty());
    }

    #[test]
    fn a_square_box_maps_its_centre_back_to_itself() {
        let q = [[10, 20], [110, 20], [110, 60], [10, 60]];
        assert_eq!(crop_dims(&q), (100, 40));
        assert_eq!(rec_ratio(&q), Some(2.5));
        // rec_map(q, 120) of the Python engine, as float32: scale 100 / 120 and 40 / 48, the
        // offset of a half pixel (its entries of about 1e-17 are rounding noise of its solve)
        let m = line_map(&q, 120, 4).expect("a map");
        let python = [
            0.8333333f32,
            0.0,
            9.916667,
            0.0,
            0.8333333,
            19.916666,
            0.0,
            0.0,
            1.0,
        ];
        for (got, want) in m.map.iter().zip(python) {
            assert!(
                (got - want).abs() <= 1e-6 * want.abs().max(1e-9),
                "{:?}",
                m.map
            );
        }
        assert_eq!(m.taps, [1, 1]);
        assert_eq!(rec_ratio(&[[0, 0], [0, 0], [0, 0], [0, 0]]), None);
    }
}
