//! Five-point alignment (`frame_engine/faces/alignment.py`): the similarity from a face's five
//! landmarks to the ArcFace template (Umeyama's least squares, scale estimated, as scikit-image's
//! `_umeyama` and so InsightFace's `estimate_norm`), its inverse, and the warp kernel's taps.
//!
//! The SVD is `nalgebra`'s (https://docs.rs/nalgebra/0.35.0/nalgebra/linalg/struct.SVD.html,
//! `try_new`, singular values in descending order as LAPACK's `gesdd` gives them). The result
//! does not depend on the singular vectors' signs: every product here pairs a left vector with
//! its right one. Sums run in the order numpy's reductions run them on five rows; the reference's
//! two small matrix products go through BLAS, whose order is the library's, so the forward matrix
//! matches the reference to its last bits, not always to the bit (the golden test states the
//! bound).

use nalgebra::{Matrix2, Matrix3, SVD, Vector2};

/// The recogniser's input side.
pub const ARCFACE_SIZE: usize = 112;
/// The ArcFace template (InsightFace `utils/face_align.py` `arcface_dst`): left eye, right eye,
/// nose tip, left and right mouth corners, in 112 x 112 output pixels.
pub const ARCFACE_DST: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];
/// Upper bound of the per-axis tap count.
pub const MAX_SAMPLES: i32 = 16;

/// A 2 x 3 affine map, rows `[a, b, tx]`, `[c, d, ty]`.
pub type Affine = [[f64; 3]; 2];

/// What the warp kernel reads for one face: its output-to-source map (`[6]` float32), its taps
/// per axis, and whether the landmarks gave a usable transform (a face without one is not
/// embedded).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceWarp {
    pub inv: [f32; 6],
    pub taps: i32,
    pub ok: bool,
}

/// The determinant of a 2 x 2 matrix as LAPACK's partial-pivot LU gives it (`np.linalg.det`),
/// so its sign at a near-singular matrix is the reference's.
fn det2(m: [[f64; 2]; 2]) -> f64 {
    let [[a, b], [c, d]] = m;
    if c.abs() > a.abs() {
        -(c * (b - a / c * d))
    } else if a == 0.0 {
        0.0
    } else {
        a * (d - c / a * b)
    }
}

fn mat2(m: &Matrix2<f64>) -> [[f64; 2]; 2] {
    [[m[(0, 0)], m[(0, 1)]], [m[(1, 0)], m[(1, 1)]]]
}

/// Column means of five points, summed in row order and divided by five (`ndarray.mean(0)`).
fn mean(p: &[[f64; 2]; 5]) -> [f64; 2] {
    let sum = |j: usize| p.iter().fold(0.0, |acc, r| acc + r[j]);
    [sum(0) / 5.0, sum(1) / 5.0]
}

fn demean(p: &[[f64; 2]; 5]) -> ([[f64; 2]; 5], [f64; 2]) {
    let m = mean(p);
    (p.map(|r| [r[0] - m[0], r[1] - m[1]]), m)
}

/// `var(axis=0).sum()` of demeaned points: each column's mean of squared deviations.
fn var_sum(p: &[[f64; 2]; 5]) -> f64 {
    let m = mean(p);
    let var = |j: usize| {
        p.iter().fold(0.0, |acc, r| {
            let x = r[j] - m[j];
            acc + x * x
        }) / 5.0
    };
    var(0) + var(1)
}

/// The rotation part `u diag(d) v` per the reference's rank cases; `None` at rank 0.
fn rotation(
    svd: &SVD<f64, nalgebra::U2, nalgebra::U2>,
    a_det: f64,
) -> Option<(Matrix2<f64>, [f64; 2])> {
    let (u, vt) = (svd.u?, svd.v_t?);
    let s = svd.singular_values;
    let tol = s[0].max(s[1]) * 2.0 * f64::EPSILON;
    let rank = s.iter().filter(|v| **v > tol).count();
    let d = [1.0, if a_det < 0.0 { -1.0 } else { 1.0 }];
    let diag = |d1: f64| Matrix2::from_diagonal(&Vector2::new(1.0, d1));
    match rank {
        0 => None,
        1 if det2(mat2(&u)) * det2(mat2(&vt)) > 0.0 => Some((u * vt, d)),
        1 => Some((u * diag(-1.0) * vt, d)),
        _ => Some((u * diag(d[1]) * vt, d)),
    }
}

/// `umeyama(src, ARCFACE_DST)[:2]` as `estimate`: the forward map of one face's landmarks (source
/// pixels to the 112 output). NaN at a degenerate set (rank 0); an error where numpy's SVD
/// fails (a non-finite landmark).
pub fn estimate(src: &[[f64; 2]; 5]) -> Result<Affine, String> {
    let (src_d, src_mean) = demean(src);
    let (dst_d, dst_mean) = demean(&ARCFACE_DST);
    let mut a = Matrix2::zeros();
    for i in 0..2 {
        for j in 0..2 {
            a[(i, j)] = (0..5).fold(0.0, |acc, k| acc + dst_d[k][i] * src_d[k][j]) / 5.0;
        }
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err("the landmarks are not finite: SVD did not converge".into());
    }
    let svd = SVD::try_new(a, true, true, f64::EPSILON, 0).ok_or("SVD did not converge")?;
    let Some((t, d)) = rotation(&svd, det2(mat2(&a))) else {
        return Ok([[f64::NAN; 3]; 2]);
    };
    let s = svd.singular_values;
    let scale = 1.0 / var_sum(&src_d) * (s[0] * d[0] + s[1] * d[1]);
    let mut out = [[0.0; 3]; 2];
    for (i, row) in out.iter_mut().enumerate() {
        let moved = t[(i, 0)] * src_mean[0] + t[(i, 1)] * src_mean[1];
        *row = [
            t[(i, 0)] * scale,
            t[(i, 1)] * scale,
            dst_mean[i] - scale * moved,
        ];
    }
    Ok(out)
}

/// The map from output pixels back to source pixels; `None` when `forward` has no inverse.
pub fn inverse(forward: &Affine) -> Option<Affine> {
    let [[a, b, tx], [c, d, ty]] = *forward;
    let inv = Matrix3::new(a, b, tx, c, d, ty, 0.0, 0.0, 1.0).try_inverse()?;
    Some([
        [inv[(0, 0)], inv[(0, 1)], inv[(0, 2)]],
        [inv[(1, 0)], inv[(1, 1)], inv[(1, 2)]],
    ])
}

/// Taps per axis for one face: the source pixels one output pixel spans, rounded up.
pub fn samples(inv: &Affine) -> i32 {
    let span = det2([[inv[0][0], inv[0][1]], [inv[1][0], inv[1][1]]])
        .abs()
        .sqrt();
    if !span.is_finite() {
        return 1;
    }
    (span - 1e-6).ceil().clamp(1.0, f64::from(MAX_SAMPLES)) as i32
}

/// `warp_params` for one face's landmarks (`x, y` five times, float32 as the detector gave them).
pub fn warp_params(landmarks: &[f32; 10]) -> Result<FaceWarp, String> {
    let mut src = [[0.0; 2]; 5];
    for (k, p) in src.iter_mut().enumerate() {
        *p = [f64::from(landmarks[2 * k]), f64::from(landmarks[2 * k + 1])];
    }
    let fwd = estimate(&src)?;
    let unusable = FaceWarp {
        inv: [0.0; 6],
        taps: 1,
        ok: false,
    };
    let finite = fwd.iter().flatten().all(|v| v.is_finite());
    if !finite || det2([[fwd[0][0], fwd[0][1]], [fwd[1][0], fwd[1][1]]]).abs() < 1e-12 {
        return Ok(unusable);
    }
    let Some(m) = inverse(&fwd) else {
        return Ok(unusable);
    };
    let flat = [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2]];
    Ok(FaceWarp {
        inv: flat.map(|v| v as f32),
        taps: samples(&m),
        ok: true,
    })
}
