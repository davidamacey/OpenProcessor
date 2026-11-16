//! The unclip of a DB box (`frame_engine/ocr/db_boxes.py` `_unclip`): pyclipper 1.4.0's
//! `PyclipperOffset` (Clipper 6.4.2, `JT_ROUND`, `ET_CLOSEDPOLYGON`, miter limit 2, arc tolerance
//! 0.25) grown by `area * ratio / perimeter`, the area and perimeter OpenCV's `contourArea` and
//! `arcLength` of the float corners.
//!
//! Ported from Clipper 6.4.2's `clipper.cpp` as pyclipper 1.4.0 ships it
//! (<https://github.com/fonttools/pyclipper>, `src/clipper.cpp`; Clipper by Angus Johnson, Boost
//! Software License 1.0, recorded in `ATTRIBUTION.md`): `ClipperOffset::AddPath` (duplicates
//! dropped), `FixOrientations`, `DoOffset`, `OffsetPoint` and `DoRound`, with pyclipper's
//! conversion of a float corner to an integer (truncation toward zero).
//!
//! `Execute` ends with a positive-fill union of the offset outline; the general union (Vatti's
//! sweep, most of `clipper.cpp`) is not ported, its answer for this input is. The corners come
//! from a rotated rectangle at least 3 pixels on a side (the post-process's gate), and moving each
//! corner by less than one pixel per axis cannot make such a quadrilateral concave or crossed, so
//! its outline grown with round joins is a simple polygon: the union keeps its vertices in their
//! order, drops repeated and collinear ones (`FixupOutPolygon`), and its answer ends at the
//! topmost point (smallest `y`, then largest `x`). The caller reads the union only through
//! `minAreaRect`, which sees the convex hull and the input order of its vertices; both are
//! reproduced (measured on 20,000 rectangles against pyclipper, and held by the goldens,
//! `tests/golden/ocr_db.json`).

// `clipper.cpp`'s `pi` (3.141592653589793238) is this double
const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = PI * 2.0;
const ARC_TOLERANCE: f64 = 0.25;

/// `Round` of `clipper.cpp`: half away from zero.
fn round(v: f64) -> i64 {
    if v < 0.0 {
        (v - 0.5) as i64
    } else {
        (v + 0.5) as i64
    }
}

/// `Area` of a closed integer path (positive when its orientation is Clipper's positive).
fn area(p: &[[i64; 2]]) -> f64 {
    if p.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    let mut j = p.len() - 1;
    for i in 0..p.len() {
        a += (p[j][0] as f64 + p[i][0] as f64) * (p[j][1] as f64 - p[i][1] as f64);
        j = i;
    }
    -a * 0.5
}

fn unit_normal(a: [i64; 2], b: [i64; 2]) -> [f64; 2] {
    if a == b {
        return [0.0, 0.0];
    }
    let (dx, dy) = ((b[0] - a[0]) as f64, (b[1] - a[1]) as f64);
    let f = 1.0 / (dx * dx + dy * dy).sqrt();
    [dy * f, -(dx * f)]
}

/// `ClipperOffset::AddPath` for a closed polygon: repeated points dropped; `None` below 3.
fn cleaned(path: &[[i64; 2]]) -> Option<Vec<[i64; 2]>> {
    let mut high = path.len().checked_sub(1)?;
    while high > 0 && path[0] == path[high] {
        high -= 1;
    }
    let mut out = vec![path[0]];
    for &p in &path[1..=high] {
        if out.last() != Some(&p) {
            out.push(p);
        }
    }
    (out.len() >= 3).then_some(out)
}

/// The offset outline (`DoOffset`) of `path` by `delta` with round joins.
fn do_offset(src: &[[i64; 2]], delta: f64) -> Vec<[i64; 2]> {
    if delta.abs() < 1e-20 {
        return src.to_vec();
    }
    let y = if ARC_TOLERANCE > delta.abs() * ARC_TOLERANCE {
        delta.abs() * ARC_TOLERANCE
    } else {
        ARC_TOLERANCE
    };
    let mut steps = PI / (1.0 - y / delta.abs()).acos();
    if steps > delta.abs() * PI {
        steps = delta.abs() * PI;
    }
    let mut sin = (TWO_PI / steps).sin();
    let cos = (TWO_PI / steps).cos();
    let steps_per_rad = steps / TWO_PI;
    if delta < 0.0 {
        sin = -sin;
    }
    let len = src.len();
    let normals: Vec<[f64; 2]> = (0..len)
        .map(|j| unit_normal(src[j], src[(j + 1) % len]))
        .collect();
    let at = |p: [i64; 2], n: [f64; 2]| {
        [
            round(p[0] as f64 + n[0] * delta),
            round(p[1] as f64 + n[1] * delta),
        ]
    };
    let mut out = Vec::new();
    let mut k = len - 1;
    for j in 0..len {
        let (nk, nj) = (normals[k], normals[j]);
        let mut sin_a = nk[0] * nj[1] - nj[0] * nk[1];
        if (sin_a * delta).abs() < 1.0 {
            if nk[0] * nj[0] + nj[1] * nk[1] > 0.0 {
                // Clipper returns here without moving `k` on to this vertex
                out.push(at(src[j], nk));
                continue;
            }
        } else {
            sin_a = sin_a.clamp(-1.0, 1.0);
        }
        if sin_a * delta < 0.0 {
            out.push(at(src[j], nk));
            out.push(src[j]);
            out.push(at(src[j], nj));
        } else {
            // `DoRound`
            let a = sin_a.atan2(nk[0] * nj[0] + nk[1] * nj[1]);
            let n = (round(steps_per_rad * a.abs()) as i32).max(1);
            let (mut x, mut yy) = (nk[0], nk[1]);
            for _ in 0..n {
                out.push(at(src[j], [x, yy]));
                let x2 = x;
                x = x * cos - sin * yy;
                yy = x2 * sin + yy * cos;
            }
            out.push(at(src[j], nj));
        }
        k = j;
    }
    out
}

/// `_unclip(box, ratio)`: the grown outline of the float corners, `None` where Python's is.
pub fn unclip(corners: &[[f32; 2]; 4], ratio: f64) -> Option<Vec<[i64; 2]>> {
    // OpenCV's `contourArea` and `arcLength` of float points
    let mut a00 = 0f64;
    let mut length = 0f64;
    let mut prev = corners[3];
    for p in corners {
        a00 += f64::from(prev[0]) * f64::from(p[1]) - f64::from(prev[1]) * f64::from(p[0]);
        let (dx, dy) = (p[0] - prev[0], p[1] - prev[1]);
        length += f64::from((dx * dx + dy * dy).sqrt());
        prev = *p;
    }
    let area_cv = (a00 * 0.5).abs();
    if length == 0.0 {
        return None;
    }
    let path: Vec<[i64; 2]> = corners.iter().map(|p| [p[0] as i64, p[1] as i64]).collect();
    let mut src = cleaned(&path)?;
    if area(&src) < 0.0 {
        src.reverse();
    }
    let grown = union_order(do_offset(&src, area_cv * ratio / length));
    (!grown.is_empty()).then_some(grown)
}

/// The union's answer for a simple outline: repeated and collinear points dropped until none is
/// left, then turned so it ends at the topmost point (smallest `y`, then largest `x`).
fn union_order(mut p: Vec<[i64; 2]>) -> Vec<[i64; 2]> {
    let mut i = 0;
    while p.len() > 2 && i < p.len() {
        let n = p.len();
        let (a, b, c) = (p[(i + n - 1) % n], p[i], p[(i + 1) % n]);
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross == 0 {
            p.remove(i);
            i = i.saturating_sub(1);
        } else {
            i += 1;
        }
    }
    if let Some(top) = (0..p.len()).min_by_key(|&k| (p[k][1], -p[k][0])) {
        p.rotate_left(top + 1);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_is_half_away_from_zero_and_orientation_is_fixed() {
        assert_eq!((round(2.5), round(-2.5), round(0.49)), (3, -3, 0));
        let cw = [[0, 0], [0, 10], [10, 10], [10, 0]];
        assert!(area(&cw) < 0.0);
        assert_eq!(cleaned(&[[1, 1], [1, 1], [2, 2], [1, 1]]), None);
    }

    #[test]
    fn a_tiny_offset_is_the_truncated_corners() {
        // pyclipper: Execute(0.0001) of [[0.9, 0.9], [10.7, 0.2], [10.6, 5.9], [-0.7, 5.5]] is
        // the corners truncated, (0, 0), (10, 0), (10, 5), (0, 5)
        let src = cleaned(&[[0, 0], [10, 0], [10, 5], [0, 5]]).expect("a polygon");
        let grown = do_offset(&src, 0.0001);
        for p in [[0, 0], [10, 0], [10, 5], [0, 5]] {
            assert!(grown.contains(&p), "{p:?} in {grown:?}");
        }
    }
}
