//! `cv2.minAreaRect` and `cv2.boxPoints` of OpenCV 5.0.0 for integer points, and the Python
//! engine's `_mini_box` over them (`frame_engine/ocr/db_boxes.py`, PaddleOCR's
//! `get_mini_boxes`). Ported from `modules/geometry/src/convhull.cpp` (`convexHull`, Sklansky's
//! scan), `rotcalipers.cpp` (`rotatingCalipers`, `minAreaRect`) and `modules/core/src/types.cpp`
//! (`RotatedRect::points`) at the tag `5.0.0`
//! (<https://github.com/opencv/opencv/tree/5.0.0/modules/geometry/src>).
//!
//! The arithmetic is OpenCV's type for type: the hull in integers, the calipers in `float` with
//! the few `double` steps the source has, and the hull's start chosen by the input indices as
//! `convexHull` chooses it (the calipers keep the last of equal areas, so the start shows in the
//! answer). OpenCV's portable build has no fused multiply-add here, and neither has Rust's `f32`.

use super::contours::Point;

/// `cv::RotatedRect` as `minAreaRect` answers it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatedRect {
    /// The centre, in input pixels.
    pub center: [f32; 2],
    /// Width and height as OpenCV orders them.
    pub size: [f32; 2],
    /// Degrees, in `[-90, 0)`.
    pub angle: f32,
}

fn sign(v: i64) -> i32 {
    i32::from(v > 0) - i32::from(v < 0)
}

/// `Sklansky_<int, int64>` over `p` (sorted), from `start` to `end`, into `stack`.
fn sklansky(p: &[Point], start: isize, end: isize, nsign: i32, sign2: i32) -> Vec<isize> {
    let at = |i: isize| p[i as usize];
    let incr: isize = if end > start { 1 } else { -1 };
    if start == end || at(start) == at(end) {
        return vec![start];
    }
    let (mut pprev, mut pcur, mut pnext) = (start, start + incr, start + 2 * incr);
    let mut stack = vec![pprev, pcur, pnext];
    let stop = end + incr;
    while pnext != stop {
        let cury = i64::from(at(pcur)[1]);
        let by = i64::from(at(pnext)[1]) - cury;
        let n = stack.len();
        if sign(by) != nsign {
            let ax = i64::from(at(pcur)[0]) - i64::from(at(pprev)[0]);
            let ay = cury - i64::from(at(pprev)[1]);
            let bx = i64::from(at(pnext)[0]) - i64::from(at(pcur)[0]);
            let convexity = ay * bx - ax * by;
            if sign(convexity) == sign2 && (ax != 0 || ay != 0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack.push(pnext);
            } else if pprev == start {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[n - 2] = pnext;
                pcur = pprev;
                pprev = stack[n - 4];
                stack.pop();
            }
        } else {
            pnext += incr;
            stack[n - 1] = pnext;
        }
    }
    stack.pop();
    stack
}

/// The hull's sorted indices: the upper chains, then the lower ones (`clockwise = false`).
fn hull_sorted(p: &[Point], miny: isize, maxy: isize) -> Vec<isize> {
    let last = p.len() as isize - 1;
    let mut out = Vec::new();
    let tl = sklansky(p, 0, maxy, -1, 1);
    let tr = sklansky(p, last, maxy, -1, -1);
    // counter-clockwise: the right chain is walked first
    let (tl, tr) = (tr, tl);
    out.extend(&tl[..tl.len().saturating_sub(1)]);
    out.extend(tr[1..].iter().rev());
    let stop = if tr.len() > 2 {
        Some(tr[1])
    } else if tl.len() > 2 {
        Some(tl[tl.len() - 2])
    } else {
        None
    };
    let mut bl = sklansky(p, 0, miny, 1, -1);
    let mut br = sklansky(p, last, miny, 1, 1);
    if let Some(stop) = stop {
        let check = if bl.len() > 2 {
            Some(bl[1])
        } else if bl.len() + br.len() > 2 {
            Some(br[2 - bl.len()])
        } else {
            None
        };
        if check.is_some_and(|c| c == stop || p[c as usize] == p[stop as usize]) {
            bl.truncate(2);
            br.truncate(2);
        }
    }
    out.extend(&bl[..bl.len().saturating_sub(1)]);
    out.extend(br[1..].iter().rev());
    out
}

/// `convexHull(points, hull, clockwise = false, returnPoints = true)`.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by_key(|&i| (points[i][0], points[i][1], i));
    let sorted: Vec<Point> = order.iter().map(|&i| points[i]).collect();
    let (mut miny, mut maxy) = (0usize, 0usize);
    for (i, q) in sorted.iter().enumerate().skip(1) {
        if sorted[miny][1] > q[1] {
            miny = i;
        }
        if sorted[maxy][1] < q[1] {
            maxy = i;
        }
    }
    let mut hull: Vec<usize> = if sorted[0] == sorted[total - 1] {
        vec![0]
    } else {
        hull_sorted(&sorted, miny as isize, maxy as isize)
            .into_iter()
            .map(|i| i as usize)
            .collect()
    };
    for h in hull.iter_mut() {
        *h = order[*h];
    }
    cyclic_shift(&mut hull);
    hull.into_iter().map(|i| points[i]).collect()
}

/// `convexHull`'s rotation of the answer so its input indices ascend or descend, when a cyclic
/// shift can make them so.
fn cyclic_shift(hull: &mut [usize]) {
    let nout = hull.len();
    if nout < 3 {
        return;
    }
    let (mut min_idx, mut max_idx, mut lt) = (0usize, 0usize, 0usize);
    for i in 1..nout {
        let idx = hull[i];
        lt += usize::from(hull[i - 1] < idx);
        if lt > 1 && lt + 2 <= i {
            break;
        }
        if idx < hull[min_idx] {
            min_idx = i;
        }
        if idx > hull[max_idx] {
            max_idx = i;
        }
    }
    let mmdist = max_idx.abs_diff(min_idx);
    if !((mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout)) {
        return;
    }
    let ascending = (max_idx + 1) % nout == min_idx;
    let i0 = if ascending { min_idx } else { max_idx };
    if i0 == 0 {
        return;
    }
    let mut shifted = Vec::with_capacity(nout);
    let mut j = i0;
    for i in 0..nout {
        let curr = hull[j];
        shifted.push(curr);
        let next_j = if j + 1 < nout { j + 1 } else { 0 };
        if i < nout - 1 && (ascending != (curr < hull[next_j])) {
            return;
        }
        j = next_j;
    }
    hull.copy_from_slice(&shifted);
}

/// `rotatingCalipers(points, n, 1.f, CALIPERS_MINAREARECT)`: the corner and the two side vectors.
fn calipers(p: &[[f32; 2]]) -> [[f32; 2]; 3] {
    let n = p.len();
    let mut vect = vec![[0f32; 2]; n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = p[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0[0], pt0[0], pt0[1], pt0[1]);
    for i in 0..n {
        if pt0[0] < left_x {
            (left_x, left) = (pt0[0], i);
        }
        if pt0[0] > right_x {
            (right_x, right) = (pt0[0], i);
        }
        if pt0[1] > top_y {
            (top_y, top) = (pt0[1], i);
        }
        if pt0[1] < bottom_y {
            (bottom_y, bottom) = (pt0[1], i);
        }
        let pt = p[if i + 1 < n { i + 1 } else { 0 }];
        let dx = f64::from(pt[0] - pt0[0]);
        let dy = f64::from(pt[1] - pt0[1]);
        vect[i] = [dx as f32, dy as f32];
        inv_len[i] = (1.0 / (dx * dx + dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    let mut seq = [bottom, right, top, left];
    let (mut minarea, mut best) = (f32::MAX, (0usize, 0f32, 0f32, 0f32, 0f32, 0usize));
    for _ in 0..n {
        let v = |i: usize| vect[seq[i]];
        let rot = [
            v(0),
            [v(1)[1], -v(1)[0]],
            [-v(2)[0], -v(2)[1]],
            [-v(3)[1], v(3)[0]],
        ];
        let mut main = 0usize;
        for i in 1..4 {
            // `firstVecIsRight`: the clockwise turn of the first against the second
            let t = [rot[i][1], -rot[i][0]];
            if t[0] * rot[main][0] + t[1] * rot[main][1] < 0.0 {
                main = i;
            }
        }
        let pindex = seq[main];
        let lead = [
            vect[pindex][0] * inv_len[pindex],
            vect[pindex][1] * inv_len[pindex],
        ];
        let (base_a, base_b) = match main {
            0 => (lead[0], lead[1]),
            1 => (lead[1], -lead[0]),
            2 => (-lead[0], -lead[1]),
            _ => (-lead[1], lead[0]),
        };
        seq[main] = if seq[main] + 1 == n { 0 } else { seq[main] + 1 };
        let dx = p[seq[1]][0] - p[seq[3]][0];
        let dy = p[seq[1]][1] - p[seq[3]][1];
        let width = dx * base_a + dy * base_b;
        let dx = p[seq[2]][0] - p[seq[0]][0];
        let dy = p[seq[2]][1] - p[seq[0]][1];
        let height = -dx * base_b + dy * base_a;
        let area = width * height;
        if area <= minarea {
            minarea = area;
            best = (seq[3], base_a, width, base_b, height, seq[0]);
        }
    }
    let (li, a1, width, b1, height, bi) = best;
    let (a2, b2) = (-b1, a1);
    let c1 = a1 * p[li][0] + p[li][1] * b1;
    let c2 = a2 * p[bi][0] + p[bi][1] * b2;
    let idet = 1.0 / (a1 * b2 - a2 * b1);
    let px = (c1 * b2 - c2 * b1) * idet;
    let py = (a1 * c2 - a2 * c1) * idet;
    [
        [px, py],
        [a1 * width, b1 * width],
        [a2 * height, b2 * height],
    ]
}

/// `cv::minAreaRect` of integer points.
pub fn min_area_rect(points: &[Point]) -> RotatedRect {
    let hull: Vec<[f32; 2]> = convex_hull(points)
        .iter()
        .map(|q| [q[0] as f32, q[1] as f32])
        .collect();
    let mut angle = -std::f64::consts::PI / 2.0;
    let mut center = [0f32; 2];
    let mut size = [0f32; 2];
    match hull.len() {
        0 => {}
        1 => center = hull[0],
        2 => {
            let (a, b) = (hull[0], hull[1]);
            center = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
            let dx = f64::from(a[0] - b[0]);
            let dy = f64::from(a[1] - b[1]);
            size = [0.0, (dx * dx + dy * dy).sqrt() as f32];
            if dx == 0.0 {
                size.swap(0, 1);
            } else if dy < 0.0 {
                angle = dy.atan2(dx);
                size.swap(0, 1);
            } else if dy > 0.0 {
                angle = -dx.atan2(dy);
            }
        }
        _ => {
            let [o, s1, s2] = calipers(&hull);
            center = [o[0] + (s1[0] + s2[0]) * 0.5, o[1] + (s1[1] + s2[1]) * 0.5];
            let len = |v: [f32; 2]| {
                (f64::from(v[0]) * f64::from(v[0]) + f64::from(v[1]) * f64::from(v[1])).sqrt()
                    as f32
            };
            size = [len(s2), len(s1)];
            if s1[0] == 0.0 && s1[1] > 0.0 {
                size.swap(0, 1);
            } else {
                angle = -f64::from(s1[0]).atan2(f64::from(s1[1]));
            }
        }
    }
    RotatedRect {
        center,
        size,
        angle: (angle * 180.0 / std::f64::consts::PI) as f32,
    }
}

/// `cv::boxPoints`: the four corners (`RotatedRect::points`).
pub fn box_points(r: &RotatedRect) -> [[f32; 2]; 4] {
    let rad = f64::from(r.angle) * std::f64::consts::PI / 180.0;
    let b = rad.cos() as f32 * 0.5;
    let a = rad.sin() as f32 * 0.5;
    let [cx, cy] = r.center;
    let [w, h] = r.size;
    let (ah, aw, bh, bw) = (a * h, a * w, b * h, b * w);
    [
        [cx - ah - bw, cy + bh - aw],
        [cx + ah - bw, cy - bh - aw],
        [cx + ah + bw, cy - bh + aw],
        [cx - ah + bw, cy + bh + aw],
    ]
}

/// `_mini_box`: the rectangle's corners top-left, top-right, bottom-right, bottom-left, and its
/// shorter side.
pub fn mini_box(points: &[Point]) -> ([[f32; 2]; 4], f32) {
    let rect = min_area_rect(points);
    let mut pts = box_points(&rect);
    // Python's `sorted` by x is stable
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]));
    let (i1, i4) = if pts[1][1] > pts[0][1] {
        (0, 1)
    } else {
        (1, 0)
    };
    let (i2, i3) = if pts[3][1] > pts[2][1] {
        (2, 3)
    } else {
        (3, 2)
    };
    (
        [pts[i1], pts[i2], pts[i3], pts[i4]],
        rect.size[0].min(rect.size[1]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tilted_triangle_has_opencvs_rectangle() {
        // cv2.minAreaRect(np.array([[1, 2], [5, 7], [3, 9]]).reshape(-1, 1, 2)), OpenCV 5.0.0
        let r = min_area_rect(&[[1, 2], [5, 7], [3, 9]]);
        assert_eq!(r.center, [3.1886797, 5.1603765]);
        assert_eq!(r.size, [2.4724896, 7.280109]);
        assert_eq!(r.angle, -15.945396);
    }

    #[test]
    fn degenerate_inputs_answer_as_opencv_does() {
        let one = min_area_rect(&[[4, 5], [4, 5]]);
        assert_eq!(
            (one.center, one.size, one.angle),
            ([4.0, 5.0], [0.0, 0.0], -90.0)
        );
        let line = min_area_rect(&[[0, 0], [3, 0], [6, 0]]);
        // cv2.minAreaRect of three points on a row: ((3, 0), (0, 6), -90)
        assert_eq!(
            (line.center, line.size, line.angle),
            ([3.0, 0.0], [0.0, 6.0], -90.0)
        );
        assert_eq!(min_area_rect(&[]).size, [0.0, 0.0]);
    }
}
