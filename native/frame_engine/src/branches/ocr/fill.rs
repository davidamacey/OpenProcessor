//! `box_score_fast` (`frame_engine/ocr/db_boxes.py` `_score`): the mean of the detector's map
//! inside a quadrilateral, through `cv2.fillPoly` of OpenCV 5.0.0 (line type 8, no shift) and
//! `cv2.mean` with that mask.
//!
//! `fillPoly` is ported from `modules/imgproc/src/drawing.cpp` at the tag `5.0.0`
//! (https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/drawing.cpp): every edge
//! drawn as an 8-connected line (`Line` over `LineIterator`, clipped by `clipLine`), then the
//! scanline fill of `FillEdgeCollection` in 16.16 fixed point with its active edge list and
//! bubble sort. `cv2.mean` sums the masked pixels in `double` in raster order and multiplies by
//! the reciprocal of their count (Intel IPP's `ippiMean_32f_C1MR` through OpenCV's HAL, matched
//! against `cv2.mean` on random masks; 0 for an empty mask).

const XY_SHIFT: u32 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

/// A one-channel mask being drawn.
struct Mask {
    data: Vec<u8>,
    w: i64,
    h: i64,
}

impl Mask {
    fn set(&mut self, x: i64, y: i64) {
        self.data[(y * self.w + x) as usize] = 1;
    }
}

/// `clipLine(Size2l, Point2l&, Point2l&)`: whether any of the line is inside.
fn clip_line(w: i64, h: i64, p1: &mut [i64; 2], p2: &mut [i64; 2]) -> bool {
    let (right, bottom) = (w - 1, h - 1);
    if w <= 0 || h <= 0 {
        return false;
    }
    let code = |p: &[i64; 2]| {
        i32::from(p[0] < 0)
            + i32::from(p[0] > right) * 2
            + i32::from(p[1] < 0) * 4
            + i32::from(p[1] > bottom) * 8
    };
    let (mut c1, mut c2) = (code(p1), code(p2));
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            p1[0] += ((a - p1[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p1[1] = a;
            c1 = i32::from(p1[0] < 0) + i32::from(p1[0] > right) * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            p2[0] += ((a - p2[1]) as f64 * (p2[0] - p1[0]) as f64 / (p2[1] - p1[1]) as f64) as i64;
            p2[1] = a;
            c2 = i32::from(p2[0] < 0) + i32::from(p2[0] > right) * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                p1[1] +=
                    ((a - p1[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p1[0] = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                p2[1] +=
                    ((a - p2[0]) as f64 * (p2[1] - p1[1]) as f64 / (p2[0] - p1[0]) as f64) as i64;
                p2[0] = a;
                c2 = 0;
            }
        }
    }
    (c1 | c2) == 0
}

/// `Line(img, pt1, pt2, color, 8)`: `LineIterator` from left to right, 8-connected.
fn line(m: &mut Mask, a: [i64; 2], b: [i64; 2]) {
    let (mut p1, mut p2) = (a, b);
    let outside = |p: &[i64; 2]| p[0] < 0 || p[0] >= m.w || p[1] < 0 || p[1] >= m.h;
    if (outside(&p1) || outside(&p2)) && !clip_line(m.w, m.h, &mut p1, &mut p2) {
        return;
    }
    let (mut dx, mut dy) = (p2[0] - p1[0], p2[1] - p1[1]);
    let (mut sx, mut sy) = (1i64, 1i64);
    if dx < 0 {
        dx = -dx;
        dy = -dy;
        p1 = p2;
    }
    if dy < 0 {
        dy = -dy;
        sy = -1;
    }
    let vert = dy > dx;
    if vert {
        std::mem::swap(&mut dx, &mut dy);
        std::mem::swap(&mut sx, &mut sy);
    }
    let mut err = dx - (dy + dy);
    let (plus_delta, minus_delta) = (dx + dx, -(dy + dy));
    // (x, y) steps when the error is negative (plus) and always (minus)
    let (plus, minus) = if vert {
        ([sy, 0], [0, sx])
    } else {
        ([0, sy], [sx, 0])
    };
    let mut p = p1;
    for _ in 0..=dx {
        m.set(p[0], p[1]);
        let neg = err < 0;
        err += minus_delta + if neg { plus_delta } else { 0 };
        p[0] += minus[0] + if neg { plus[0] } else { 0 };
        p[1] += minus[1] + if neg { plus[1] } else { 0 };
    }
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    y0: i64,
    y1: i64,
    x: i64,
    dx: i64,
    next: Option<usize>,
}

/// `CollectPolyEdges` with no shift and no offset: the outline drawn, the edges collected.
fn collect_edges(m: &mut Mask, v: &[[i64; 2]]) -> Vec<Edge> {
    let mut edges = Vec::with_capacity(v.len() + 2);
    let mut pt0 = v[v.len() - 1];
    for &pt1 in v {
        let (mut t0, mut t1) = (pt0, pt1);
        line(m, t0, t1);
        let (mut c0y, mut c1y) = (pt0[1], pt1[1]);
        let off = |t: &[i64; 2]| t[0] < 0 || t[0] >= m.w || t[1] < 0 || t[1] >= m.h;
        if off(&t0) || off(&t1) {
            clip_line(m.w, m.h, &mut t0, &mut t1);
            if t0[1] != t1[1] {
                (c0y, c1y) = (t0[1], t1[1]);
            }
        }
        let (c0x, c1x) = (t0[0] << XY_SHIFT, t1[0] << XY_SHIFT);
        if pt0[1] != pt1[1] {
            let dx = (c1x - c0x) / (c1y - c0y);
            edges.push(if pt0[1] < pt1[1] {
                Edge {
                    y0: pt0[1],
                    y1: pt1[1],
                    x: c0x + (pt0[1] - c0y) * dx,
                    dx,
                    next: None,
                }
            } else {
                Edge {
                    y0: pt1[1],
                    y1: pt0[1],
                    x: c1x + (pt1[1] - c1y) * dx,
                    dx,
                    next: None,
                }
            });
        }
        pt0 = pt1;
    }
    edges
}

/// `FillEdgeCollection` for a one-byte colour of 1.
fn fill_edges(m: &mut Mask, mut edges: Vec<Edge>) {
    let total = edges.len();
    if total < 2 {
        return;
    }
    let (mut y_min, mut y_max) = (i64::MAX, i64::MIN);
    let (mut x_min, mut x_max) = (i64::MAX, -1i64);
    for e in &edges {
        let x1 = e.x + (e.y1 - e.y0) * e.dx;
        y_min = y_min.min(e.y0);
        y_max = y_max.max(e.y1);
        x_min = x_min.min(e.x).min(x1);
        x_max = x_max.max(e.x).max(x1);
    }
    if y_max < 0 || y_min >= m.h || x_max < 0 || x_min >= m.w << XY_SHIFT {
        return;
    }
    edges.sort_by_key(|a| (a.y0, a.x, a.dx));
    // the end marker, then the head of the active list
    let sentinel = Edge {
        y0: i64::MAX,
        y1: 0,
        x: 0,
        dx: 0,
        next: None,
    };
    edges.push(sentinel);
    edges.push(sentinel);
    let head = total + 1;
    let mut i = 0usize;
    let y_max = y_max.min(m.h);
    let mut y = edges[0].y0;
    while y < y_max {
        scan_row(m, &mut edges, head, &mut i, total, y);
        sort_active(&mut edges, head);
        y += 1;
    }
}

/// One row of `FillEdgeCollection`: retire, insert and draw between pairs of active edges.
fn scan_row(m: &mut Mask, e: &mut [Edge], head: usize, i: &mut usize, total: usize, y: i64) {
    let delta = XY_ONE - 1;
    let mut draw = false;
    let mut prelast = head;
    let mut last = e[head].next;
    while last.is_some() || e[*i].y0 == y {
        if let Some(l) = last.filter(|&l| e[l].y1 == y) {
            e[prelast].next = e[l].next;
            last = e[l].next;
            continue;
        }
        let keep_prelast = prelast;
        match last {
            Some(l) if e[*i].y0 > y || e[l].x < e[*i].x => {
                prelast = l;
                last = e[l].next;
            }
            _ if *i < total => {
                e[prelast].next = Some(*i);
                e[*i].next = last;
                prelast = *i;
                *i += 1;
            }
            _ => break,
        }
        if draw {
            if y >= 0 {
                let (a, b) = (e[keep_prelast].x, e[prelast].x);
                let (x1, x2) = if a > b {
                    ((b + delta) >> XY_SHIFT, a >> XY_SHIFT)
                } else {
                    ((a + delta) >> XY_SHIFT, b >> XY_SHIFT)
                };
                if x1 < m.w && x2 >= 0 {
                    for x in x1.max(0)..=x2.min(m.w - 1) {
                        m.set(x, y);
                    }
                }
            }
            e[keep_prelast].x += e[keep_prelast].dx;
            e[prelast].x += e[prelast].dx;
        }
        draw = !draw;
    }
}

/// The bubble sort of the active list by `x`.
fn sort_active(e: &mut [Edge], head: usize) {
    let mut keep_prelast: Option<usize> = None;
    loop {
        let mut prelast = head;
        let mut last = e[head].next;
        let mut last_exchange: Option<usize> = None;
        while let Some(l) = last {
            if Some(l) == keep_prelast {
                break;
            }
            let Some(te) = e[l].next else { break };
            if e[l].x > e[te].x {
                e[prelast].next = Some(te);
                e[l].next = e[te].next;
                e[te].next = Some(l);
                prelast = te;
                last_exchange = Some(prelast);
            } else {
                prelast = l;
                last = Some(te);
            }
        }
        let Some(exchanged) = last_exchange else {
            break;
        };
        keep_prelast = Some(exchanged);
        if Some(exchanged) == e[head].next || exchanged == head {
            break;
        }
    }
}

/// `cv2.fillPoly(mask, [poly], 1)` into a zeroed `w x h` mask.
pub fn fill_poly(poly: &[[i64; 2]], w: usize, h: usize) -> Vec<u8> {
    let mut m = Mask {
        data: vec![0; w * h],
        w: w as i64,
        h: h as i64,
    };
    if !poly.is_empty() {
        let edges = collect_edges(&mut m, poly);
        fill_edges(&mut m, edges);
    }
    m.data
}

/// `_score`: the mean of `pred` (`w x h`) inside the quadrilateral `quad` (map pixels).
pub fn box_score(pred: &[f32], w: usize, h: usize, quad: &[[f32; 2]; 4]) -> f64 {
    let (wi, hi) = (w as i64, h as i64);
    let lo = |v: f32, n: i64| (v.floor() as i64).min(n - 1).max(0);
    let up = |v: f32, n: i64| (v.ceil() as i64).min(n - 1).max(0);
    let xs = quad.map(|p| p[0]);
    let ys = quad.map(|p| p[1]);
    let (min_of, max_of) = (
        |a: [f32; 4]| a.into_iter().fold(f32::INFINITY, f32::min),
        |a: [f32; 4]| a.into_iter().fold(f32::NEG_INFINITY, f32::max),
    );
    let (xmin, xmax) = (lo(min_of(xs), wi), up(max_of(xs), wi));
    let (ymin, ymax) = (lo(min_of(ys), hi), up(max_of(ys), hi));
    let (mw, mh) = ((xmax - xmin + 1) as usize, (ymax - ymin + 1) as usize);
    // float32 minus an integer stays float32, then `astype(np.int32)` truncates
    let poly: Vec<[i64; 2]> = quad
        .iter()
        .map(|p| [(p[0] - xmin as f32) as i64, (p[1] - ymin as f32) as i64])
        .collect();
    let mask = fill_poly(&poly, mw, mh);
    let (mut sum, mut count) = (0f64, 0u64);
    for (r, row) in mask.chunks_exact(mw).enumerate() {
        let base = (ymin as usize + r) * w + xmin as usize;
        for (c, on) in row.iter().enumerate() {
            if *on != 0 {
                sum += f64::from(pred[base + c]);
                count += 1;
            }
        }
    }
    // Intel IPP, which OpenCV's build calls for the masked mean, multiplies by the reciprocal
    if count == 0 {
        0.0
    } else {
        sum * (1.0 / count as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(m: &[u8], w: usize) -> Vec<String> {
        m.chunks(w)
            .map(|r| r.iter().map(|v| if *v != 0 { '#' } else { '.' }).collect())
            .collect()
    }

    #[test]
    fn a_diamond_is_filled_as_opencv_fills_it() {
        // cv2.fillPoly(np.zeros((7, 7), np.uint8), [np.array([[3, 0], [6, 3], [3, 6], [0, 3]])], 1)
        let m = fill_poly(&[[3, 0], [6, 3], [3, 6], [0, 3]], 7, 7);
        assert_eq!(
            rows(&m, 7),
            [
                "...#...", "..###..", ".#####.", "#######", ".#####.", "..###..", "...#..."
            ]
        );
    }

    #[test]
    fn a_polygon_past_the_mask_is_clipped_and_the_score_is_the_masked_mean() {
        let m = fill_poly(&[[-2, -2], [3, -2], [3, 1], [-2, 1]], 3, 3);
        assert_eq!(rows(&m, 3), ["###", "###", "..."]);
        let pred = [0.5f32, 1.0, 0.0, 0.25];
        let s = box_score(
            &pred,
            2,
            2,
            &[[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
        );
        assert_eq!(s, 0.75);
    }
}
