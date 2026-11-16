//! `cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)` of OpenCV 5.0.0, the call
//! the Python engine's DB post-process makes (`frame_engine/ocr/db_boxes.py`), ported from
//! `modules/imgproc/src/contours_new.cpp` and `contours_common.hpp` at the tag `5.0.0`
//! (https://github.com/opencv/opencv/tree/5.0.0/modules/imgproc/src).
//!
//! The Python binding always asks for the hierarchy, so the call takes the Suzuki border
//! following path, not the hierarchy-free fast path the same function has for `RETR_LIST`. What
//! the post-process reads of a contour is its point list (the minimum-area rectangle's convex hull
//! depends on the order of the points) and the order of the contours (the order of the boxes), so
//! both are reproduced step for step: the one-pixel zero border OpenCV adds, the raster scan, the
//! marks it leaves in the image (`2`, and `2 | 0x80` on a right border), a point at every change of
//! direction, and the list order: every contour is a child of the root inserted first, so the
//! answer is the reverse of the order the scan found them.

/// A point of a contour, `x, y` in the mask's pixels.
pub type Point = [i32; 2];

/// The direction codes' steps, `chainCodeDeltas`.
const DELTAS: [[isize; 2]; 8] = [
    [1, 0],
    [1, -1],
    [0, -1],
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
];
/// `MAX_SIZE`: the direction search runs up to code 15 (twice round).
const MAX_SIZE: u8 = 16;
/// `MASK8_NEW`, a pixel of a traced border.
const NEW: i8 = 2;
/// `MASK8_NEW | MASK8_RIGHT`, a pixel of a border's right side.
const RIGHT: i8 = (0x80u8 | 0x02) as i8;

/// The padded image the scan marks, and its row step.
struct Scan {
    img: Vec<i8>,
    step: usize,
}

impl Scan {
    fn delta(&self, s: u8) -> isize {
        let [dx, dy] = DELTAS[usize::from(s % 8)];
        dx + dy * self.step as isize
    }

    fn at(&self, i: usize, s: u8) -> usize {
        i.wrapping_add_signed(self.delta(s))
    }

    /// `icvFetchContourEx<schar>` with `CHAIN_APPROX_SIMPLE`: the border from `start`, marking
    /// it, a point at every change of direction (`origin` is `start` in the mask's pixels).
    fn fetch(&mut self, start: usize, origin: Point, is_hole: bool) -> Vec<Point> {
        let i0 = start;
        let mut pt = origin;
        let mut out = Vec::new();
        let mut s_end: u8 = if is_hole { 0 } else { 4 };
        let mut s = s_end;
        let mut i1;
        loop {
            s = s.wrapping_sub(1) & 7;
            i1 = self.at(i0, s);
            if self.img[i1] != 0 || s == s_end {
                break;
            }
        }
        if s == s_end {
            self.img[i0] = RIGHT;
            out.push(pt);
            return out;
        }
        let mut i3 = i0;
        let mut prev_s = s ^ 4;
        loop {
            s_end = s;
            s = s.min(MAX_SIZE - 1);
            let mut i4 = i3;
            while s < MAX_SIZE - 1 {
                s += 1;
                i4 = self.at(i3, s);
                if self.img[i4] != 0 {
                    break;
                }
            }
            s &= 7;
            if u32::from(s).wrapping_sub(1) < u32::from(s_end) {
                self.img[i3] = RIGHT;
            } else if self.img[i3] == 1 {
                self.img[i3] = NEW;
            }
            if s != prev_s {
                out.push(pt);
            }
            prev_s = s;
            let [dx, dy] = DELTAS[usize::from(s)];
            pt = [pt[0] + dx as i32, pt[1] + dy as i32];
            if i4 == i0 && i3 == i1 {
                break;
            }
            i3 = i4;
            s = (s + 4) & 7;
        }
        out
    }
}

/// The contours of the non-zero pixels of `mask` (`w x h`, row after row), in OpenCV's order.
pub fn find_contours(mask: &[u8], w: usize, h: usize) -> Vec<Vec<Point>> {
    let step = w + 2;
    let mut img = vec![0i8; step * (h + 2)];
    for (y, row) in mask.chunks_exact(w.max(1)).take(h).enumerate() {
        for (x, v) in row.iter().enumerate() {
            img[(y + 1) * step + x + 1] = i8::from(*v != 0);
        }
    }
    let mut scan = Scan { img, step };
    let (width, height) = (step - 1, h + 1);
    let mut found = Vec::new();
    // `findNext` resumes one pixel right of the last contour's start with `prev` read again
    let mut y = 1;
    let mut x = 1;
    let mut prev = 0i32;
    while y < height {
        while x < width {
            let mut p = prev;
            while x < width {
                p = i32::from(scan.img[y * step + x]);
                if p != prev {
                    break;
                }
                x += 1;
            }
            if x >= width {
                break;
            }
            let outer = prev == 0 && p == 1;
            if outer || (p == 0 && prev >= 1) {
                let sx = if outer { x } else { x - 1 };
                let origin = [sx as i32 - 1, y as i32 - 1];
                found.push(scan.fetch(y * step + sx, origin, !outer));
                x += 1;
                prev = i32::from(scan.img[y * step + x - 1]);
                continue;
            }
            prev = p;
            x += 1;
        }
        y += 1;
        x = 1;
        prev = 0;
    }
    found.reverse();
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(rows: &[&str]) -> (Vec<u8>, usize, usize) {
        let w = rows[0].len();
        let data = rows
            .iter()
            .flat_map(|r| r.bytes().map(|b| u8::from(b == b'#')))
            .collect();
        (data, w, rows.len())
    }

    #[test]
    fn a_square_is_its_four_corners_from_the_top_left_going_down() {
        // cv2.findContours of a 2 x 2 block at (1, 1) in a 5 x 5 mask
        let (m, w, h) = mask(&[".....", ".##..", ".##..", ".....", "....."]);
        assert_eq!(
            find_contours(&m, w, h),
            vec![vec![[1, 1], [1, 2], [2, 2], [2, 1]]]
        );
    }

    #[test]
    fn a_single_pixel_and_a_ring_give_outer_and_hole_contours_last_found_first() {
        let (m, w, h) = mask(&["#....", ".....", "..###", "..#.#", "..###"]);
        let got = find_contours(&m, w, h);
        assert_eq!(got.len(), 3, "the pixel, the ring and its hole: {got:?}");
        assert_eq!(got[2], vec![[0, 0]], "the pixel was found first");
        assert_eq!(got[1], vec![[2, 2], [2, 4], [4, 4], [4, 2]]);
        assert_eq!(got[0], vec![[2, 3], [3, 2], [4, 3], [3, 4]]);
        assert!(find_contours(&[0; 6], 3, 2).is_empty());
    }
}
