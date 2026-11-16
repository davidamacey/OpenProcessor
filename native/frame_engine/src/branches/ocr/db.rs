//! Text boxes from the detector's probability map: `boxes_from_map` and `sort_reading_order` of
//! `frame_engine/ocr/db_boxes.py` (PaddleX release/3.3 `DBPostProcess.boxes_from_bitmap`,
//! `box_type="quad"`), over the OpenCV and Clipper ports of this directory.

use super::contours::find_contours;
use super::fill::box_score;
use super::min_rect::mini_box;
use super::offset::unclip;

/// `MIN_SIZE`: the shorter side a box needs before its unclip (and `MIN_SIZE + 2` after).
const MIN_SIZE: f32 = 3.0;
/// `DbParams.max_candidates`.
const MAX_CANDIDATES: usize = 3000;

/// The DB post-process's settings (`DbParams`; `use_dilation` is off, as the engine runs it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DbParams {
    /// The map level a pixel must exceed to be text.
    pub thresh: f64,
    /// The mean map level a box needs inside it.
    pub box_thresh: f64,
    /// How far a box grows (area times ratio over perimeter).
    pub unclip_ratio: f64,
}

/// A text box in the frame's pixels (top-left, top-right, bottom-right, bottom-left) and its
/// detector score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextBox {
    /// The corners, top-left, top-right, bottom-right, bottom-left, in frame pixels.
    pub quad: [[i32; 2]; 4],
    /// The mean map level inside the box before its unclip.
    pub score: f32,
}

/// `boxes_from_map(pred, dest_w, dest_h, params)` for a map of `w x h` (row after row), in the
/// order OpenCV found the contours.
pub fn boxes_from_map(
    pred: &[f32],
    w: usize,
    h: usize,
    dest: (u32, u32),
    params: DbParams,
) -> Vec<TextBox> {
    // numpy compares a float32 array with a Python float in float32
    let thresh = params.thresh as f32;
    let mask: Vec<u8> = pred.iter().map(|v| u8::from(*v > thresh)).collect();
    let (dest_w, dest_h) = (f64::from(dest.0), f64::from(dest.1));
    let (sx, sy) = (dest_w / w as f64, dest_h / h as f64);
    let mut out = Vec::new();
    for contour in find_contours(&mask, w, h).iter().take(MAX_CANDIDATES) {
        let (points, side) = mini_box(contour);
        if side < MIN_SIZE {
            continue;
        }
        let score = box_score(pred, w, h, &points);
        if params.box_thresh > score {
            continue;
        }
        let Some(grown) = unclip(&points, params.unclip_ratio) else {
            continue;
        };
        let grown: Vec<[i32; 2]> = grown
            .iter()
            .map(|p| p.map(|v| i32::try_from(v).unwrap_or(i32::MAX)))
            .collect();
        let (quad, side) = mini_box(&grown);
        if side < MIN_SIZE + 2.0 {
            continue;
        }
        // `np.clip(np.round(x * s), 0, dest)` of float32 corners: numpy keeps it all in float32
        let scale = |v: f32, s: f64, max: f64| {
            (v * s as f32).round_ties_even().clamp(0.0, max as f32) as i32
        };
        out.push(TextBox {
            quad: quad.map(|p| [scale(p[0], sx, dest_w), scale(p[1], sy, dest_h)]),
            score: score as f32,
        });
    }
    out
}

/// `sort_reading_order`: indices that order the boxes by their first corner's `(y, x)`, then
/// swap neighbours within 10 rows into `x` order (`SortQuadBoxes`).
pub fn sort_reading_order(quads: &[[[i32; 2]; 4]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..quads.len()).collect();
    order.sort_by_key(|&i| (quads[i][0][1], quads[i][0][0]));
    for i in 0..quads.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            let (a, b) = (quads[order[j]][0], quads[order[j + 1]][0]);
            if (b[1] - a[1]).abs() < 10 && b[0] < a[0] {
                order.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbours_on_one_line_are_read_left_to_right() {
        let q = |x: i32, y: i32| [[x, y], [x + 5, y], [x + 5, y + 5], [x, y + 5]];
        let quads = [q(50, 12), q(10, 15), q(0, 40), q(30, 5)];
        // sort_reading_order of the same boxes, Python engine: (5) then the (12) and (15) pair
        assert_eq!(sort_reading_order(&quads), vec![3, 1, 0, 2]);
        assert!(sort_reading_order(&[]).is_empty());
    }

    #[test]
    fn an_empty_map_has_no_boxes() {
        let p = DbParams {
            thresh: 0.2,
            box_thresh: 0.45,
            unclip_ratio: 1.4,
        };
        assert!(boxes_from_map(&[0.0; 64], 8, 8, (80, 80), p).is_empty());
    }
}
