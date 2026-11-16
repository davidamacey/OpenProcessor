//! SCRFD post-processing (`frame_engine/faces/scrfd.py`: `anchor_centers`, `decode`,
//! `to_source`, `nms`, `merge`), run on the finish pool.
//!
//! The arithmetic is numpy's float32, operation for operation: per stride, anchor centres on the
//! stride grid, two anchors per centre, distances times the stride; each axis mapped back by its
//! own scale; one greedy NMS over every pass with the `+1` area convention, stable descending
//! order. A Python float meeting a float32 array (`score_min`, `nms`, the edge margin) is cast to
//! float32 first, as NumPy 2 does for a Python scalar (NEP 50).

use super::passes::DetPass;

/// The detector's strides.
pub const FPN_STRIDES: [usize; 3] = [8, 16, 32];
/// Anchors per grid centre.
pub const NUM_ANCHORS: usize = 2;
/// Output names of the batch-patched SCRFD (`export/scrfd_onnx.py`), scores, boxes and landmarks
/// per stride, in this order.
pub const OUTPUT_NAMES: [&str; 9] = [
    "score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32",
];
/// A detection this close to an inner tile edge (input pixels) is left to the neighbour.
pub const EDGE_MARGIN: f32 = 2.0;

/// Detections as rows: scores, boxes `x1, y1, x2, y2`, five landmarks `x, y` each.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dets {
    /// One score per detection.
    pub scores: Vec<f32>,
    /// One box per detection: `x1, y1, x2, y2`.
    pub boxes: Vec<[f32; 4]>,
    /// Five landmarks per detection: `x, y` each.
    pub landmarks: Vec<[f32; 10]>,
}

impl Dets {
    /// The number of detections.
    pub fn len(&self) -> usize {
        self.scores.len()
    }

    /// Whether there is no detection.
    pub fn is_empty(&self) -> bool {
        self.scores.is_empty()
    }

    fn push(&mut self, score: f32, b: [f32; 4], k: [f32; 10]) {
        self.scores.push(score);
        self.boxes.push(b);
        self.landmarks.push(k);
    }

    fn extend(&mut self, other: &Dets) {
        self.scores.extend_from_slice(&other.scores);
        self.boxes.extend_from_slice(&other.boxes);
        self.landmarks.extend_from_slice(&other.landmarks);
    }

    fn pick(&self, rows: &[usize]) -> Dets {
        let mut out = Dets::default();
        for &r in rows {
            out.push(self.scores[r], self.boxes[r], self.landmarks[r]);
        }
        out
    }
}

/// One stride's outputs for one detector input: `anchors` scores, `anchors x 4` distances,
/// `anchors x 10` landmark offsets.
#[derive(Debug, Clone, Copy)]
pub struct StrideOut<'a> {
    /// One score per anchor.
    pub scores: &'a [f32],
    /// Four distances per anchor: left, top, right, bottom, in strides.
    pub boxes: &'a [f32],
    /// Ten landmark offsets per anchor, in strides.
    pub kps: &'a [f32],
}

/// The faces of a photo, best score first, and how many past the cap were dropped.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Faces {
    /// The faces kept, best score first.
    pub dets: Dets,
    /// Faces past the cap, dropped.
    pub overflow: usize,
}

/// `decode`: one input's detections at or above `threshold`, in input pixels, stride by stride
/// and anchor by anchor. `outputs` is per stride of [`FPN_STRIDES`]; an output whose length does
/// not match the input's anchor grid is an error.
///
/// # Errors
///
/// A stride's outputs do not have one row per anchor of the input's grid.
pub fn decode(
    outputs: &[StrideOut<'_>; 3],
    net_w: usize,
    net_h: usize,
    threshold: f32,
) -> Result<Dets, String> {
    let mut out = Dets::default();
    for (stride, o) in FPN_STRIDES.iter().zip(outputs) {
        let cols = net_w / stride;
        let anchors = (net_h / stride) * cols * NUM_ANCHORS;
        if o.scores.len() != anchors || o.boxes.len() != anchors * 4 || o.kps.len() != anchors * 10
        {
            return Err(format!(
                "stride {stride}: {} scores, {} box and {} landmark values for {anchors} anchors",
                o.scores.len(),
                o.boxes.len(),
                o.kps.len()
            ));
        }
        let s = *stride as f32;
        for (a, &score) in o.scores.iter().enumerate() {
            if score < threshold || score.is_nan() {
                continue;
            }
            let centre = a / NUM_ANCHORS;
            let (cx, cy) = (
                ((centre % cols) * stride) as f32,
                ((centre / cols) * stride) as f32,
            );
            let d = &o.boxes[a * 4..a * 4 + 4];
            let b = [cx - d[0] * s, cy - d[1] * s, cx + d[2] * s, cy + d[3] * s];
            let mut k = [0f32; 10];
            for (j, v) in k.iter_mut().enumerate() {
                let c = if j % 2 == 0 { cx } else { cy };
                *v = o.kps[a * 10 + j] * s + c;
            }
            out.push(score, b, k);
        }
    }
    Ok(out)
}

/// Whether a box survives the pass's inner edges (a tile drops what one of them cuts).
fn inside_edges(p: &DetPass, b: &[f32; 4]) -> bool {
    let [il, it, ir, ib] = p.inner;
    let (ww, wh) = ((p.window[2] - 2) as f32, (p.window[3] - 2) as f32);
    (!il || b[0] > EDGE_MARGIN)
        && (!it || b[1] > EDGE_MARGIN)
        && (!ir || b[2] < ww)
        && (!ib || b[3] < wh)
}

/// `to_source`: one pass's detections in source pixels, without those an inner tile edge cuts.
pub fn to_source(p: &DetPass, w: i64, h: i64, dets: &Dets) -> Dets {
    let (sx, sy) = (
        (p.scaled[0] as f64 / w as f64) as f32,
        (p.scaled[1] as f64 / h as f64) as f32,
    );
    let (ox, oy) = (p.window[0] as f32, p.window[1] as f32);
    let mut out = Dets::default();
    for ((score, b), k) in dets.scores.iter().zip(&dets.boxes).zip(&dets.landmarks) {
        if !inside_edges(p, b) {
            continue;
        }
        let mut kk = [0f32; 10];
        for (j, v) in kk.iter_mut().enumerate() {
            *v = if j % 2 == 0 {
                (k[j] + ox) / sx
            } else {
                (k[j] + oy) / sy
            };
        }
        out.push(
            *score,
            [
                (b[0] + ox) / sx,
                (b[1] + oy) / sy,
                (b[2] + ox) / sx,
                (b[3] + oy) / sy,
            ],
            kk,
        );
    }
    out
}

/// `argsort(-scores, kind='stable')`: descending, ties in their order, NaN last.
fn descending(scores: &[f32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (-scores[a], -scores[b]);
        match (x.is_nan(), y.is_nan()) {
            (false, false) => x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal),
            (nx, ny) => nx.cmp(&ny),
        }
    });
    order
}

/// `np.minimum` and `np.maximum`: a NaN on either side gives NaN (`f32::min` would drop it).
fn np_min(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

fn np_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `nms`: the rows kept by greedy NMS, best first (the reference's `SCRFD.nms`).
pub fn nms(boxes: &[[f32; 4]], scores: &[f32], threshold: f32) -> Vec<usize> {
    let area = |b: &[f32; 4]| (b[2] - b[0] + 1.0) * (b[3] - b[1] + 1.0);
    let mut order = descending(scores);
    let mut keep = Vec::new();
    while let Some((&i, rest)) = order.split_first() {
        keep.push(i);
        let bi = &boxes[i];
        let ai = area(bi);
        order = rest
            .iter()
            .copied()
            .filter(|&r| {
                let br = &boxes[r];
                let w = np_max(0.0, np_min(bi[2], br[2]) - np_max(bi[0], br[0]) + 1.0);
                let h = np_max(0.0, np_min(bi[3], br[3]) - np_max(bi[1], br[1]) + 1.0);
                let inter = w * h;
                inter / (ai + area(br) - inter) <= threshold
            })
            .collect();
    }
    keep
}

/// `merge`: NMS over every pass's source-pixel detections; the best `cap` are kept, the rest
/// counted.
pub fn merge(parts: &[Dets], nms_threshold: f32, cap: usize) -> Faces {
    let mut all = Dets::default();
    for p in parts {
        all.extend(p);
    }
    let keep = nms(&all.boxes, &all.scores, nms_threshold);
    Faces {
        dets: all.pick(&keep[..keep.len().min(cap)]),
        overflow: keep.len().saturating_sub(cap),
    }
}
