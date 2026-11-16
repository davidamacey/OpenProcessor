//! `detect` (`frame_engine/branches/detect.py`): the detector letterbox (round 1), its boxes
//! (settle), the blur sums (finish).
//!
//! The letterbox is the frame resampled (antialiased) into the canvas at the letterbox geometry,
//! on the model's fill. The detector is a fused end-to-end model: one tensor `[batch, 300, 6]`,
//! rows `x1, y1, x2, y2, score, class` in letterboxed input pixels, zero-padded past the last
//! detection; a row counts when its score is above 0. Each detection's rectangle
//! (`crop_rect(undo_letterbox(box))`) is what the crop branches cut. The blur sums are the
//! frame's and every detection's clamped rectangle (`blur_rect`): pixels, sum and sum of squares
//! of OpenCV's Laplacian on the grey plane.

use gpu_frame::OutputTensor;

use super::{InputSpec, fp32_output};
use crate::geometry::{
    GeometryError, Letterbox, blur_rect, crop_rect, letterbox_geometry, undo_letterbox,
};
use crate::kernels::{Filter, Resample};
use crate::pipeline::Tensor;
use crate::wire::plan::DetectorSpec;

pub const INPUT: &str = "images";
pub const OUTPUT: &str = "output0";

/// The detector's input for a frame of `w x h`: the canvas, its fill and the one resample.
#[derive(Debug, Clone, PartialEq)]
pub struct Canvas {
    pub input: InputSpec,
    pub fill: u8,
    pub resample: Resample,
    pub letterbox: Letterbox,
}

pub fn canvas(spec: &DetectorSpec, w: u32, h: u32) -> Result<Canvas, GeometryError> {
    let net = i64::from(spec.input_size);
    let lb = letterbox_geometry(i64::from(w), i64::from(h), net)?;
    let offset = usize::try_from(lb.top * net + lb.left).map_err(|_| GeometryError::EmptyWindow)?;
    Ok(Canvas {
        input: InputSpec {
            name: INPUT,
            shape: [1, 3, net, net],
        },
        fill: u8::try_from(spec.letterbox_fill).unwrap_or(u8::MAX),
        resample: Resample {
            roi: [0, 0, i64::from(w), i64::from(h)],
            size: [lb.new_w, lb.new_h],
            window: [0, 0, lb.new_w, lb.new_h],
            dst_offset: offset,
            plane: net * net,
            row_stride: net,
            filter: Filter::Antialias,
        },
        letterbox: lb,
    })
}

/// The detector's answer as the photo keeps it.
#[derive(Debug, Clone, PartialEq)]
pub struct Detections {
    /// Letterboxed input pixels, one row per detection.
    pub boxes: Vec<[f32; 4]>,
    pub scores: Vec<f32>,
    pub classes: Vec<i32>,
    /// Each detection's rectangle in frame pixels (`x1, y1, x2, y2`, exclusive ends).
    pub rects: Vec<[i64; 4]>,
    pub letterbox: Letterbox,
}

fn frame_box(b: &[f32; 4], lb: &Letterbox) -> [f64; 4] {
    undo_letterbox(b.map(f64::from), lb.scale, (lb.pad_w, lb.pad_h))
}

/// `Detect.settle`: the rows of the first batch whose score is above 0.
pub fn settle(
    outputs: &[OutputTensor],
    lb: &Letterbox,
    w: u32,
    h: u32,
) -> Result<Detections, String> {
    let (shape, values) = fp32_output(outputs, OUTPUT)?;
    let (Some(&cols), true) = (shape.last(), shape.len() == 3) else {
        return Err(format!(
            "{OUTPUT} has shape {shape:?}, not [batch, rows, 6]"
        ));
    };
    if cols != 6 {
        return Err(format!("{OUTPUT} rows have {cols} values, not 6"));
    }
    let per_batch = usize::try_from(shape[1]).unwrap_or(0) * 6;
    let mut d = Detections {
        boxes: Vec::new(),
        scores: Vec::new(),
        classes: Vec::new(),
        rects: Vec::new(),
        letterbox: *lb,
    };
    let (rows, _) = values[..per_batch.min(values.len())].as_chunks::<6>();
    for row in rows.iter().filter(|r| r[4] > 0.0) {
        let b = [row[0], row[1], row[2], row[3]];
        d.rects
            .push(crop_rect(frame_box(&b, lb), i64::from(w), i64::from(h)));
        d.boxes.push(b);
        d.scores.push(row[4]);
        // numpy's astype(np.int32) truncates toward zero
        d.classes.push(row[5] as i32);
    }
    Ok(d)
}

/// The rectangles the blur kernel sums: the frame, then each detection's clamped rectangle that
/// is not empty; and per detection, its clamped rectangle or `None`.
pub fn blur_rects(d: &Detections, w: u32, h: u32) -> (Vec<[i64; 4]>, Vec<Option<[i64; 4]>>) {
    let (w, h) = (i64::from(w), i64::from(h));
    let clamped: Vec<Option<[i64; 4]>> = d
        .boxes
        .iter()
        .map(|b| blur_rect(frame_box(b, &d.letterbox), w, h))
        .collect();
    let mut summed = vec![[0, 0, w, h]];
    summed.extend(clamped.iter().flatten());
    (summed, clamped)
}

/// `blur_sums`, `int64 [1 + detections, 3]`: (pixels, sum, sum of squares) of the frame, then
/// of each detection, `(0, 0, 0)` for one whose rectangle is empty. `sums` holds one pair per
/// summed rectangle, in [`blur_rects`] order.
pub fn blur_sums(
    w: u32,
    h: u32,
    clamped: &[Option<[i64; 4]>],
    sums: &[[i64; 2]],
) -> Result<Tensor, String> {
    let expected = 1 + clamped.iter().flatten().count();
    if sums.len() != expected {
        return Err(format!(
            "{} blur sums for {expected} rectangles",
            sums.len()
        ));
    }
    let mut next = sums.iter();
    let mut data = Vec::with_capacity(3 * (1 + clamped.len()));
    let mut row = |pixels: i64, pair: Option<&[i64; 2]>| {
        let [s, q] = pair.copied().unwrap_or([0, 0]);
        data.extend([pixels, s, q]);
    };
    row(i64::from(w) * i64::from(h), next.next());
    for r in clamped {
        match r {
            Some([x1, y1, x2, y2]) => row((x2 - x1) * (y2 - y1), next.next()),
            None => row(0, None),
        }
    }
    Ok(Tensor::Int64 {
        shape: vec![1 + clamped.len() as i64, 3],
        data,
    })
}

/// The tensors of the settle: `letterbox`, `det_boxes`, `det_scores`, `det_classes`.
pub fn outputs(d: &Detections) -> Vec<(&'static str, Tensor)> {
    let n = d.boxes.len() as i64;
    let lb = &d.letterbox;
    vec![
        (
            "letterbox",
            Tensor::Fp64 {
                shape: vec![3],
                data: vec![lb.scale, lb.pad_w, lb.pad_h],
            },
        ),
        (
            "det_boxes",
            Tensor::Fp32 {
                shape: vec![n, 4],
                data: d.boxes.iter().flatten().copied().collect(),
            },
        ),
        (
            "det_scores",
            Tensor::Fp32 {
                shape: vec![n],
                data: d.scores.clone(),
            },
        ),
        (
            "det_classes",
            Tensor::Int32 {
                shape: vec![n],
                data: d.classes.clone(),
            },
        ),
    ]
}

#[cfg(test)]
#[path = "detect_tests.rs"]
mod tests;
