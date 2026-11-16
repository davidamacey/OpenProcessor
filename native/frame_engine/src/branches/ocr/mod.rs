//! `ocr` (`frame_engine/branches/ocr.py`): text in a photo, read and stored so photos can be found
//! by it. The plan-level arithmetic of the branch, each step a function the photo flow calls:
//!
//! 1. Round 1, [`det_canvas`]: the frame resized as PaddleX `DetResizeForTest` does
//!    (`lines::det_size`), written as uint8 BGR (the planes in reverse order) by the shared
//!    resampler; the model normalises and answers the probability map `[1, 1, H, W]`.
//! 2. Settle, [`settle`] (host work): DB boxes (`db`), lines too small to cut dropped, past
//!    `max_lines` the lowest detector scores dropped and counted in `ocr_overflow`, the rest in
//!    reading order.
//! 3. Round 2, [`line_batches`]: PaddleOCR's batches by width over height, each line warped from
//!    the full-resolution frame by `kernels/ocr_warp.cu` straight into its slot (the warp the
//!    launch pool runs, [`LineWarp`]).
//! 4. Finish, [`outputs`]: the recogniser's `indices` and `scores` `[B, T]` collapsed (`ctc`),
//!    every line back in reading order.
//!
//! A photo with no text costs the detector call and nothing else.

pub mod contours;
pub mod ctc;
pub mod db;
pub mod fill;
pub mod lines;
pub mod min_rect;
pub mod offset;
pub mod warp;

use bytes::Bytes;
use gpu_frame::OutputTensor;

use self::ctc::CtcDecoder;
use self::db::{DbParams, TextBox, boxes_from_map, sort_reading_order};
use self::lines::{REC_HEIGHT, det_size, line_map, plan_batches, rec_ratio, resized_width};
use super::{InputSpec, fp32_output, int32_output};
use crate::kernels::{Filter, Resample};
use crate::pipeline::Tensor;
use crate::wire::plan::OcrSpec;

/// The text detector's input tensor.
pub const DET_INPUT: &str = "x";
/// The text detector's probability map.
pub const DET_OUTPUT: &str = "fetch_name_0";
/// The text detector's outputs read back.
pub const DET_OUTPUTS: [&str; 1] = [DET_OUTPUT];
/// The recogniser's input tensor.
pub const REC_INPUT: &str = "x";
/// The recogniser's best class per step (INT32 `[B, T]`).
pub const REC_INDICES: &str = "indices";
/// The recogniser's probability of that class (FP32 `[B, T]`).
pub const REC_SCORES: &str = "scores";
/// The recogniser's outputs read back.
pub const REC_OUTPUTS: [&str; 2] = [REC_INDICES, REC_SCORES];
/// The widest recogniser input of the build's optimisation profile (`export_paddleocr_rec.py`).
pub const REC_MAX_WIDTH: i64 = 2048;
/// Written right of a line's resized width (`ocr/kernels.py` `PAD_VALUE`).
pub const PAD_VALUE: u8 = 128;

/// The detector's input for a frame of `w x h`: one resample into BGR planes, so the first
/// plane written (red) is the last of the tensor.
///
/// # Errors
/// An unknown `det_limit`, or a size past the address range.
pub fn det_canvas(spec: &OcrSpec, w: u32, h: u32) -> Result<(InputSpec, Resample), String> {
    let (rh, rw) = det_size(
        i64::from(h),
        i64::from(w),
        i64::from(spec.det_side),
        &spec.det_limit,
        i64::from(spec.det_max_side),
    )
    .ok_or_else(|| format!("unknown det_limit {:?}", spec.det_limit))?;
    let plane = rh * rw;
    let dst_offset = usize::try_from(2 * plane).map_err(|e| e.to_string())?;
    Ok((
        InputSpec {
            name: DET_INPUT,
            shape: [1, 3, rh, rw],
        },
        Resample {
            roi: [0, 0, i64::from(w), i64::from(h)],
            size: [rw, rh],
            window: [0, 0, rw, rh],
            dst_offset,
            plane: -plane,
            row_stride: rw,
            filter: if spec.det_antialias {
                Filter::Antialias
            } else {
                Filter::Bilinear
            },
        },
    ))
}

/// The lines the photo keeps, in reading order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Settled {
    /// The kept lines.
    pub boxes: Vec<TextBox>,
    /// Lines dropped past `max_lines` (`ocr_overflow`).
    pub overflow: i32,
}

/// `Ocr.settle`: the boxes of the detector's answer for a frame of `w x h`.
///
/// # Errors
/// A detector answer without the map or of another shape than `[1, 1, H, W]`.
pub fn settle(answer: &[OutputTensor], spec: &OcrSpec, w: u32, h: u32) -> Result<Settled, String> {
    let (shape, pred) = fp32_output(answer, DET_OUTPUT)?;
    let [_, _, mh, mw] = *shape else {
        return Err(format!(
            "{DET_OUTPUT} has shape {shape:?}, not [1, 1, H, W]"
        ));
    };
    let (mh, mw) = (
        usize::try_from(mh).map_err(|e| e.to_string())?,
        usize::try_from(mw).map_err(|e| e.to_string())?,
    );
    if mh * mw != pred.len() || mh == 0 || mw == 0 {
        return Err(format!("{DET_OUTPUT} has shape {shape:?}"));
    }
    let params = DbParams {
        thresh: spec.thresh,
        box_thresh: spec.box_thresh,
        unclip_ratio: spec.unclip_ratio,
    };
    let mut boxes: Vec<TextBox> = boxes_from_map(&pred, mw, mh, (w, h), params)
        .into_iter()
        .filter(|b| rec_ratio(&b.quad).is_some())
        .collect();
    let max = usize::try_from(spec.max_lines).unwrap_or(usize::MAX);
    let mut overflow = 0;
    if boxes.len() > max {
        overflow = i32::try_from(boxes.len() - max).unwrap_or(i32::MAX);
        // `np.sort(np.argsort(-scores, kind='stable')[:max_lines])`
        let mut top: Vec<usize> = (0..boxes.len()).collect();
        top.sort_by(|a, b| boxes[*b].score.total_cmp(&boxes[*a].score));
        top.truncate(max);
        top.sort_unstable();
        boxes = top.into_iter().map(|i| boxes[i]).collect();
    }
    let quads: Vec<[[i32; 2]; 4]> = boxes.iter().map(|b| b.quad).collect();
    let boxes = sort_reading_order(&quads)
        .into_iter()
        .map(|i| boxes[i])
        .collect();
    Ok(Settled { boxes, overflow })
}

/// What the warp kernel writes for one recogniser call (`LineWarp.warp_into`'s arguments): one
/// row of `maps` (9 floats), `widths` and `taps` (2) per line, into a `[n, 3, 48, width]` slice.
#[derive(Debug, Clone, PartialEq)]
pub struct LineWarp {
    /// One float32 row of 9 per line (`line_map`).
    pub maps: Vec<f32>,
    /// Each line's resized width.
    pub widths: Vec<i32>,
    /// Each line's samples per axis (2 per line).
    pub taps: Vec<i32>,
    /// The call's padded width.
    pub batch_width: i64,
    /// Written right of each line's width.
    pub pad: u8,
}

impl LineWarp {
    /// Lines in the call.
    pub fn lines(&self) -> usize {
        self.widths.len()
    }
}

/// One recogniser call: the lines it reads (indices into the settled lines) and their warp.
#[derive(Debug, Clone, PartialEq)]
pub struct LineBatch {
    /// Indices into the settled lines.
    pub indices: Vec<usize>,
    /// The recogniser input of the call.
    pub input: InputSpec,
    /// What the warp kernel writes into it.
    pub warp: LineWarp,
}

/// `Ocr.round2`'s batches for the settled lines.
///
/// # Errors
/// A line whose perspective map is singular.
pub fn line_batches(settled: &Settled, spec: &OcrSpec) -> Result<Vec<LineBatch>, String> {
    let ratios: Vec<f64> = settled
        .boxes
        .iter()
        .map(|b| rec_ratio(&b.quad).unwrap_or(0.0))
        .collect();
    let batch = usize::try_from(spec.batch).unwrap_or(1);
    plan_batches(&ratios, batch, REC_MAX_WIDTH)
        .into_iter()
        .map(|b| {
            let n = b.indices.len();
            let mut warp = LineWarp {
                maps: Vec::with_capacity(n * 9),
                widths: Vec::with_capacity(n),
                taps: Vec::with_capacity(n * 2),
                batch_width: b.width,
                pad: PAD_VALUE,
            };
            for &i in &b.indices {
                let width = resized_width(ratios[i], b.width);
                let m = line_map(&settled.boxes[i].quad, width, spec.taps)
                    .ok_or_else(|| format!("text line {i} has a singular perspective map"))?;
                warp.maps.extend(m.map);
                warp.widths
                    .push(i32::try_from(width).map_err(|e| e.to_string())?);
                warp.taps.extend(m.taps);
            }
            Ok(LineBatch {
                input: InputSpec {
                    name: REC_INPUT,
                    shape: [n as i64, 3, REC_HEIGHT, b.width],
                },
                indices: b.indices,
                warp,
            })
        })
        .collect()
}

/// `Ocr.finish`: the lines of every call decoded, the six outputs in the wire's names.
///
/// # Errors
/// A recogniser answer without its outputs, of a shape that does not match its lines, or an
/// index outside the dictionary.
pub fn outputs(
    settled: &Settled,
    batches: &[(Vec<usize>, Vec<OutputTensor>)],
    decoder: &CtcDecoder,
) -> Result<Vec<(&'static str, Tensor)>, String> {
    let n = settled.boxes.len();
    let mut texts = vec![Bytes::new(); n];
    let mut conf = vec![0f32; n];
    let mut low = vec![0f32; n];
    for (indices, answer) in batches {
        let (shape, idx) = int32_output(answer, REC_INDICES)?;
        let (_, scores) = fp32_output(answer, REC_SCORES)?;
        let steps = usize::try_from(*shape.get(1).unwrap_or(&0)).map_err(|e| e.to_string())?;
        if scores.len() != idx.len() || idx.len() != indices.len() * steps {
            return Err(format!(
                "recogniser answer of shape {shape:?} for {} lines",
                indices.len()
            ));
        }
        for (k, &i) in indices.iter().enumerate() {
            let row = k * steps..(k + 1) * steps;
            let line = decoder.decode(&idx[row.clone()], &scores[row])?;
            texts[i] = Bytes::from(line.text.into_bytes());
            conf[i] = line.mean;
            low[i] = line.lowest;
        }
    }
    let rows = n as i64;
    let fp32 = |shape: Vec<i64>, data: Vec<f32>| Tensor::Fp32 { shape, data };
    Ok(vec![
        (
            "ocr_quads",
            fp32(
                vec![rows, 8],
                settled
                    .boxes
                    .iter()
                    .flat_map(|b| b.quad.iter().flatten().map(|v| *v as f32))
                    .collect(),
            ),
        ),
        ("ocr_texts", Tensor::Bytes(texts)),
        ("ocr_scores", fp32(vec![rows], conf)),
        ("ocr_char_min", fp32(vec![rows], low)),
        (
            "ocr_det_scores",
            fp32(vec![rows], settled.boxes.iter().map(|b| b.score).collect()),
        ),
        (
            "ocr_overflow",
            Tensor::Int32 {
                shape: vec![1],
                data: vec![settled.overflow],
            },
        ),
    ])
}

#[cfg(test)]
mod tests;
