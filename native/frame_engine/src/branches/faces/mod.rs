//! `faces` (round 1, settle on the finish pool) and `face_embed` (round 2)
//! (`frame_engine/branches/faces.py`): SCRFD on passes cut from the frame, then ArcFace on
//! five-point aligned faces sampled from the full-resolution frame.
//!
//! `faces`: every pass of the photo ([`passes::plan_passes`]: the whole frame in the detector's
//! input, the optional coarse square and tiles) is resampled from the frame, plain bilinear as
//! the reference's `cv2.resize`, into its own zero-filled input at the top-left; the passes go to
//! the detector in calls of [`PASSES_PER_CALL`]. The settle decodes each pass (anchors, the score
//! threshold), maps it to source pixels and runs one NMS over every pass, the `max_faces` cap
//! counted in `face_overflow`. The face blur sums are `blur_sums`' arithmetic over each face box.
//!
//! `face_embed`: each face's similarity to the ArcFace template, its crop warped from the frame
//! into its slot of the recogniser batch (calls of the plan's `batch` faces), the vectors
//! L2-normalised. A face whose landmarks give no transform is not embedded (`face_embedded`); a
//! photo with no face makes no call.
//!
//! Both models are the uint8-input builds (input `images`, the normalisation in the graph).

pub mod align;
pub mod embed;
pub mod passes;
pub mod scrfd;
pub mod warp;

use gpu_frame::OutputTensor;

use self::passes::DetPass;
use self::scrfd::{Dets, Faces, StrideOut};
use super::InputSpec;
use crate::geometry::blur_rect;
use crate::kernels::{Filter, Resample};
use crate::pipeline::Tensor;
use crate::wire::plan::FaceSpec;

/// Both models' input name, and the recogniser's output.
pub const INPUT: &str = "images";
pub const OUTPUT: &str = "output";
/// Detector inputs per call: a 20 MP photo with tiles is a few tens of passes, never one call.
pub const PASSES_PER_CALL: usize = 8;

/// One detector call: its input, the passes it holds (in batch order) and their resamples.
#[derive(Debug, Clone, PartialEq)]
pub struct DetCall {
    pub input: InputSpec,
    pub passes: Vec<DetPass>,
    pub resamples: Vec<Resample>,
}

/// `Faces.round1`'s cut for a frame of `w x h`: the passes in calls of [`PASSES_PER_CALL`], each
/// pass's window at the top-left of its zero-filled `net x net` input.
pub fn detector_calls(spec: &FaceSpec, w: u32, h: u32) -> Result<Vec<DetCall>, String> {
    let net = i64::from(spec.input_size);
    let all = passes::plan_passes(
        i64::from(w),
        i64::from(h),
        net,
        i64::from(spec.coarse),
        i64::from(spec.tile_side),
        i64::from(spec.overlap),
    )
    .ok_or_else(|| format!("no face passes for a {w}x{h} frame"))?;
    let square = usize::try_from(3 * net * net).map_err(|_| "a face input past usize")?;
    Ok(all
        .chunks(PASSES_PER_CALL)
        .map(|chunk| DetCall {
            input: InputSpec {
                name: INPUT,
                shape: [chunk.len() as i64, 3, net, net],
            },
            resamples: chunk
                .iter()
                .enumerate()
                .map(|(i, p)| Resample {
                    roi: [0, 0, i64::from(w), i64::from(h)],
                    size: p.scaled,
                    window: p.window,
                    dst_offset: i * square,
                    plane: net * net,
                    row_stride: net,
                    filter: Filter::Bilinear,
                })
                .collect(),
            passes: chunk.to_vec(),
        })
        .collect())
}

/// Batch element `i` of a `[batch, ...]` float32 output, checked to split evenly.
fn element<'a>(shape: &[i64], values: &'a [f32], i: usize) -> Result<&'a [f32], String> {
    let batch = shape
        .first()
        .and_then(|b| usize::try_from(*b).ok())
        .filter(|b| *b > 0 && values.len().is_multiple_of(*b))
        .ok_or_else(|| format!("a face detector output of shape {shape:?}"))?;
    let per = values.len() / batch;
    values
        .get(i * per..(i + 1) * per)
        .ok_or_else(|| format!("no batch element {i} in shape {shape:?}"))
}

/// One float32 output: its shape and values.
type Fp32Output = (Vec<i64>, Vec<f32>);

/// One call's nine outputs, in [`scrfd::OUTPUT_NAMES`] order.
fn call_outputs(outputs: &[OutputTensor]) -> Result<Vec<Fp32Output>, String> {
    scrfd::OUTPUT_NAMES
        .iter()
        .map(|name| super::fp32_output(outputs, name).map(|(s, v)| (s.to_vec(), v)))
        .collect()
}

/// `Faces.settle`: every pass of every call decoded at `score_min`, mapped to source pixels,
/// merged by one NMS at `nms` and capped at `max_faces`. `answers` holds one answer per call of
/// `calls`, in order. Runs on the finish pool.
pub fn settle(
    spec: &FaceSpec,
    w: u32,
    h: u32,
    calls: &[DetCall],
    answers: &[Vec<OutputTensor>],
) -> Result<Faces, String> {
    if calls.len() != answers.len() {
        return Err(format!(
            "{} detector answers for {} calls",
            answers.len(),
            calls.len()
        ));
    }
    let net = spec.input_size as usize;
    let threshold = spec.score_min as f32;
    let mut parts = Vec::new();
    for (call, answer) in calls.iter().zip(answers) {
        let outs = call_outputs(answer)?;
        for (i, p) in call.passes.iter().enumerate() {
            let part = |k: usize| element(&outs[k].0, &outs[k].1, i);
            let strides = [
                StrideOut {
                    scores: part(0)?,
                    boxes: part(3)?,
                    kps: part(6)?,
                },
                StrideOut {
                    scores: part(1)?,
                    boxes: part(4)?,
                    kps: part(7)?,
                },
                StrideOut {
                    scores: part(2)?,
                    boxes: part(5)?,
                    kps: part(8)?,
                },
            ];
            let found = scrfd::decode(&strides, net, net, threshold)?;
            parts.push(scrfd::to_source(p, i64::from(w), i64::from(h), &found));
        }
    }
    Ok(scrfd::merge(
        &parts,
        spec.nms as f32,
        spec.max_faces as usize,
    ))
}

/// The settle's tensors: `face_boxes [n, 4]`, `face_scores [n]`, `face_landmarks [n, 10]` in
/// source pixels, best score first, and `face_overflow [1]`.
pub fn outputs(f: &Faces) -> Vec<(&'static str, Tensor)> {
    let d: &Dets = &f.dets;
    let n = d.len() as i64;
    vec![
        (
            "face_boxes",
            Tensor::Fp32 {
                shape: vec![n, 4],
                data: d.boxes.iter().flatten().copied().collect(),
            },
        ),
        (
            "face_scores",
            Tensor::Fp32 {
                shape: vec![n],
                data: d.scores.clone(),
            },
        ),
        (
            "face_landmarks",
            Tensor::Fp32 {
                shape: vec![n, 10],
                data: d.landmarks.iter().flatten().copied().collect(),
            },
        ),
        (
            "face_overflow",
            Tensor::Int32 {
                shape: vec![1],
                data: vec![i32::try_from(f.overflow).unwrap_or(i32::MAX)],
            },
        ),
    ]
}

/// Each face box clamped to the frame (`blur_rect`), `None` where nothing is inside; the
/// non-empty ones are what the blur kernel sums, in order.
pub fn blur_rects(f: &Faces, w: u32, h: u32) -> Vec<Option<[i64; 4]>> {
    f.dets
        .boxes
        .iter()
        .map(|b| blur_rect(b.map(f64::from), i64::from(w), i64::from(h)))
        .collect()
}

/// `face_blur_sums`, `int64 [faces, 3]`: (pixels, sum, sum of squares) of each clamped face box,
/// `(0, 0, 0)` for an empty one. `sums` holds one pair per non-empty rectangle, in order.
pub fn blur_sums(clamped: &[Option<[i64; 4]>], sums: &[[i64; 2]]) -> Result<Tensor, String> {
    let mut next = sums.iter();
    let mut data = Vec::with_capacity(3 * clamped.len());
    for r in clamped {
        data.extend(match r {
            Some([x1, y1, x2, y2]) => {
                let [s, q] = next.next().ok_or("fewer face blur sums than boxes")?;
                [(x2 - x1) * (y2 - y1), *s, *q]
            }
            None => [0, 0, 0],
        });
    }
    if next.next().is_some() {
        return Err("more face blur sums than boxes".into());
    }
    Ok(Tensor::Int64 {
        shape: vec![clamped.len() as i64, 3],
        data,
    })
}

#[cfg(test)]
mod tests;
