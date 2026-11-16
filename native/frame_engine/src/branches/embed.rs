//! `embed_frame` (round 1) and `embed_crops` (round 2) (`frame_engine/branches/embed.py`):
//! encoder squares cut from the frame, each encoder a model of its own.
//!
//! The whole-frame square is the frame resampled (antialiased) with its shorter edge at the
//! encoder size, centre crop (`whole_square_geometry`). A crop square is the whole rectangle
//! scaled to fit the square with its aspect kept, on a square of the plan's fill
//! (`letterbox_geometry`), antialiased when the rectangle is larger than the square and plain
//! bilinear when it is smaller (the same filter, `filter_weights`). The squares go to the
//! encoder in calls of the plan's batch; `embed_classes` picks which detections get one.

use std::collections::BTreeSet;

use gpu_frame::OutputTensor;

use super::{InputSpec, fp32_output};
use crate::geometry::{GeometryError, letterbox_geometry, whole_square_geometry};
use crate::kernels::{Filter, Resample};
use crate::pipeline::Tensor;
use crate::wire::plan::{CropEmbedSpec, EmbedSpec};

pub const INPUT: &str = "images";
pub const OUTPUT: &str = "image_embeddings";

fn usize_of(v: i64) -> Result<usize, GeometryError> {
    usize::try_from(v).map_err(|_| GeometryError::EmptyWindow)
}

/// The frame encoder's input for a frame of `w x h`: one square, one resample.
pub fn frame_square(
    spec: &EmbedSpec,
    w: u32,
    h: u32,
) -> Result<(InputSpec, Resample), GeometryError> {
    let t = i64::from(spec.size);
    let g = whole_square_geometry(i64::from(w), i64::from(h), t)?;
    Ok((
        InputSpec {
            name: INPUT,
            shape: [1, 3, t, t],
        },
        Resample {
            roi: [0, 0, i64::from(w), i64::from(h)],
            size: [g.new_w, g.new_h],
            window: [g.left, g.top, t, t],
            dst_offset: 0,
            plane: t * t,
            row_stride: t,
            filter: Filter::Antialias,
        },
    ))
}

/// One crop encoder call: its rectangles' squares in one input, on the fill.
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub input: InputSpec,
    pub fill: u8,
    pub resamples: Vec<Resample>,
}

/// What `embed_crops` sends: which detections are picked, and the calls of their squares.
#[derive(Debug, Clone, PartialEq)]
pub struct CropPlan {
    pub picked: Vec<bool>,
    pub chunks: Vec<Chunk>,
}

/// `EmbedCrops.round2`'s cut: the picked rectangles (every one, or those whose detector class
/// is in `embed_classes`), in chunks of the plan's batch.
pub fn crop_plan(
    spec: &CropEmbedSpec,
    embed_classes: Option<&BTreeSet<u32>>,
    rects: &[[i64; 4]],
    classes: Option<&[i32]>,
) -> Result<CropPlan, GeometryError> {
    let picked: Vec<bool> = match (embed_classes, classes) {
        (Some(wanted), Some(classes)) => classes
            .iter()
            .map(|c| u32::try_from(*c).is_ok_and(|c| wanted.contains(&c)))
            .collect(),
        _ => vec![true; rects.len()],
    };
    let t = i64::from(spec.size);
    let square = usize_of(3 * t * t)?;
    let batch = usize::try_from(spec.batch.max(1)).unwrap_or(1);
    let chosen: Vec<&[i64; 4]> = rects
        .iter()
        .zip(&picked)
        .filter_map(|(r, keep)| keep.then_some(r))
        .collect();
    let mut chunks = Vec::new();
    for group in chosen.chunks(batch) {
        let mut resamples = Vec::with_capacity(group.len());
        for (k, [x1, y1, x2, y2]) in group.iter().copied().enumerate() {
            let (rw, rh) = (x2 - x1, y2 - y1);
            let g = letterbox_geometry(rw, rh, t)?;
            resamples.push(Resample {
                roi: [*x1, *y1, rw, rh],
                size: [g.new_w, g.new_h],
                window: [0, 0, g.new_w, g.new_h],
                dst_offset: k * square + usize_of(g.top * t + g.left)?,
                plane: t * t,
                row_stride: t,
                filter: Filter::Antialias,
            });
        }
        chunks.push(Chunk {
            input: InputSpec {
                name: INPUT,
                shape: [group.len() as i64, 3, t, t],
            },
            fill: u8::try_from(spec.fill).unwrap_or(u8::MAX),
            resamples,
        });
    }
    Ok(CropPlan { picked, chunks })
}

/// The vectors of one encoder answer: `[rows, dim]` float32.
pub fn vectors(outputs: &[OutputTensor]) -> Result<(usize, usize, Vec<f32>), String> {
    let (shape, values) = fp32_output(outputs, OUTPUT)?;
    match shape {
        [rows, dim] => Ok((
            usize::try_from(*rows).unwrap_or(0),
            usize::try_from(*dim).unwrap_or(0),
            values,
        )),
        other => Err(format!("{OUTPUT} has shape {other:?}, not [rows, dim]")),
    }
}

/// `whole_embedding`: the first (only) row of the frame encoder's answer.
pub fn whole_embedding(outputs: &[OutputTensor]) -> Result<Tensor, String> {
    let (rows, dim, mut data) = vectors(outputs)?;
    if rows == 0 {
        return Err(format!("{OUTPUT} has no row"));
    }
    data.truncate(dim);
    Ok(Tensor::Fp32 {
        shape: vec![dim as i64],
        data,
    })
}

/// `crop_embeddings` (`[picked, dim]`, `(0, 0)` when nothing was picked: no row tells the crop
/// encoder's width) and `crop_embedded`, one flag per detection.
pub fn crop_outputs(
    picked: &[bool],
    answers: &[Vec<OutputTensor>],
) -> Result<[(&'static str, Tensor); 2], String> {
    let mut data = Vec::new();
    let (mut rows, mut width) = (0usize, None);
    for answer in answers {
        let (n, dim, values) = vectors(answer)?;
        if width.is_some_and(|w| w != dim) {
            return Err(format!("{OUTPUT} widths differ between calls"));
        }
        width = Some(dim);
        rows += n;
        data.extend(values);
    }
    let dim = if rows == 0 { 0 } else { width.unwrap_or(0) };
    Ok([
        (
            "crop_embeddings",
            Tensor::Fp32 {
                shape: vec![rows as i64, dim as i64],
                data,
            },
        ),
        (
            "crop_embedded",
            Tensor::Bool {
                shape: vec![picked.len() as i64],
                data: picked.to_vec(),
            },
        ),
    ])
}

#[cfg(test)]
#[path = "embed_tests.rs"]
mod tests;
