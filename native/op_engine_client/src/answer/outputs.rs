//! The tensor groups of the branches: names, datatypes and shapes are
//! `frame_engine_wire.OUTPUT_DATATYPES`, `BRANCH_OUTPUTS` and the contract's shapes.

use super::tensor::{Matrix, Outputs};
use super::{Answer, AnswerError, CropEmbeddings};
use crate::plan::{Branch, CropsMode, EnginePlan};

/// The decoded frame's size in pixels (`size`, INT32 `[2]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSize {
    /// Pixels across.
    pub width: u32,
    /// Pixels down.
    pub height: u32,
}

/// The detector's letterbox of the frame (`letterbox`, FP64 `[3]`): the scale and the pad of
/// the canvas, to put boxes back in frame pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    /// The frame's scale on the canvas.
    pub scale: f64,
    /// Horizontal pad of the canvas.
    pub pad_w: f64,
    /// Vertical pad of the canvas.
    pub pad_h: f64,
}

/// `detect`: boxes as `x1, y1, x2, y2` on the letterboxed canvas, with a score and a class each.
#[derive(Debug, Clone, PartialEq)]
pub struct Detections {
    /// One box per detection, in canvas pixels.
    pub boxes: Vec<[f32; 4]>,
    /// One score per detection.
    pub scores: Vec<f32>,
    /// One detector class id per detection.
    pub classes: Vec<i32>,
    /// How the frame sits on the canvas.
    pub letterbox: Letterbox,
    /// `[pixels, sum, sum of squares]` of the Laplacian: row 0 the whole frame, then one per box.
    pub blur_sums: Vec<[i64; 3]>,
}

/// `faces`: boxes and five landmarks each, in pixels of the oriented frame, best score first.
#[derive(Debug, Clone, PartialEq)]
pub struct Faces {
    /// `x1, y1, x2, y2` per face.
    pub boxes: Vec<[f32; 4]>,
    /// One score per face.
    pub scores: Vec<f32>,
    /// Five `x, y` pairs per face.
    pub landmarks: Vec<[f32; 10]>,
    /// `[pixels, sum, sum of squares]` per face.
    pub blur_sums: Vec<[i64; 3]>,
    /// Faces past `max_faces`, detected and dropped.
    pub overflow: i32,
}

/// `face_embed`: a vector per embedded face (a degenerate landmark set is not embedded).
#[derive(Debug, Clone, PartialEq)]
pub struct FaceEmbeddings {
    /// One row per embedded face, in face order. `0 x 0` when no call was made.
    pub vectors: Matrix,
    /// Per face, whether it was embedded.
    pub embedded: Vec<bool>,
}

/// `ocr`: one row per text line, in reading order.
#[derive(Debug, Clone, PartialEq)]
pub struct TextLines {
    /// `x, y` of the top-left, top-right, bottom-right and bottom-left corner, in pixels of the
    /// oriented frame.
    pub quads: Vec<[f32; 8]>,
    /// The text of each line.
    pub texts: Vec<String>,
    /// Mean probability of the decoded characters.
    pub scores: Vec<f32>,
    /// The lowest character probability.
    pub char_min: Vec<f32>,
    /// The detector's mean map value in the box.
    pub det_scores: Vec<f32>,
    /// Lines past `max_lines`, detected and dropped.
    pub overflow: i32,
}

const DETECT: [&str; 5] = [
    "det_boxes",
    "det_scores",
    "det_classes",
    "letterbox",
    "blur_sums",
];
const EMBED_FRAME: [&str; 1] = ["whole_embedding"];
const EMBED_CROPS: [&str; 2] = ["crop_embeddings", "crop_embedded"];
const CROPS: [&str; 1] = ["crop_jpegs"];
const FACES: [&str; 5] = [
    "face_boxes",
    "face_scores",
    "face_landmarks",
    "face_blur_sums",
    "face_overflow",
];
const FACE_EMBED: [&str; 2] = ["face_embeddings", "face_embedded"];
const OCR: [&str; 6] = [
    "ocr_quads",
    "ocr_texts",
    "ocr_scores",
    "ocr_char_min",
    "ocr_det_scores",
    "ocr_overflow",
];
const DERIVATIVES: [&str; 1] = ["derivative_jpegs"];

/// The outputs a decoded answer to `plan` carries besides the four always present, in the order of
/// `frame_engine_wire.expected_outputs` (the branches in vocabulary order, crops, renditions).
#[must_use]
pub fn expected_outputs(plan: &EnginePlan) -> Vec<&'static str> {
    let mut names: Vec<&'static str> = Vec::new();
    for branch in &plan.branches {
        names.extend(match branch {
            Branch::Detect => DETECT.as_slice(),
            Branch::EmbedFrame => EMBED_FRAME.as_slice(),
            Branch::EmbedCrops => EMBED_CROPS.as_slice(),
            Branch::Faces => FACES.as_slice(),
            Branch::FaceEmbed => FACE_EMBED.as_slice(),
            Branch::Ocr => OCR.as_slice(),
        });
    }
    if plan.crops == CropsMode::Inline {
        names.extend(CROPS);
    }
    if plan.has_derivatives() {
        names.extend(DERIVATIVES);
    }
    names
}

/// Reads a group when the engine sent it: all of its outputs, or none (a branch that failed
/// sends none); some without the others is refused.
fn group<T>(
    tensors: &Outputs<'_>,
    names: &[&'static str],
    read: impl FnOnce() -> Result<T, AnswerError>,
) -> Result<Option<T>, AnswerError> {
    let missing = names.iter().copied().find(|name| !tensors.has(name));
    match missing {
        None => read().map(Some),
        Some(_) if names.iter().all(|name| !tensors.has(name)) => Ok(None),
        Some(name) => Err(AnswerError::Missing(name)),
    }
}

fn same_length(name: &'static str, lengths: &[usize]) -> Result<(), AnswerError> {
    if lengths.windows(2).all(|pair| pair[0] == pair[1]) {
        Ok(())
    } else {
        Err(AnswerError::Shape {
            name,
            detail: format!("row counts differ: {lengths:?}"),
        })
    }
}

fn read_detections(t: &Outputs<'_>) -> Result<Detections, AnswerError> {
    let boxes = t.rows::<f32, 4>("det_boxes")?;
    let scores = t.list::<f32>("det_scores")?;
    let classes = t.list::<i32>("det_classes")?;
    same_length("det_scores", &[boxes.len(), scores.len(), classes.len()])?;
    let [scale, pad_w, pad_h] = t.fixed::<f64, 3>("letterbox")?;
    Ok(Detections {
        boxes,
        scores,
        classes,
        letterbox: Letterbox {
            scale,
            pad_w,
            pad_h,
        },
        blur_sums: t.rows::<i64, 3>("blur_sums")?,
    })
}

fn read_faces(t: &Outputs<'_>) -> Result<Faces, AnswerError> {
    let boxes = t.rows::<f32, 4>("face_boxes")?;
    let scores = t.list::<f32>("face_scores")?;
    let landmarks = t.rows::<f32, 10>("face_landmarks")?;
    let blur_sums = t.rows::<i64, 3>("face_blur_sums")?;
    same_length(
        "face_scores",
        &[boxes.len(), scores.len(), landmarks.len(), blur_sums.len()],
    )?;
    let [overflow] = t.fixed::<i32, 1>("face_overflow")?;
    Ok(Faces {
        boxes,
        scores,
        landmarks,
        blur_sums,
        overflow,
    })
}

fn read_text(t: &Outputs<'_>) -> Result<TextLines, AnswerError> {
    let quads = t.rows::<f32, 8>("ocr_quads")?;
    let texts = t.texts("ocr_texts")?;
    let scores = t.list::<f32>("ocr_scores")?;
    let char_min = t.list::<f32>("ocr_char_min")?;
    let det_scores = t.list::<f32>("ocr_det_scores")?;
    same_length(
        "ocr_texts",
        &[
            quads.len(),
            texts.len(),
            scores.len(),
            char_min.len(),
            det_scores.len(),
        ],
    )?;
    let [overflow] = t.fixed::<i32, 1>("ocr_overflow")?;
    Ok(TextLines {
        quads,
        texts,
        scores,
        char_min,
        det_scores,
        overflow,
    })
}

fn read_size(t: &Outputs<'_>) -> Result<FrameSize, AnswerError> {
    let [width, height] = t.fixed::<i32, 2>("size")?;
    match (u32::try_from(width), u32::try_from(height)) {
        (Ok(width), Ok(height)) => Ok(FrameSize { width, height }),
        _ => Err(AnswerError::Shape {
            name: "size",
            detail: format!("{width} x {height}"),
        }),
    }
}

/// Fills the answer's tensor groups from the response, for the branches `plan` asked for.
pub(super) fn read_into(
    answer: &mut Answer,
    t: &Outputs<'_>,
    plan: &EnginePlan,
) -> Result<(), AnswerError> {
    answer.size = Some(read_size(t)?);
    if plan.runs(Branch::Detect) {
        answer.detections = group(t, &DETECT, || read_detections(t))?;
    }
    if plan.runs(Branch::EmbedFrame) {
        answer.whole_embedding = group(t, &EMBED_FRAME, || t.list::<f32>("whole_embedding"))?;
    }
    if plan.runs(Branch::EmbedCrops) {
        answer.crop_embeddings = group(t, &EMBED_CROPS, || {
            Ok(CropEmbeddings {
                vectors: t.matrix("crop_embeddings")?,
                embedded: t.list::<bool>("crop_embedded")?,
            })
        })?;
    }
    if plan.crops == CropsMode::Inline {
        answer.crop_jpegs = group(t, &CROPS, || t.byte_items("crop_jpegs"))?;
    }
    if plan.runs(Branch::Faces) {
        answer.faces = group(t, &FACES, || read_faces(t))?;
    }
    if plan.runs(Branch::FaceEmbed) {
        answer.face_embeddings = group(t, &FACE_EMBED, || {
            Ok(FaceEmbeddings {
                vectors: t.matrix("face_embeddings")?,
                embedded: t.list::<bool>("face_embedded")?,
            })
        })?;
    }
    if plan.runs(Branch::Ocr) {
        answer.text = group(t, &OCR, || read_text(t))?;
    }
    if plan.has_derivatives() {
        answer.derivative_jpegs = group(t, &DERIVATIVES, || t.byte_items("derivative_jpegs"))?;
    }
    Ok(())
}

/// An answer in which the photo was decoded and every branch ran carries every output the plan
/// asked for.
pub(super) fn require_all(answer: &Answer, plan: &EnginePlan) -> Result<(), AnswerError> {
    let present = [
        (
            plan.runs(Branch::Detect),
            answer.detections.is_some(),
            DETECT[0],
        ),
        (
            plan.runs(Branch::EmbedFrame),
            answer.whole_embedding.is_some(),
            EMBED_FRAME[0],
        ),
        (
            plan.runs(Branch::EmbedCrops),
            answer.crop_embeddings.is_some(),
            EMBED_CROPS[0],
        ),
        (
            plan.crops == CropsMode::Inline,
            answer.crop_jpegs.is_some(),
            CROPS[0],
        ),
        (plan.runs(Branch::Faces), answer.faces.is_some(), FACES[0]),
        (
            plan.runs(Branch::FaceEmbed),
            answer.face_embeddings.is_some(),
            FACE_EMBED[0],
        ),
        (plan.runs(Branch::Ocr), answer.text.is_some(), OCR[0]),
        (
            plan.has_derivatives(),
            answer.derivative_jpegs.is_some(),
            DERIVATIVES[0],
        ),
    ];
    match present.iter().find(|(wanted, got, _)| *wanted && !*got) {
        Some((_, _, name)) => Err(AnswerError::Missing(name)),
        None => Ok(()),
    }
}
