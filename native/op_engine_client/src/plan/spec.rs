//! The settings objects of a plan, field for field and default for default what
//! `src/domain/frame_engine_wire.py` writes (`DetectorSpec`, `EmbedSpec`, `CropEmbedSpec`,
//! `FaceSpec`, `OcrSpec`, `RenditionSpec`). The engine checks the ranges and answers
//! `INVALID_ARGUMENT` for a plan it refuses; this crate writes what it is given.

use serde::{Deserialize, Serialize};

/// The pad value of a crop's square when the caller names none (`CROP_FILL`).
pub const CROP_FILL: u8 = 128;
/// The longest side of the OCR detector's resized frame when none is named (`OCR_MAX_DET_SIDE`).
pub const OCR_MAX_DET_SIDE: u32 = 1920;
/// The quality of a rendition when the caller names none (`DEFAULT_RENDITION_QUALITY`).
pub const DEFAULT_RENDITION_QUALITY: u8 = 85;

/// How the whole frame becomes the encoder's square (`FRAME_RECIPES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrameRecipe {
    /// Shorter side to `size`, centre crop, antialiased bilinear.
    #[default]
    WholeFrame,
}

/// How a box becomes the crop encoder's square (`CROP_RECIPES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CropRecipe {
    /// The whole box scaled to fit the square, the rest filled.
    #[default]
    Letterbox,
}

/// How the OCR detector resizes the frame (`DET_LIMITS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetLimit {
    /// Raise the shorter side to `det_side`, never shrink.
    Min,
    /// Cap the longer side at `det_side`, never enlarge.
    #[default]
    Max,
}

/// The format of a rendition (`RENDITION_FORMATS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenditionFormat {
    /// JPEG.
    #[default]
    Jpeg,
}

/// A fused end-to-end detector.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorSpec {
    /// The model server's name of the detector.
    pub model: String,
    /// The square the frame is letterboxed into.
    pub input_size: u32,
    /// The grey of the letterbox's pad.
    pub letterbox_fill: u8,
}

/// The whole-frame encoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbedSpec {
    /// The model server's name of the encoder.
    pub model: String,
    /// The square's side.
    pub size: u32,
    /// How the frame becomes the square.
    #[serde(default)]
    pub frame_recipe: FrameRecipe,
}

fn crop_fill() -> u8 {
    CROP_FILL
}

/// The crop encoder: one `size` square per box, called in chunks of `batch`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CropEmbedSpec {
    /// The model server's name of the crop encoder.
    pub model: String,
    /// The square's side.
    pub size: u32,
    /// Squares per encoder call.
    pub batch: u32,
    /// How a box becomes the square.
    #[serde(default)]
    pub crop_recipe: CropRecipe,
    /// The pad of the letterbox.
    #[serde(default = "crop_fill")]
    pub fill: u8,
}

impl CropEmbedSpec {
    /// A crop encoder with the default recipe and pad.
    #[must_use]
    pub fn new(model: impl Into<String>, size: u32, batch: u32) -> Self {
        Self {
            model: model.into(),
            size,
            batch,
            crop_recipe: CropRecipe::default(),
            fill: CROP_FILL,
        }
    }
}

/// The `faces` and `face_embed` branches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaceSpec {
    /// The model server's name of the face detector.
    pub detector: String,
    /// The detector's square.
    pub input_size: u32,
    /// The model server's name of the recogniser.
    pub recogniser: String,
    /// Aligned faces per recogniser call.
    pub batch: u32,
    /// The lowest detector score kept.
    pub score_min: f64,
    /// The non-maximum suppression threshold.
    pub nms: f64,
    /// Faces kept per photo; the rest are counted in `face_overflow`.
    pub max_faces: u32,
    /// A pass of the whole frame in this square (0: none).
    #[serde(default)]
    pub coarse: u32,
    /// Tiles of the frame scaled to this longer side (0: none).
    #[serde(default)]
    pub tile_side: u32,
    /// Input pixels the tiles share.
    #[serde(default)]
    pub overlap: u32,
}

/// The `ocr` branch. A plan read from text takes the parser's defaults for the fields after
/// `det_side` (`_ocr` of the wire module); the Python dataclass has none, so a plan built here
/// names every field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrSpec {
    /// The model server's name of the text detector.
    pub detector: String,
    /// The model server's name of the recogniser.
    pub recogniser: String,
    /// Lines per recogniser call.
    pub batch: u32,
    /// Lines kept per photo.
    pub max_lines: u32,
    /// The side `det_limit` applies to.
    pub det_side: u32,
    /// How the frame is resized.
    #[serde(default)]
    pub det_limit: DetLimit,
    /// The longest side of the resized frame.
    #[serde(default = "det_max_side")]
    pub det_max_side: u32,
    /// Whether the shrink is filtered.
    #[serde(default)]
    pub det_antialias: bool,
    /// The post-process's threshold on the probability map.
    #[serde(default = "det_thresh")]
    pub thresh: f64,
    /// The post-process's threshold on a box's score.
    #[serde(default = "det_box_thresh")]
    pub box_thresh: f64,
    /// How far a box is grown.
    #[serde(default = "det_unclip_ratio")]
    pub unclip_ratio: f64,
    /// Samples per recogniser pixel, each way.
    #[serde(default = "one")]
    pub taps: u32,
}

fn det_max_side() -> u32 {
    OCR_MAX_DET_SIDE
}

fn det_thresh() -> f64 {
    0.2
}

fn det_box_thresh() -> f64 {
    0.45
}

fn det_unclip_ratio() -> f64 {
    1.4
}

fn one() -> u32 {
    1
}

fn rendition_quality() -> u8 {
    DEFAULT_RENDITION_QUALITY
}

/// One stored rendition: the long side fits `side`, never enlarged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenditionSpec {
    /// The name the answer's `branch_status` and the caller know it by.
    pub name: String,
    /// The long side.
    pub side: u32,
    /// The JPEG quality.
    #[serde(default = "rendition_quality")]
    pub quality: u8,
    /// The format.
    #[serde(default)]
    pub format: RenditionFormat,
}
