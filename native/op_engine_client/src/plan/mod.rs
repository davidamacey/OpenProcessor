//! The plan: the JSON text of the `plan` input, built as `src/domain/frame_engine_wire.py`'s
//! `EnginePlan.to_json` builds it, and the names of the answer it asks for.
//!
//! The vocabulary (branches, fields, defaults, ranges) is that module's and
//! `contracts/json/frame_engine_v1.json`'s; the engine's parser is the judge of a plan, so this
//! crate checks nothing a plan's ranges say, it writes what it is given. What it does guarantee
//! is the shape: branches in vocabulary order without repeats, every default written, keys
//! sorted, the text as Python spells it ([`pyjson`]).
//! `contracts/json/frame_engine_plan_build_cases.json` holds, for plans made by the Python
//! builders and by parsing the parser's own cases, the text Python writes; the tests here write
//! the same text.

mod build;
pub mod pyjson;
mod spec;

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use self::build::{CropVectors, EncoderModels, IngestPlan};
pub use self::spec::{
    CROP_FILL, CropEmbedSpec, CropRecipe, DEFAULT_RENDITION_QUALITY, DetLimit, DetectorSpec,
    EmbedSpec, FaceSpec, FrameRecipe, OCR_MAX_DET_SIDE, OcrSpec, RenditionFormat, RenditionSpec,
};

/// The protocol version a plan names: `1`, written and read as the integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ProtocolVersion;

impl ProtocolVersion {
    /// The version of `frame_engine_wire.PROTOCOL_VERSION`.
    pub const NUMBER: u8 = 1;
}

impl Serialize for ProtocolVersion {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u8(Self::NUMBER)
    }
}

impl<'de> Deserialize<'de> for ProtocolVersion {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let number = u8::deserialize(deserializer)?;
        if number == Self::NUMBER {
            Ok(Self)
        } else {
            Err(serde::de::Error::custom(format!(
                "protocol version {number}, this client speaks {}",
                Self::NUMBER
            )))
        }
    }
}

/// A branch of the engine's plan vocabulary (`ENGINE_BRANCHES`), in the order the engine runs
/// and `frame_engine_v1.json` lists them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Branch {
    /// The detector.
    Detect,
    /// The whole-frame vector.
    EmbedFrame,
    /// One vector per box.
    EmbedCrops,
    /// Faces: boxes, landmarks, blur sums.
    Faces,
    /// One vector per face (needs `faces`).
    FaceEmbed,
    /// Text lines.
    Ocr,
}

impl Branch {
    /// Every branch, in vocabulary order.
    pub const ALL: [Branch; 6] = [
        Branch::Detect,
        Branch::EmbedFrame,
        Branch::EmbedCrops,
        Branch::Faces,
        Branch::FaceEmbed,
        Branch::Ocr,
    ];

    /// The name the plan, the contract and `branch_status` use.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Branch::Detect => "detect",
            Branch::EmbedFrame => "embed_frame",
            Branch::EmbedCrops => "embed_crops",
            Branch::Faces => "faces",
            Branch::FaceEmbed => "face_embed",
            Branch::Ocr => "ocr",
        }
    }
}

/// Whether the answer carries the crop JPEGs (`CropsMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CropsMode {
    /// One JPEG per box, in the answer.
    Inline,
    /// None.
    #[default]
    None,
}

/// One request's plan. Fields are public: build one with struct syntax and `..Default::default()`,
/// or through [`EnginePlan::ingest`] and [`EnginePlan::stored_boxes_embed`], the two plans the
/// Python server sends.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnginePlan {
    /// Always the protocol version this client speaks.
    #[serde(default)]
    pub version: ProtocolVersion,
    /// The branches to run, in vocabulary order by construction.
    pub branches: BTreeSet<Branch>,
    /// The detector, when `detect` runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector: Option<DetectorSpec>,
    /// The whole-frame encoder, when `embed_frame` runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed: Option<EmbedSpec>,
    /// The crop encoder, when `embed_crops` runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_embed: Option<CropEmbedSpec>,
    /// The detector class ids whose boxes get a crop vector; `None` is every box (an empty set
    /// is none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_classes: Option<BTreeSet<u32>>,
    /// Whether the answer carries crop JPEGs.
    #[serde(default)]
    pub crops: CropsMode,
    /// Per conditional branch (`faces`, `ocr`), the detector classes that make it run.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub if_detected: BTreeMap<Branch, Vec<u32>>,
    /// The face detector and recogniser, when `faces` runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub faces: Option<FaceSpec>,
    /// The text detector and recogniser, when `ocr` runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr: Option<OcrSpec>,
    /// The stored renditions; they are reported in `branch_status` as `derivatives`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derivatives: Vec<RenditionSpec>,
}

impl EnginePlan {
    /// The plan text exactly as Python's `EnginePlan.to_json` writes it.
    ///
    /// # Errors
    ///
    /// Returns the serialiser's error; a plan made of these types has none.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        pyjson::dumps(&serde_json::to_value(self)?)
    }

    /// Whether `branch` is in the plan.
    #[must_use]
    pub fn runs(&self, branch: Branch) -> bool {
        self.branches.contains(&branch)
    }

    /// Whether the plan names renditions (`derivatives`).
    #[must_use]
    pub fn has_derivatives(&self) -> bool {
        !self.derivatives.is_empty()
    }
}
