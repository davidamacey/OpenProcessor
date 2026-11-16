//! The engine protocol v1 client: BYTES codec, plan builder, answer parser, channel pool and readiness, plus the model server readiness and load calls.
#![forbid(unsafe_code)]

pub mod answer;
pub mod codec;
pub mod plan;
pub mod proto;

pub use answer::{
    Answer, AnswerError, BranchError, BranchName, BranchState, CpuReason, CropEmbeddings,
    DecodeRoute, Detections, FaceEmbeddings, Faces, FrameSize, Letterbox, Matrix, PhotoStatus,
    TextLines,
};

pub use plan::{
    Branch, CROP_FILL, CropEmbedSpec, CropRecipe, CropVectors, CropsMode,
    DEFAULT_RENDITION_QUALITY, DetLimit, DetectorSpec, EmbedSpec, EncoderModels, EnginePlan,
    FaceSpec, FrameRecipe, IngestPlan, OcrSpec, ProtocolVersion, RenditionFormat, RenditionSpec,
};
