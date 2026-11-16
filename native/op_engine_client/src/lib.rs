//! The engine protocol v1 client: what the Rust server uses to ask the image engine for a
//! photo's objects, vectors, crops, faces and text, written against the protocol
//! (`docs/design/v070_rebuild/17_native_engine_plan.md` section 1.1) and matched to the Python
//! client it replaces (`src/clients/frame_engine.py`, `src/domain/frame_engine_wire.py`).
//!
//! | Module | Concern |
//! |---|---|
//! | [`codec`] | the BYTES tensor of KServe raw contents |
//! | [`input`] | how a photo is named in a request: its bytes, or the file the engine reads itself |
//! | [`plan`] | the `plan` input: the plan types, the ingest and stored-boxes plans, the JSON text as Python writes it |
//! | [`answer`] | the answer to one photo, typed: status, route, branch states, tensors |
//! | [`client`] | the channel pool, the in-flight cap, a deadline on every call, retries, readiness |
//! | [`retry`], [`settings`] | which failures are sent again and how, the settings |
//! | [`model_server`] | the model server's readiness and load calls |
//! | [`proto`] | the KServe messages, generated from the one copy of the protos |
#![forbid(unsafe_code)]

pub mod answer;
mod channel;
pub mod client;
pub mod codec;
pub mod error;
pub mod input;
pub mod model_server;
pub mod observer;
pub mod plan;
pub mod proto;
pub mod retry;
pub mod settings;

pub use answer::{
    Answer, AnswerError, BranchError, BranchName, BranchState, CpuReason, CropEmbeddings,
    DecodeRoute, Detections, FaceEmbeddings, Faces, FrameSize, Letterbox, Matrix, PhotoStatus,
    TextLines,
};
pub use client::{EngineClient, MODEL, Readiness};
pub use error::{ConfigError, EngineError};
pub use input::PhotoInput;
pub use model_server::{LoadError, ModelServer};
pub use observer::{CallObserver, ProbeOutcome};
pub use plan::{
    Branch, CROP_FILL, CropEmbedSpec, CropRecipe, CropVectors, CropsMode,
    DEFAULT_RENDITION_QUALITY, DetLimit, DetectorSpec, EmbedSpec, EncoderModels, EnginePlan,
    FaceSpec, FrameRecipe, IngestPlan, OcrSpec, ProtocolVersion, RenditionFormat, RenditionSpec,
};
pub use settings::{EngineSettings, ModelServerSettings};
