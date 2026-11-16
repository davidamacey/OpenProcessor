//! The answer to one photo, typed, read as `src/clients/frame_engine.py` `_answer` and
//! `src/domain/frame_engine_wire.py` read it: the status, the decode route, one state per branch,
//! and, for a decoded photo, the tensors of the branches the plan asked for.
//!
//! The statuses, routes, branch states and error reasons are closed enums checked against
//! `contracts/json/frame_engine_v1.json` by the tests. Something a newer engine says that this
//! client does not know is an [`AnswerError`], never a silent default. Datatypes and shapes are
//! the contract's too; an output that does not fit them is refused.

mod branches;
mod error;
mod outputs;
mod tensor;

use std::collections::BTreeMap;

use bytes::Bytes;

pub use self::branches::{
    BranchError, BranchName, BranchState, CpuReason, DecodeRoute, PhotoStatus,
};
pub use self::error::AnswerError;
pub use self::outputs::{
    Detections, FaceEmbeddings, Faces, FrameSize, Letterbox, TextLines, expected_outputs,
};
pub use self::tensor::Matrix;
use self::tensor::Outputs;
use crate::plan::EnginePlan;
use crate::proto::ModelInferResponse;

/// One photo's answer. The tensor groups are `Some` when the plan asked for the branch and the
/// engine produced it; a photo that was not decoded (`status` is not [`PhotoStatus::Ok`]) has none.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// Whether the photo was decoded, and if not, why.
    pub status: PhotoStatus,
    /// Which decoder produced the pixels.
    pub decode_route: DecodeRoute,
    /// One state per requested branch (and `derivatives` when the plan names renditions).
    pub branches: BTreeMap<BranchName, BranchState>,
    /// The decoded frame's size in pixels, once decoded.
    pub size: Option<FrameSize>,
    /// `detect`: boxes, scores, classes, the letterbox and the blur sums.
    pub detections: Option<Detections>,
    /// `embed_frame`: the whole frame's vector, not normalised.
    pub whole_embedding: Option<Vec<f32>>,
    /// `embed_crops`: one vector per picked box, and which boxes were picked.
    pub crop_embeddings: Option<CropEmbeddings>,
    /// `crops` inline: one JPEG per box, as slices of the response.
    pub crop_jpegs: Option<Vec<Bytes>>,
    /// `faces`: boxes, scores, landmarks and blur sums.
    pub faces: Option<Faces>,
    /// `face_embed`: one vector per embedded face, and which faces were embedded.
    pub face_embeddings: Option<FaceEmbeddings>,
    /// `ocr`: one row per text line.
    pub text: Option<TextLines>,
    /// `derivatives`: one JPEG per rendition, in the plan's order.
    pub derivative_jpegs: Option<Vec<Bytes>>,
}

/// `embed_crops`' answer: the engine embeds only the boxes the plan's `embed_classes` picks, so
/// the matrix has a row per picked box and `embedded` has a flag per box.
#[derive(Debug, Clone, PartialEq)]
pub struct CropEmbeddings {
    /// One row per picked box, in box order. `0 x 0` when nothing was picked.
    pub vectors: Matrix,
    /// Per box, whether it was picked.
    pub embedded: Vec<bool>,
}

impl Answer {
    /// Whether the photo was decoded and every requested branch ran (`EngineAnswer.ok`).
    #[must_use]
    pub fn ok(&self) -> bool {
        self.status == PhotoStatus::Ok && self.branches.values().all(|s| *s == BranchState::Ok)
    }

    /// The first branch (by name order) that is not `ok`, with its state.
    #[must_use]
    pub fn first_unfinished_branch(&self) -> Option<(BranchName, BranchState)> {
        self.branches
            .iter()
            .find(|(_, state)| **state != BranchState::Ok)
            .map(|(name, state)| (*name, *state))
    }

    /// Reads `response`, the engine's answer to a request that carried `plan`.
    ///
    /// # Errors
    ///
    /// [`AnswerError`] for a status, route, branch or reason the contract does not list, a
    /// `branch_status` that is not the object the protocol says or does not name the plan's
    /// branches, an output of the wrong datatype or shape, a response without `raw_output_contents`
    /// for each output, and, when the photo was decoded and every branch ran, a missing output.
    pub fn parse(response: &ModelInferResponse, plan: &EnginePlan) -> Result<Self, AnswerError> {
        let tensors = Outputs::new(response)?;
        let status = PhotoStatus::parse(&tensors.text("status")?)?;
        let decode_route = DecodeRoute::parse(&tensors.text("decode_route")?)?;
        let branches = branches::parse_states(&tensors.text("branch_status")?)?;
        let mut answer = Answer {
            status,
            decode_route,
            branches,
            size: None,
            detections: None,
            whole_embedding: None,
            crop_embeddings: None,
            crop_jpegs: None,
            faces: None,
            face_embeddings: None,
            text: None,
            derivative_jpegs: None,
        };
        if status != PhotoStatus::Ok {
            return Ok(answer);
        }
        branches::check_reported(&answer.branches, plan)?;
        outputs::read_into(&mut answer, &tensors, plan)?;
        if answer.ok() {
            outputs::require_all(&answer, plan)?;
        }
        Ok(answer)
    }
}
