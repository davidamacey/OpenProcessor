//! `status`, `decode_route` and `branch_status`: the three outputs every answer carries.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::AnswerError;
use crate::plan::{Branch, EnginePlan};

/// `status`: `ok`, or why the photo was not decoded (`photo_statuses` of the contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhotoStatus {
    /// Decoded.
    Ok,
    /// The bytes are not an image the engine can read.
    Corrupt,
    /// More pixels than the engine accepts.
    Huge,
    /// More device memory than the engine's budget allows for one photo.
    OverBudget,
    /// The engine's device-memory wait ran out: its load, not the file; a retry later can succeed.
    GpuMemory,
    /// The decoder failed on the photo.
    Decoder,
}

impl PhotoStatus {
    /// Every status, with the name the engine writes.
    pub const ALL: [(PhotoStatus, &'static str); 6] = [
        (PhotoStatus::Ok, "ok"),
        (PhotoStatus::Corrupt, "corrupt"),
        (PhotoStatus::Huge, "huge"),
        (PhotoStatus::OverBudget, "over_budget"),
        (PhotoStatus::GpuMemory, "gpu_memory"),
        (PhotoStatus::Decoder, "decoder"),
    ];

    pub(super) fn parse(text: &str) -> Result<Self, AnswerError> {
        Self::ALL
            .iter()
            .find(|(_, name)| *name == text)
            .map(|(status, _)| *status)
            .ok_or_else(|| AnswerError::UnknownStatus(text.to_owned()))
    }

    /// The name the engine writes.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(status, _)| *status == self)
            .map_or("", |(_, name)| *name)
    }

    /// Whether the status is the engine's load and not the file (`_TRANSIENT_REASONS`: the
    /// photo is worth sending again later).
    #[must_use]
    pub fn is_transient(self) -> bool {
        self == PhotoStatus::GpuMemory
    }
}

/// Why the engine decoded a photo on the host (`CPU_REASONS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    /// Not a JPEG.
    NotJpeg,
    /// A JPEG of a precision nvJPEG does not take.
    Precision,
    /// A JPEG with a component count nvJPEG does not take.
    Components,
    /// A progressive JPEG.
    Progressive,
    /// A truncated JPEG.
    Truncated,
    /// Too small for the GPU to be worth it.
    Small,
}

/// `decode_route`: which decoder produced the pixels (`decode_routes` of the contract).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeRoute {
    /// nvJPEG.
    Gpu,
    /// The host decoder, for this reason.
    Cpu(CpuReason),
    /// Nothing was decoded.
    None,
}

impl DecodeRoute {
    /// Every route, with the name the engine writes.
    pub const ALL: [(DecodeRoute, &'static str); 8] = [
        (DecodeRoute::Gpu, "gpu"),
        (DecodeRoute::Cpu(CpuReason::NotJpeg), "cpu_not_jpeg"),
        (DecodeRoute::Cpu(CpuReason::Precision), "cpu_precision"),
        (DecodeRoute::Cpu(CpuReason::Components), "cpu_components"),
        (DecodeRoute::Cpu(CpuReason::Progressive), "cpu_progressive"),
        (DecodeRoute::Cpu(CpuReason::Truncated), "cpu_truncated"),
        (DecodeRoute::Cpu(CpuReason::Small), "cpu_small"),
        (DecodeRoute::None, "none"),
    ];

    pub(super) fn parse(text: &str) -> Result<Self, AnswerError> {
        Self::ALL
            .iter()
            .find(|(_, name)| *name == text)
            .map(|(route, _)| *route)
            .ok_or_else(|| AnswerError::UnknownDecodeRoute(text.to_owned()))
    }

    /// The name the engine writes and the images document stores.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(route, _)| *route == self)
            .map_or("", |(_, name)| *name)
    }
}

/// An entry of `branch_status`: a branch of the plan, or `derivatives` (a plan field, not a
/// branch, reported under its own name).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BranchName {
    /// A branch of the plan vocabulary.
    Branch(Branch),
    /// The stored renditions.
    Derivatives,
}

impl BranchName {
    fn parse(text: &str) -> Result<Self, AnswerError> {
        if text == "derivatives" {
            return Ok(Self::Derivatives);
        }
        Branch::ALL
            .iter()
            .find(|branch| branch.name() == text)
            .map(|branch| Self::Branch(*branch))
            .ok_or_else(|| AnswerError::UnknownBranch(text.to_owned()))
    }

    /// The name `branch_status` uses.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Branch(branch) => branch.name(),
            Self::Derivatives => "derivatives",
        }
    }
}

/// Why a branch failed (`branch_error_reasons`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchError {
    /// In the plan vocabulary, not built in this engine.
    UnsupportedBranch,
    /// The model call failed.
    ModelError,
    /// A branch this one needs failed.
    DependencyFailed,
    /// The photo was not decoded: nothing ran.
    ImageFailed,
}

impl BranchError {
    /// Every reason, with the name the engine writes.
    pub const ALL: [(BranchError, &'static str); 4] = [
        (BranchError::UnsupportedBranch, "unsupported_branch"),
        (BranchError::ModelError, "model_error"),
        (BranchError::DependencyFailed, "dependency_failed"),
        (BranchError::ImageFailed, "image_failed"),
    ];

    /// The name the engine writes.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(reason, _)| *reason == self)
            .map_or("", |(_, name)| *name)
    }
}

/// One branch's state (`branch_states`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchState {
    /// It ran.
    Ok,
    /// It did not run (a condition of the plan was not met).
    NotRun,
    /// It failed.
    Error(BranchError),
}

#[derive(Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum WireState {
    Ok,
    NotRun,
    Error { reason: String },
}

fn state_of(wire: WireState) -> Result<BranchState, AnswerError> {
    Ok(match wire {
        WireState::Ok => BranchState::Ok,
        WireState::NotRun => BranchState::NotRun,
        WireState::Error { reason } => BranchState::Error(
            BranchError::ALL
                .iter()
                .find(|(_, name)| *name == reason)
                .map(|(error, _)| *error)
                .ok_or_else(|| AnswerError::UnknownBranchState(format!("error: {reason}")))?,
        ),
    })
}

/// `branch_status`: one JSON object, an entry per reported branch.
pub(super) fn parse_states(text: &str) -> Result<BTreeMap<BranchName, BranchState>, AnswerError> {
    let wire: BTreeMap<String, serde_json::Value> =
        serde_json::from_str(text).map_err(|e| AnswerError::BranchStatus(e.to_string()))?;
    let mut states = BTreeMap::new();
    for (name, entry) in wire {
        let state = serde_json::from_value::<WireState>(entry)
            .map_err(|e| AnswerError::UnknownBranchState(format!("{name}: {e}")))?;
        states.insert(BranchName::parse(&name)?, state_of(state)?);
    }
    Ok(states)
}

/// The names `branch_status` must carry for `plan`: its branches, and `derivatives` when it
/// names renditions (`EnginePlan.reported`).
pub(super) fn reported(plan: &EnginePlan) -> BTreeSet<BranchName> {
    let mut names: BTreeSet<BranchName> = plan
        .branches
        .iter()
        .copied()
        .map(BranchName::Branch)
        .collect();
    if plan.has_derivatives() {
        names.insert(BranchName::Derivatives);
    }
    names
}

/// A decoded photo's `branch_status` has exactly one entry per requested branch.
pub(super) fn check_reported(
    states: &BTreeMap<BranchName, BranchState>,
    plan: &EnginePlan,
) -> Result<(), AnswerError> {
    let wanted = reported(plan);
    let got: BTreeSet<BranchName> = states.keys().copied().collect();
    if let Some(name) = wanted.difference(&got).next() {
        return Err(AnswerError::BranchStatus(format!(
            "no entry for {}",
            name.name()
        )));
    }
    if let Some(name) = got.difference(&wanted).next() {
        return Err(AnswerError::BranchStatus(format!(
            "an entry for {}, which the plan did not ask for",
            name.name()
        )));
    }
    Ok(())
}
