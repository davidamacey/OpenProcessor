//! Why an answer could not be read.

use std::error::Error;
use std::fmt;

/// An answer the client refuses to read. Every variant names what was wrong; an engine newer than
/// this client (a status, route, branch or reason it does not know) is one of them, never a guess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnswerError {
    /// `status` is not one of `contracts/json/frame_engine_v1.json`'s `photo_statuses`.
    UnknownStatus(String),
    /// `decode_route` is not one of the contract's `decode_routes`.
    UnknownDecodeRoute(String),
    /// `branch_status` names a branch the plan vocabulary does not have.
    UnknownBranch(String),
    /// `branch_status` has a state or error reason the contract does not list.
    UnknownBranchState(String),
    /// `branch_status` is not the object of states the protocol says.
    BranchStatus(String),
    /// An output the answer must carry is missing.
    Missing(&'static str),
    /// An output has the wrong datatype.
    Datatype {
        /// The output.
        name: &'static str,
        /// What the engine said.
        got: String,
        /// What the contract says.
        want: &'static str,
    },
    /// An output's shape or byte count does not fit its datatype or its row width.
    Shape {
        /// The output.
        name: &'static str,
        /// What did not fit.
        detail: String,
    },
    /// The response does not carry its outputs as `raw_output_contents`, one per output.
    RawContents(String),
    /// A text output is not UTF-8.
    Utf8(&'static str),
}

impl fmt::Display for AnswerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownStatus(s) => write!(f, "unknown photo status {s:?}"),
            Self::UnknownDecodeRoute(s) => write!(f, "unknown decode route {s:?}"),
            Self::UnknownBranch(s) => write!(f, "unknown branch {s:?} in branch_status"),
            Self::UnknownBranchState(s) => write!(f, "unknown branch state {s:?}"),
            Self::BranchStatus(s) => write!(f, "branch_status: {s}"),
            Self::Missing(n) => write!(f, "output {n} is missing"),
            Self::Datatype { name, got, want } => {
                write!(f, "output {name} is {got}, the contract says {want}")
            }
            Self::Shape { name, detail } => write!(f, "output {name}: {detail}"),
            Self::RawContents(s) => write!(f, "raw_output_contents: {s}"),
            Self::Utf8(n) => write!(f, "output {n} is not UTF-8"),
        }
    }
}

impl Error for AnswerError {}
