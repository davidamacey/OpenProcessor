//! The typed errors of a call.

use std::error::Error;
use std::fmt;

use tonic::Code;

use crate::answer::AnswerError;

/// A client that cannot be built: a setting the deployment must fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The address is not a URI the transport can use.
    Url {
        /// The address as given.
        url: String,
        /// What the parser said.
        reason: String,
    },
    /// A pool must have at least one channel and one in-flight permit.
    Empty(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Url { url, reason } => write!(f, "engine address {url:?}: {reason}"),
            Self::Empty(what) => write!(f, "{what} must be at least 1"),
        }
    }
}

impl Error for ConfigError {}

/// The engine gave no answer for the photo. Every variant is a photo the caller cannot store
/// a result for; what the person sees differs (`Python`: an engine that is down, not ready or
/// still overloaded is `engine_unavailable`, a retry later can succeed).
#[derive(Debug)]
pub enum EngineError {
    /// `RESOURCE_EXHAUSTED` until the photo's deadline, or no in-flight permit before it: the
    /// engine (or this client) is overloaded.
    Overloaded {
        /// Calls made.
        attempts: u32,
        /// The last status's message.
        message: String,
    },
    /// `UNAVAILABLE` (not ready, draining, model server down, cannot connect) or a call lost in
    /// transit, still failing after the retries.
    Unavailable {
        /// Calls made.
        attempts: u32,
        /// The last status's message.
        message: String,
    },
    /// The engine refused the request itself (`INVALID_ARGUMENT`: a malformed request or plan,
    /// `NOT_FOUND`: no such model). Sending it again changes nothing.
    Rejected {
        /// The status.
        code: Code,
        /// The status's message: the plan fault's name comes first.
        message: String,
    },
    /// Any other status: the engine's own failure for this photo (`INTERNAL` starting with
    /// `frame engine: `), a call past its deadline, an answer the transport cut. Not retried.
    Failed {
        /// The status.
        code: Code,
        /// Calls made.
        attempts: u32,
        /// The status's message.
        message: String,
    },
    /// The engine answered, and the answer is not one this client reads.
    Answer(AnswerError),
    /// The plan could not be written as JSON.
    Plan(serde_json::Error),
}

impl EngineError {
    /// Whether the last answer was the overload status.
    #[must_use]
    pub fn overloaded(&self) -> bool {
        matches!(self, Self::Overloaded { .. })
    }

    /// Whether a retry later can succeed (the engine's load or absence, not the photo): the
    /// upload answers 503 for these.
    #[must_use]
    pub fn transient(&self) -> bool {
        matches!(self, Self::Overloaded { .. } | Self::Unavailable { .. })
    }

    /// The gRPC status name of the last attempt (`UNAVAILABLE`, `INTERNAL`, ...), as the
    /// `op_engine_call_failures_total` metric labels it; `NOT_GRPC` when no status was involved.
    #[must_use]
    pub fn code_name(&self) -> &'static str {
        match self {
            Self::Overloaded { .. } => code_name(Code::ResourceExhausted),
            Self::Unavailable { .. } => code_name(Code::Unavailable),
            Self::Rejected { code, .. } | Self::Failed { code, .. } => code_name(*code),
            Self::Answer(_) | Self::Plan(_) => "NOT_GRPC",
        }
    }

    /// How many calls were made.
    #[must_use]
    pub fn attempts(&self) -> u32 {
        match self {
            Self::Overloaded { attempts, .. }
            | Self::Unavailable { attempts, .. }
            | Self::Failed { attempts, .. } => *attempts,
            Self::Rejected { .. } | Self::Answer(_) | Self::Plan(_) => 1,
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overloaded { attempts, message } => {
                write!(
                    f,
                    "the engine stayed overloaded after {attempts} calls: {message}"
                )
            }
            Self::Unavailable { attempts, message } => {
                write!(
                    f,
                    "the engine was unavailable for {attempts} calls: {message}"
                )
            }
            Self::Rejected { code, message } => {
                write!(
                    f,
                    "the engine refused the request ({}): {message}",
                    code_name(*code)
                )
            }
            Self::Failed {
                code,
                attempts,
                message,
            } => {
                write!(
                    f,
                    "the engine failed ({}) after {attempts} calls: {message}",
                    code_name(*code)
                )
            }
            Self::Answer(error) => write!(f, "the engine's answer cannot be read: {error}"),
            Self::Plan(error) => write!(f, "the plan cannot be written: {error}"),
        }
    }
}

impl Error for EngineError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Answer(error) => Some(error),
            Self::Plan(error) => Some(error),
            _ => None,
        }
    }
}

/// A gRPC code under the name the status table, the contract and the metric use.
#[must_use]
pub fn code_name(code: Code) -> &'static str {
    match code {
        Code::Ok => "OK",
        Code::Cancelled => "CANCELLED",
        Code::Unknown => "UNKNOWN",
        Code::InvalidArgument => "INVALID_ARGUMENT",
        Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        Code::NotFound => "NOT_FOUND",
        Code::AlreadyExists => "ALREADY_EXISTS",
        Code::PermissionDenied => "PERMISSION_DENIED",
        Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        Code::FailedPrecondition => "FAILED_PRECONDITION",
        Code::Aborted => "ABORTED",
        Code::OutOfRange => "OUT_OF_RANGE",
        Code::Unimplemented => "UNIMPLEMENTED",
        Code::Internal => "INTERNAL",
        Code::Unavailable => "UNAVAILABLE",
        Code::DataLoss => "DATA_LOSS",
        Code::Unauthenticated => "UNAUTHENTICATED",
    }
}
