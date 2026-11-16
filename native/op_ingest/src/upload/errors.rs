//! Turning the errors of the body and of the landing zone into refusals.

use std::error::Error;
use std::io::ErrorKind;

use http_body_util::LengthLimitError;
use op_storage::error::StorageError;
use tower_http::timeout::TimeoutError;

use super::refusal::Refusal;
use super::sink::SinkError;
use crate::disk_space::InsufficientStorage;
use crate::settings::UploadSettings;

/// FastAPI's answer for a body it cannot parse, when the parser gave no message of its own.
pub const BODY_PARSE_DETAIL: &str = "There was an error parsing the body";

/// The refusal for an error raised while reading the body: the size limit (413 with the same
/// body as Python's), the idle limit (408), or a body that is not a complete form (400).
pub(super) fn body_refusal(error: &(dyn Error + 'static), settings: &UploadSettings) -> Refusal {
    let mut next: Option<&(dyn Error + 'static)> = Some(error);
    while let Some(e) = next {
        if e.downcast_ref::<LengthLimitError>().is_some() {
            return Refusal::FileTooLarge {
                max_file_size_mb: settings.max_file_mb(),
            };
        }
        if e.downcast_ref::<TimeoutError>().is_some() {
            return Refusal::BodyIdle {
                seconds: settings.body_idle_timeout.as_secs(),
            };
        }
        next = e.source();
    }
    tracing::debug!(error = %error, "the upload body is not a complete form");
    Refusal::Malformed(BODY_PARSE_DETAIL.to_string())
}

/// The refusal for a failed landing write: a full disk is the 507 answer, anything else is a
/// server error whose cause is logged.
pub(super) fn storage_refusal(error: &StorageError, volume: &'static str) -> Refusal {
    if let StorageError::Io { source, .. } = error
        && source.kind() == ErrorKind::StorageFull
    {
        return Refusal::InsufficientStorage(InsufficientStorage {
            volume,
            free_ratio: 0.0,
        });
    }
    tracing::error!(error = %error, "a landing write failed");
    Refusal::Internal
}

impl From<SinkError> for Refusal {
    fn from(error: SinkError) -> Self {
        match error {
            SinkError::Unavailable { detail, code } => Self::Unavailable { detail, code },
            SinkError::Failed(why) => {
                tracing::error!(error = %why, "the ingest sink failed");
                Self::Internal
            }
        }
    }
}

impl From<crate::admission::Busy> for Refusal {
    fn from(busy: crate::admission::Busy) -> Self {
        tracing::debug!(bound = ?busy.bound, "an upload part could not be admitted in time");
        Self::Busy
    }
}
