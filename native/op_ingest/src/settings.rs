//! The receiver's settings, read once at startup from an [`Env`] snapshot.
//!
//! The limits keep the names the Python route reads today (`src/config/curation.py`,
//! `src/config/settings.py`, `src/core/disk_space.py`) and their defaults. The admission,
//! hand-off and timeout bounds are new: Python had no bound on photos in work (a request held its
//! whole body in memory) and no idle-body limit.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use op_config::{Env, EnvError};

use crate::disk_space::DiskThresholds;

/// Largest photos per request (Python: `upload_max_images_per_request`, default 128).
pub const MAX_IMAGES_KEY: &str = "OP_UPLOAD_MAX_IMAGES_PER_REQUEST";
/// Largest accepted bytes per request (default 512 MiB).
pub const MAX_BYTES_KEY: &str = "OP_UPLOAD_MAX_BYTES_PER_REQUEST";
/// The accepted extensions, comma separated (default `.jpg,.jpeg,.png`).
pub const EXTENSIONS_KEY: &str = "OP_UPLOAD_ACCEPTED_EXTENSIONS";
/// Largest single file in MB (default 50). Python's name has no `OP_` prefix.
pub const MAX_FILE_MB_KEY: &str = "MAX_FILE_SIZE_MB";
/// Photos being received or worked on at once, in the whole process.
pub const PHOTOS_IN_WORK_KEY: &str = "OP_UPLOAD_PHOTOS_IN_WORK";
/// Bytes of photos being received or worked on at once, in the whole process.
pub const BYTES_IN_FLIGHT_KEY: &str = "OP_UPLOAD_BYTES_IN_FLIGHT";
/// Photos one request may have waiting for the sink.
pub const HANDOFF_QUEUE_KEY: &str = "OP_UPLOAD_HANDOFF_QUEUE";
/// Seconds a part may wait for admission before the request is answered 503.
pub const ADMISSION_WAIT_KEY: &str = "OP_UPLOAD_ADMISSION_WAIT_S";
/// Seconds a request body may stay silent before the request is answered 408.
pub const BODY_IDLE_KEY: &str = "OP_UPLOAD_BODY_IDLE_TIMEOUT_S";
/// Free space below this percentage of the landing volume is a warning.
pub const DISK_LOW_KEY: &str = "OP_DISK_LOW_PCT";
/// Free space below this percentage of the landing volume refuses new uploads.
pub const DISK_CRITICAL_KEY: &str = "OP_DISK_CRITICAL_PCT";

const MIB: u64 = 1024 * 1024;
/// Boundaries, part headers and the small form fields around the files
/// (`MULTIPART_OVERHEAD_BYTES` in `src/core/upload_limits.py`).
const MULTIPART_OVERHEAD: u64 = MIB;

/// Why the settings were refused.
#[derive(Debug)]
pub enum SettingsError {
    /// A variable is not set to a value of its type.
    Env(EnvError),
    /// A variable has a value of its type that cannot work.
    Invalid {
        /// The environment variable name.
        key: &'static str,
        /// What is wrong with it.
        reason: String,
    },
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Env(e) => e.fmt(f),
            Self::Invalid { key, reason } => write!(f, "{key}: {reason}"),
        }
    }
}

impl Error for SettingsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Env(e) => Some(e),
            Self::Invalid { .. } => None,
        }
    }
}

impl From<EnvError> for SettingsError {
    fn from(e: EnvError) -> Self {
        Self::Env(e)
    }
}

/// Everything the receiver is configured with.
#[derive(Debug, Clone, PartialEq)]
pub struct UploadSettings {
    /// Photos one request may carry; more is a 413.
    pub max_images: usize,
    /// Bytes of accepted photos one request may carry; more is a 413.
    pub max_total_bytes: u64,
    /// Bytes of one file; more is a 413.
    pub max_file_bytes: u64,
    /// The file extensions that are taken, lower case with the dot.
    pub accepted_extensions: Vec<String>,
    /// The disk guard's thresholds.
    pub disk: DiskThresholds,
    /// Photos in work, process wide.
    pub photos_in_work: usize,
    /// Bytes in flight, process wide.
    pub bytes_in_flight: u64,
    /// Photos one request may have queued for the sink.
    pub handoff_queue: usize,
    /// How long a part waits for admission.
    pub admission_wait: Duration,
    /// How long a body may stay silent.
    pub body_idle_timeout: Duration,
}

impl UploadSettings {
    /// Read the settings from `env`.
    ///
    /// # Errors
    ///
    /// [`SettingsError`] naming the variable that is not a number, is zero where a bound is
    /// needed, lists no extension, puts the critical disk threshold above the low one, or sets
    /// bytes in flight below the largest file (such a file could never be admitted).
    pub fn from_env(env: &Env) -> Result<Self, SettingsError> {
        let max_file_mb: u64 = positive(env, MAX_FILE_MB_KEY, 50_u64)?;
        let settings = Self {
            max_images: positive(env, MAX_IMAGES_KEY, 128_usize)?,
            max_total_bytes: positive(env, MAX_BYTES_KEY, 512 * MIB)?,
            max_file_bytes: max_file_mb.saturating_mul(MIB),
            accepted_extensions: extensions(env)?,
            disk: DiskThresholds::from_percent(
                env.get(DISK_LOW_KEY, 10.0_f64)?,
                env.get(DISK_CRITICAL_KEY, 3.0_f64)?,
            )?,
            photos_in_work: positive(env, PHOTOS_IN_WORK_KEY, 64_usize)?,
            bytes_in_flight: positive(env, BYTES_IN_FLIGHT_KEY, 2048 * MIB)?,
            handoff_queue: positive(env, HANDOFF_QUEUE_KEY, 4_usize)?,
            admission_wait: Duration::from_secs(positive(env, ADMISSION_WAIT_KEY, 10_u64)?),
            body_idle_timeout: Duration::from_secs(positive(env, BODY_IDLE_KEY, 60_u64)?),
        };
        if settings.bytes_in_flight < settings.max_file_bytes {
            return Err(SettingsError::Invalid {
                key: BYTES_IN_FLIGHT_KEY,
                reason: format!(
                    "{} is below the largest file ({} bytes, {MAX_FILE_MB_KEY}), which could never be admitted",
                    settings.bytes_in_flight, settings.max_file_bytes
                ),
            });
        }
        Ok(settings)
    }

    /// The largest request body: the larger of one file and the accepted bytes, plus the form
    /// around them (`request_exceeds_ceiling` in `src/core/upload_limits.py`).
    #[must_use]
    pub fn body_ceiling(&self) -> u64 {
        self.max_file_bytes
            .max(self.max_total_bytes)
            .saturating_add(MULTIPART_OVERHEAD)
    }

    /// The file cap in whole MB, as the 413 answer names it.
    #[must_use]
    pub fn max_file_mb(&self) -> u64 {
        self.max_file_bytes / MIB
    }
}

fn positive<T>(env: &Env, key: &'static str, default: T) -> Result<T, SettingsError>
where
    T: std::str::FromStr + PartialEq + Default,
    T::Err: fmt::Display,
{
    let value = env.get(key, default)?;
    if value == T::default() {
        return Err(SettingsError::Invalid {
            key,
            reason: "must be at least 1".into(),
        });
    }
    Ok(value)
}

/// `OP_UPLOAD_ACCEPTED_EXTENSIONS` as Python reads it: split on commas, each trimmed and lower
/// cased, empty entries dropped. An empty list refuses every photo, so it is refused here.
fn extensions(env: &Env) -> Result<Vec<String>, SettingsError> {
    let raw: String = env.get(EXTENSIONS_KEY, ".jpg,.jpeg,.png".to_string())?;
    let list: Vec<String> = raw
        .split(',')
        .map(|e| e.trim().to_lowercase())
        .filter(|e| !e.is_empty())
        .collect();
    if list.is_empty() {
        return Err(SettingsError::Invalid {
            key: EXTENSIONS_KEY,
            reason: "lists no extension, so every photo would be refused".into(),
        });
    }
    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(pairs: &[(&str, &str)]) -> Result<UploadSettings, SettingsError> {
        UploadSettings::from_env(&Env::from_pairs(
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())),
        ))
    }

    #[test]
    fn the_defaults_are_the_ones_the_python_route_serves() {
        let s = read(&[]).unwrap();
        assert_eq!(s.max_images, 128);
        assert_eq!(s.max_total_bytes, 512 * 1024 * 1024);
        assert_eq!(s.max_file_bytes, 50 * 1024 * 1024);
        assert_eq!(s.accepted_extensions, [".jpg", ".jpeg", ".png"]);
        assert_eq!(s.body_ceiling(), 512 * 1024 * 1024 + 1024 * 1024);
        assert_eq!(s.max_file_mb(), 50);
    }

    #[test]
    fn the_ceiling_is_the_larger_of_one_file_and_the_accepted_bytes() {
        let s = read(&[("MAX_FILE_SIZE_MB", "600")]).unwrap();
        assert_eq!(s.body_ceiling(), 601 * 1024 * 1024);
    }

    #[test]
    fn extensions_are_trimmed_lower_cased_and_empty_entries_dropped() {
        let s = read(&[(EXTENSIONS_KEY, " .JPG , ,.Png ")]).unwrap();
        assert_eq!(s.accepted_extensions, [".jpg", ".png"]);
    }

    #[test]
    fn what_cannot_work_is_refused_naming_the_variable() {
        for (key, value) in [
            (MAX_IMAGES_KEY, "0"),
            (MAX_BYTES_KEY, "0"),
            (MAX_FILE_MB_KEY, "0"),
            (PHOTOS_IN_WORK_KEY, "0"),
            (HANDOFF_QUEUE_KEY, "0"),
            (ADMISSION_WAIT_KEY, "0"),
            (BODY_IDLE_KEY, "0"),
            (MAX_IMAGES_KEY, "-1"),
            (MAX_BYTES_KEY, "lots"),
            (EXTENSIONS_KEY, " , "),
            (DISK_CRITICAL_KEY, "20"),
        ] {
            let message = read(&[(key, value)]).unwrap_err().to_string();
            assert!(message.contains(key), "{key}={value}: {message}");
        }
    }

    #[test]
    fn bytes_in_flight_below_the_largest_file_is_refused() {
        let message = read(&[(BYTES_IN_FLIGHT_KEY, "1000")])
            .unwrap_err()
            .to_string();
        assert!(
            message.contains(BYTES_IN_FLIGHT_KEY) && message.contains(MAX_FILE_MB_KEY),
            "{message}"
        );
    }
}
