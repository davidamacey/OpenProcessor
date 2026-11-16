//! The disk guard of the landing volume: the thresholds and the 507 message of
//! `src/core/disk_space.py`, measured on the volume the uploads are written to.
//!
//! A volume is [`DiskLevel::Low`] below the low threshold (a warning in the log) and
//! [`DiskLevel::Critical`] below the critical one, where a new upload is refused with 507 instead
//! of failing half way through a write. The free fraction is [`Landing::free_ratio`] (`statvfs`,
//! the call behind Python's `shutil.disk_usage`).

use std::error::Error;
use std::fmt;

use op_storage::error::StorageError;
use op_storage::landing::Landing;

use crate::settings::{DISK_CRITICAL_KEY, DISK_LOW_KEY, SettingsError};

/// How full the volume is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskLevel {
    /// Enough space.
    Ok,
    /// Below the low threshold: a warning.
    Low,
    /// Below the critical threshold: new writes are refused.
    Critical,
}

/// The low and critical thresholds, as fractions of the volume.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiskThresholds {
    low: f64,
    critical: f64,
}

impl DiskThresholds {
    /// Thresholds from percentages of the volume.
    ///
    /// # Errors
    ///
    /// [`SettingsError::Invalid`] naming the variable when a value is not a number from 0 to
    /// 100, or the critical value is above the low one.
    pub fn from_percent(low_pct: f64, critical_pct: f64) -> Result<Self, SettingsError> {
        for (key, value) in [(DISK_LOW_KEY, low_pct), (DISK_CRITICAL_KEY, critical_pct)] {
            if !(0.0..=100.0).contains(&value) {
                return Err(SettingsError::Invalid {
                    key,
                    reason: format!("{value} is not a percentage from 0 to 100"),
                });
            }
        }
        if critical_pct > low_pct {
            return Err(SettingsError::Invalid {
                key: DISK_CRITICAL_KEY,
                reason: format!("{critical_pct} must not exceed {DISK_LOW_KEY} ({low_pct})"),
            });
        }
        Ok(Self {
            low: low_pct / 100.0,
            critical: critical_pct / 100.0,
        })
    }

    /// The level of a volume with `free_ratio` (0.0 to 1.0) of it free.
    #[must_use]
    pub fn classify(&self, free_ratio: f64) -> DiskLevel {
        if free_ratio < self.critical {
            DiskLevel::Critical
        } else if free_ratio < self.low {
            DiskLevel::Low
        } else {
            DiskLevel::Ok
        }
    }
}

/// The volume is too full for a new write.
#[derive(Debug, Clone, PartialEq)]
pub struct InsufficientStorage {
    /// The volume's name in the answer.
    pub volume: &'static str,
    /// The free fraction that was measured.
    pub free_ratio: f64,
}

impl fmt::Display for InsufficientStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the {} volume has {:.1}% free; free space before starting new writes",
            self.volume,
            self.free_ratio * 100.0
        )
    }
}

/// Why the guard refused.
#[derive(Debug)]
pub enum DiskError {
    /// The volume is below the critical threshold.
    Critical(InsufficientStorage),
    /// The volume could not be measured; an unknown state refuses.
    Unmeasurable(StorageError),
}

impl fmt::Display for DiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical(c) => c.fmt(f),
            Self::Unmeasurable(e) => write!(f, "free space could not be measured: {e}"),
        }
    }
}

impl Error for DiskError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Critical(_) => None,
            Self::Unmeasurable(e) => Some(e),
        }
    }
}

/// Measures the landing volume and refuses at critical.
#[derive(Debug, Clone)]
pub struct DiskGuard {
    landing: Landing,
    thresholds: DiskThresholds,
    volume: &'static str,
}

impl DiskGuard {
    /// A guard for the volume of `landing`, called `volume` in the answers (Python calls its
    /// upload volume `data`).
    #[must_use]
    pub fn new(landing: Landing, thresholds: DiskThresholds, volume: &'static str) -> Self {
        Self {
            landing,
            thresholds,
            volume,
        }
    }

    /// The volume's name in the answers.
    #[must_use]
    pub fn volume(&self) -> &'static str {
        self.volume
    }

    /// Measure the volume; log a warning below low.
    ///
    /// # Errors
    ///
    /// [`DiskError::Critical`] below the critical threshold, [`DiskError::Unmeasurable`] when
    /// the volume cannot be measured.
    pub async fn check(&self) -> Result<DiskLevel, DiskError> {
        let free_ratio = self
            .landing
            .free_ratio()
            .await
            .map_err(DiskError::Unmeasurable)?;
        let level = self.thresholds.classify(free_ratio);
        if level != DiskLevel::Ok {
            tracing::warn!(
                volume = self.volume,
                free_percent = free_ratio * 100.0,
                ?level,
                "disk space is low"
            );
        }
        if level == DiskLevel::Critical {
            return Err(DiskError::Critical(InsufficientStorage {
                volume: self.volume,
                free_ratio,
            }));
        }
        Ok(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_uses_strictly_below_like_python() {
        let t = DiskThresholds::from_percent(10.0, 3.0).unwrap();
        assert_eq!(t.classify(0.03), DiskLevel::Low);
        assert_eq!(t.classify(0.029_999), DiskLevel::Critical);
        assert_eq!(t.classify(0.1), DiskLevel::Ok);
        assert_eq!(t.classify(0.099_9), DiskLevel::Low);
    }

    #[test]
    fn thresholds_refuse_what_python_refuses() {
        assert!(DiskThresholds::from_percent(3.0, 10.0).is_err());
        assert!(DiskThresholds::from_percent(101.0, 3.0).is_err());
        assert!(DiskThresholds::from_percent(10.0, -1.0).is_err());
        assert!(DiskThresholds::from_percent(f64::NAN, 3.0).is_err());
        assert!(DiskThresholds::from_percent(f64::INFINITY, 3.0).is_err());
    }

    #[test]
    fn the_message_matches_the_python_text() {
        let m = InsufficientStorage {
            volume: "data",
            free_ratio: 0.0123,
        };
        assert_eq!(
            m.to_string(),
            "the data volume has 1.2% free; free space before starting new writes"
        );
    }
}
