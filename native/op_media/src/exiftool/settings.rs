//! The pool's settings, read through [`op_config::Env`] like every other `OP_*` setting.

use std::path::PathBuf;
use std::time::Duration;

use op_config::{Env, EnvError};

/// The ExifTool program: a name looked up on `PATH` or a path.
pub const BIN_KEY: &str = "OP_EXIFTOOL_BIN";
/// How many long-lived ExifTool processes this server process keeps.
pub const POOL_SIZE_KEY: &str = "OP_EXIFTOOL_POOL_SIZE";
/// The longest one read may take once it has a process, in seconds.
pub const READ_TIMEOUT_KEY: &str = "OP_EXIFTOOL_READ_TIMEOUT_S";
/// The longest a read waits for a free process, in seconds.
pub const QUEUE_WAIT_KEY: &str = "OP_EXIFTOOL_QUEUE_WAIT_S";
/// How many reads may wait for a free process at once; one more is refused at once.
pub const MAX_WAITING_KEY: &str = "OP_EXIFTOOL_MAX_WAITING";

const MAX_POOL_SIZE: usize = 64;
const MAX_WAITING: usize = 100_000;
const MAX_SECONDS: f64 = 3600.0;
const DEFAULT_READ_TIMEOUT_S: f64 = 10.0;
const DEFAULT_QUEUE_WAIT_S: f64 = 30.0;
const DEFAULT_WAITING_PER_PROCESS: usize = 4;

/// What the pool needs to start and to bound its work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExifToolSettings {
    /// The ExifTool program.
    pub binary: PathBuf,
    /// Processes kept, at least 1.
    pub pool_size: usize,
    /// One read's bound once it has a process; a process that exceeds it is killed.
    pub read_timeout: Duration,
    /// How long a read waits for a free process before it is refused.
    pub queue_wait: Duration,
    /// Reads allowed to wait at once; the next is refused without waiting (back pressure).
    pub max_waiting: usize,
}

impl ExifToolSettings {
    /// Read the settings from `env`. The pool defaults to half the cores, from 2 to 16 processes
    /// (each is a Perl interpreter of some tens of megabytes); the first run on the real corpus
    /// (plan 18, M5) tunes it.
    ///
    /// # Errors
    ///
    /// [`EnvError::Invalid`], naming the key, for a value that does not parse or is out of range
    /// (a pool of 0 or more than 64, a bound that is not above 0 seconds or is over an hour).
    pub fn from_env(env: &Env) -> Result<Self, EnvError> {
        let cores = std::thread::available_parallelism().map_or(2, usize::from);
        let pool_size = env.get(POOL_SIZE_KEY, (cores / 2).clamp(2, 16))?;
        if !(1..=MAX_POOL_SIZE).contains(&pool_size) {
            return Err(out_of_range(POOL_SIZE_KEY, pool_size, "must be 1 to 64"));
        }
        let max_waiting = env.get(MAX_WAITING_KEY, pool_size * DEFAULT_WAITING_PER_PROCESS)?;
        if max_waiting > MAX_WAITING {
            return Err(out_of_range(
                MAX_WAITING_KEY,
                max_waiting,
                "must be at most 100000",
            ));
        }
        Ok(Self {
            binary: PathBuf::from(env.get(BIN_KEY, String::from("exiftool"))?),
            pool_size,
            read_timeout: seconds(env, READ_TIMEOUT_KEY, DEFAULT_READ_TIMEOUT_S)?,
            queue_wait: seconds(env, QUEUE_WAIT_KEY, DEFAULT_QUEUE_WAIT_S)?,
            max_waiting,
        })
    }
}

fn seconds(env: &Env, key: &str, default: f64) -> Result<Duration, EnvError> {
    let value: f64 = env.get(key, default)?;
    if value > 0.0 && value <= MAX_SECONDS {
        Ok(Duration::from_secs_f64(value))
    } else {
        Err(out_of_range(
            key,
            value,
            "must be above 0 and at most 3600 seconds",
        ))
    }
}

fn out_of_range(key: &str, value: impl ToString, reason: &str) -> EnvError {
    EnvError::Invalid {
        key: key.into(),
        value: value.to_string(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_ones_the_python_client_has_where_it_has_any() {
        let s = ExifToolSettings::from_env(&Env::default()).unwrap();
        assert_eq!(s.binary, PathBuf::from("exiftool"));
        assert_eq!(s.read_timeout, Duration::from_secs(10));
        assert!((2..=16).contains(&s.pool_size));
        assert_eq!(s.max_waiting, s.pool_size * 4);
    }

    #[test]
    fn every_key_overrides() {
        let env = Env::from_pairs([
            (BIN_KEY, "/opt/exiftool"),
            (POOL_SIZE_KEY, "3"),
            (READ_TIMEOUT_KEY, "2.5"),
            (QUEUE_WAIT_KEY, "7"),
            (MAX_WAITING_KEY, "0"),
        ]);
        let s = ExifToolSettings::from_env(&env).unwrap();
        assert_eq!(s.binary, PathBuf::from("/opt/exiftool"));
        assert_eq!(s.pool_size, 3);
        assert_eq!(s.read_timeout, Duration::from_millis(2500));
        assert_eq!(s.queue_wait, Duration::from_secs(7));
        assert_eq!(s.max_waiting, 0);
    }

    #[test]
    fn a_bad_value_is_refused_and_names_its_key() {
        for (key, value) in [
            (POOL_SIZE_KEY, "0"),
            (POOL_SIZE_KEY, "65"),
            (POOL_SIZE_KEY, "many"),
            (READ_TIMEOUT_KEY, "0"),
            (READ_TIMEOUT_KEY, "-1"),
            (READ_TIMEOUT_KEY, "nan"),
            (QUEUE_WAIT_KEY, "3601"),
            (MAX_WAITING_KEY, "100001"),
        ] {
            let err = ExifToolSettings::from_env(&Env::from_pairs([(key, value)])).unwrap_err();
            assert!(err.to_string().contains(key), "{key}={value}: {err}");
        }
    }
}
