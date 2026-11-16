//! A pool of long-lived ExifTool processes read by path.
//!
//! Each process is `exiftool -stay_open True -@ -`, the documented way to avoid paying ExifTool's
//! start-up per file (<https://exiftool.org/exiftool_pod.html>, `-stay_open`). One read is one
//! request on one process, framed with `-execute<n>` and answered up to `{ready<n>}` (see
//! [`ExifToolPool::read`]). The arguments are the ones `src/clients/exiftool_client.py` sends:
//! `-j -fast` and one `-<tag>` per wanted tag, with `-G -n` given once at start-up as
//! `-common_args`.
//!
//! A process that fails or exceeds the read bound is killed and reaped, and the next read starts a
//! replacement. A read that finds every process busy waits, but only up to a bound and only behind
//! a bounded number of other waiting reads; past either it is refused with
//! [`ExifToolError::Busy`], so a flood of photos pushes back on the caller instead of queueing
//! without limit. The pool is per server process: several processes each hold their own.

mod child;
mod settings;

use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use serde_json::{Map, Value};
use tokio::sync::{Semaphore, TryAcquireError};
use tokio::time::timeout;

use child::{Child, ChildError};
pub use settings::{
    BIN_KEY, ExifToolSettings, MAX_WAITING_KEY, POOL_SIZE_KEY, QUEUE_WAIT_KEY, READ_TIMEOUT_KEY,
};

/// The tags of one file as ExifTool printed them (`-j -G -n`), the first row of its answer.
pub type Tags = Map<String, Value>;

/// Why a read did not give tags. The messages never carry ExifTool's own text: it can hold a
/// coordinate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExifToolError {
    /// The pool could not start a process (the program is missing, is not ExifTool 12.15 or newer,
    /// or did not answer). `binary` is the value of `OP_EXIFTOOL_BIN`.
    Start {
        /// The program that was tried.
        binary: String,
    },
    /// The process did not answer in time and was killed; a timeout is not retried.
    Timeout,
    /// The process died, closed its output or answered more than 16 MiB, twice in a row.
    Failed,
    /// ExifTool printed nothing: it found no file or could not open it.
    EmptyOutput,
    /// ExifTool's answer was not a JSON list of one object per file, twice in a row.
    BadOutput,
    /// The path cannot be sent: it is empty, holds a line break or a NUL byte, or starts with `-`
    /// or `#`, which ExifTool would read as an option or a comment.
    InvalidPath,
    /// Every process is busy and the read may not wait (any longer).
    Busy(Busy),
    /// The pool was closed.
    Closed,
}

/// Why a read was refused for load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Busy {
    /// `OP_EXIFTOOL_MAX_WAITING` reads were already waiting.
    QueueFull,
    /// No process became free within `OP_EXIFTOOL_QUEUE_WAIT_S`.
    WaitedTooLong,
}

impl std::fmt::Display for ExifToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Start { binary } => write!(
                f,
                "exiftool could not be started from {BIN_KEY}={binary:?} (missing, not ExifTool 12.15 or newer, or no answer to -ver; setpriv from util-linux is also needed)"
            ),
            Self::Timeout => write!(f, "exiftool did not answer within {READ_TIMEOUT_KEY}"),
            Self::Failed => write!(f, "exiftool died or answered too much, twice"),
            Self::EmptyOutput => write!(f, "exiftool printed nothing for the file"),
            Self::BadOutput => write!(f, "exiftool's answer was not the expected JSON, twice"),
            Self::InvalidPath => write!(f, "the path cannot be sent to exiftool"),
            Self::Busy(Busy::QueueFull) => {
                write!(
                    f,
                    "exiftool is busy and {MAX_WAITING_KEY} reads already wait"
                )
            }
            Self::Busy(Busy::WaitedTooLong) => {
                write!(f, "no exiftool process became free within {QUEUE_WAIT_KEY}")
            }
            Self::Closed => write!(f, "the exiftool pool is closed"),
        }
    }
}

impl std::error::Error for ExifToolError {}

/// The pool. Share it between tasks behind an `Arc`.
pub struct ExifToolPool {
    settings: ExifToolSettings,
    /// Processes between reads. The processes are this server process's own resource, not shared
    /// state; the lock is held only to move one in or out.
    idle: Mutex<Vec<Child>>,
    /// One permit per process the pool may have running.
    processes: Semaphore,
    /// One permit per read that may be running or waiting.
    admission: Semaphore,
}

impl ExifToolPool {
    /// Start `settings.pool_size` processes now, so a missing or wrong ExifTool fails at start-up
    /// and not at the first photo. Call it from the runtime's own threads (see the module
    /// documentation of the child process).
    ///
    /// # Errors
    ///
    /// [`ExifToolError::Start`] when a process cannot start; the ones already started are killed.
    pub async fn start(settings: ExifToolSettings) -> Result<Self, ExifToolError> {
        let mut started = Vec::with_capacity(settings.pool_size);
        for _ in 0..settings.pool_size {
            started.push(Self::spawn(&settings).await?);
        }
        Ok(Self {
            processes: Semaphore::new(settings.pool_size),
            admission: Semaphore::new(settings.pool_size + settings.max_waiting),
            idle: Mutex::new(started),
            settings,
        })
    }

    async fn spawn(settings: &ExifToolSettings) -> Result<Child, ExifToolError> {
        Child::start(settings.binary.as_os_str())
            .await
            .map_err(|_| ExifToolError::Start {
                binary: settings.binary.display().to_string(),
            })
    }

    /// Read the tags `tags` (for example `EXIF:Orientation`) of the file at `path`.
    ///
    /// The request is `-j`, `-fast`, one `-<tag>` per tag and the path, then `-execute<n>`; the
    /// answer is the first object of the JSON list ExifTool prints, an empty map for an empty list.
    /// A process that dies or garbles its answer is replaced and the read tried once more on the
    /// fresh process; a timeout is not retried. If the future is dropped while a read runs, the
    /// process of that read is killed with it.
    ///
    /// # Errors
    ///
    /// See [`ExifToolError`].
    pub async fn read(&self, path: &Path, tags: &[&str]) -> Result<Tags, ExifToolError> {
        let request = request_lines(path, tags)?;
        let lines: Vec<&[u8]> = request.iter().map(Vec::as_slice).collect();
        let _admitted = self.admission.try_acquire().map_err(|e| match e {
            TryAcquireError::NoPermits => ExifToolError::Busy(Busy::QueueFull),
            TryAcquireError::Closed => ExifToolError::Closed,
        })?;
        let _process = match timeout(self.settings.queue_wait, self.processes.acquire()).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => return Err(ExifToolError::Closed),
            Err(_) => return Err(ExifToolError::Busy(Busy::WaitedTooLong)),
        };
        let mut last = ExifToolError::Failed;
        for _ in 0..2 {
            let mut child = self.checkout().await?;
            match child.execute(&lines, self.settings.read_timeout).await {
                Ok(answer) => match parse_answer(&answer) {
                    Ok(tags) => {
                        self.give_back(child).await;
                        return Ok(tags);
                    }
                    Err(ExifToolError::EmptyOutput) => {
                        self.give_back(child).await;
                        return Err(ExifToolError::EmptyOutput);
                    }
                    Err(error) => last = error,
                },
                Err(ChildError::Timeout) => {
                    child.discard().await;
                    tracing::warn!("exiftool read timed out; the process was killed");
                    return Err(ExifToolError::Timeout);
                }
                Err(_) => last = ExifToolError::Failed,
            }
            child.discard().await;
            tracing::warn!(error = %last, "exiftool read failed; the process was replaced");
        }
        Err(last)
    }

    /// A running process: an idle one, else a new one (the pool never has more than its size, the
    /// permit the caller holds says so).
    async fn checkout(&self) -> Result<Child, ExifToolError> {
        loop {
            let candidate = self
                .idle
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop();
            let Some(mut child) = candidate else {
                return Self::spawn(&self.settings).await;
            };
            if child.has_exited() {
                child.discard().await;
            } else {
                return Ok(child);
            }
        }
    }

    async fn give_back(&self, child: Child) {
        if self.processes.is_closed() {
            child.quit().await;
        } else {
            self.idle
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(child);
        }
    }

    /// Stop the pool: new and waiting reads get [`ExifToolError::Closed`], each idle process is
    /// asked to quit and reaped (killed after five seconds), and a process still in a read is
    /// reaped when that read ends.
    pub async fn close(&self) {
        self.processes.close();
        self.admission.close();
        let idle = std::mem::take(&mut *self.idle.lock().unwrap_or_else(PoisonError::into_inner));
        for child in idle {
            child.quit().await;
        }
    }
}

/// The lines of one read: the arguments of `src/clients/exiftool_client.py`.
fn request_lines(path: &Path, tags: &[&str]) -> Result<Vec<Vec<u8>>, ExifToolError> {
    let bytes = path.as_os_str().as_bytes();
    let unsafe_start = matches!(bytes.first(), Some(b'-' | b'#'));
    if bytes.is_empty() || unsafe_start || bytes.iter().any(|b| matches!(b, b'\n' | b'\r' | 0)) {
        return Err(ExifToolError::InvalidPath);
    }
    let mut lines: Vec<Vec<u8>> = vec![b"-j".to_vec(), b"-fast".to_vec()];
    lines.extend(tags.iter().map(|tag| format!("-{tag}").into_bytes()));
    lines.push(bytes.to_vec());
    Ok(lines)
}

/// The first object of ExifTool's JSON list.
fn parse_answer(answer: &[u8]) -> Result<Tags, ExifToolError> {
    if answer.iter().all(u8::is_ascii_whitespace) {
        return Err(ExifToolError::EmptyOutput);
    }
    match serde_json::from_slice::<Value>(answer) {
        Ok(Value::Array(rows)) => match rows.into_iter().next() {
            None => Ok(Tags::new()),
            Some(Value::Object(tags)) => Ok(tags),
            Some(_) => Err(ExifToolError::BadOutput),
        },
        _ => Err(ExifToolError::BadOutput),
    }
}
