//! One long-lived `exiftool -stay_open True -@ -` process and the request framing of its protocol.
//!
//! ExifTool reads one argument per line from standard input; `-execute<n>` runs the arguments
//! read so far, and when it is done ExifTool prints `{ready<n>}` on its own line to standard
//! output (<https://exiftool.org/exiftool_pod.html>, the options `-stay_open` and `-execute[NUM]`).
//! A request is therefore answered by reading standard output up to the line `{ready<n>}`.
//! Standard error is drained by a task of its own, so ExifTool never blocks on a full pipe, and is
//! never logged: it can hold file names and coordinates.
//!
//! The process is started through `setpriv --pdeathsig TERM --` (util-linux, `setpriv(1)`): ExifTool
//! does not exit when its standard input closes, and the kernel sends the signal when the thread
//! that started it ends, so a server killed outright does not leave children behind. Start the
//! pool from the runtime's own threads (never from a short-lived `spawn_blocking` thread).

use std::ffi::OsStr;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;
use tokio::time::timeout;

/// The oldest ExifTool this driver accepts: Python's client requires the same.
const MIN_VERSION: (u32, u32) = (12, 15);
/// A larger answer is a fault, not a photo (ExifTool answers a few hundred bytes).
const MAX_OUTPUT: usize = 16 * 1024 * 1024;
const CHUNK: usize = 64 * 1024;
const QUIT_GRACE: Duration = Duration::from_secs(5);
const START_BOUND: Duration = Duration::from_secs(30);

/// Why one request failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChildError {
    /// The process could not be started.
    Spawn,
    /// It did not answer within the bound; it is killed.
    Timeout,
    /// It closed its output or a pipe broke (it died).
    Gone,
    /// The answer was larger than [`MAX_OUTPUT`].
    TooLarge,
    /// It started but is not ExifTool 12.15 or newer.
    NotExifTool,
}

/// A running ExifTool process.
pub(super) struct Child {
    process: tokio::process::Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    drain: JoinHandle<()>,
    next_id: u64,
}

impl Drop for Child {
    fn drop(&mut self) {
        self.drain.abort();
    }
}

impl Child {
    /// Start ExifTool and check it answers `-ver`.
    pub(super) async fn start(binary: &OsStr) -> Result<Self, ChildError> {
        let mut command = Command::new("setpriv");
        command
            .args(["--pdeathsig", "TERM", "--"])
            .arg(binary)
            .args(["-stay_open", "True", "-@", "-", "-common_args", "-G", "-n"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let mut process = command.spawn().map_err(|_| ChildError::Spawn)?;
        let (Some(stdin), Some(stdout), Some(mut stderr)) = (
            process.stdin.take(),
            process.stdout.take(),
            process.stderr.take(),
        ) else {
            return Err(ChildError::Spawn);
        };
        let drain = tokio::spawn(async move {
            let mut sink = [0_u8; 8192];
            while stderr.read(&mut sink).await.is_ok_and(|n| n > 0) {}
        });
        let mut child = Self {
            process,
            stdin,
            stdout,
            drain,
            next_id: 1,
        };
        let answer = child.execute(&[b"-ver"], START_BOUND).await?;
        if parse_version(&answer).is_some_and(|v| v >= MIN_VERSION) {
            Ok(child)
        } else {
            Err(ChildError::NotExifTool)
        }
    }

    /// Whether the process has already exited (so it must not be reused).
    pub(super) fn has_exited(&mut self) -> bool {
        !matches!(self.process.try_wait(), Ok(None))
    }

    /// Send one request (`lines` are its arguments, one per line) and return what ExifTool wrote to
    /// standard output before `{ready<n>}`. The whole exchange is bounded by `bound`.
    pub(super) async fn execute(
        &mut self,
        lines: &[&[u8]],
        bound: Duration,
    ) -> Result<Vec<u8>, ChildError> {
        let id = self.next_id;
        self.next_id += 1;
        let mut request = Vec::new();
        for line in lines {
            request.extend_from_slice(line);
            request.push(b'\n');
        }
        request.extend_from_slice(format!("-execute{id}\n").as_bytes());
        let marker = format!("{{ready{id}}}\n").into_bytes();
        timeout(bound, async {
            self.stdin
                .write_all(&request)
                .await
                .map_err(|_| ChildError::Gone)?;
            self.stdin.flush().await.map_err(|_| ChildError::Gone)?;
            read_until_marker(&mut self.stdout, &marker).await
        })
        .await
        .unwrap_or(Err(ChildError::Timeout))
    }

    /// Kill the process and reap it, so no zombie is left.
    pub(super) async fn discard(mut self) {
        // The error says the process was already gone, which is what was wanted.
        let _ = self.process.kill().await;
    }

    /// Ask ExifTool to quit (`-stay_open False`), wait a few seconds, then kill and reap it.
    pub(super) async fn quit(mut self) {
        let asked = async {
            self.stdin.write_all(b"-stay_open\nFalse\n").await?;
            self.stdin.flush().await
        }
        .await
        .is_ok();
        if asked && timeout(QUIT_GRACE, self.process.wait()).await.is_ok() {
            return;
        }
        self.discard().await;
    }
}

async fn read_until_marker(stdout: &mut ChildStdout, marker: &[u8]) -> Result<Vec<u8>, ChildError> {
    let mut answer = Vec::new();
    let mut chunk = vec![0_u8; CHUNK];
    loop {
        let n = stdout
            .read(&mut chunk)
            .await
            .map_err(|_| ChildError::Gone)?;
        if n == 0 {
            return Err(ChildError::Gone);
        }
        answer.extend_from_slice(&chunk[..n]);
        if answer.ends_with(marker) {
            answer.truncate(answer.len() - marker.len());
            return Ok(answer);
        }
        if answer.len() > MAX_OUTPUT {
            return Err(ChildError::TooLarge);
        }
    }
}

/// `12.76` as `(12, 76)`.
fn parse_version(answer: &[u8]) -> Option<(u32, u32)> {
    let text = std::str::from_utf8(answer).ok()?.trim();
    let (major, minor) = text.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_are_read_and_compared() {
        assert_eq!(parse_version(b"12.76\n"), Some((12, 76)));
        assert_eq!(parse_version(b"13.00"), Some((13, 0)));
        assert!(parse_version(b"12.15").unwrap() >= MIN_VERSION);
        assert!(parse_version(b"12.14").unwrap() < MIN_VERSION);
        assert!(parse_version(b"11.99").unwrap() < MIN_VERSION);
        assert_eq!(parse_version(b"not a version"), None);
        assert_eq!(parse_version(b""), None);
        assert_eq!(parse_version(&[0xff, 0xfe]), None);
    }
}
