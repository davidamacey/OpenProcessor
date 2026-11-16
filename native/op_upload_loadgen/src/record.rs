//! What one request leaves behind, and the stamps file those records are written to.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::Serialize;

/// How one request ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The server answered 2xx.
    Answered {
        /// Photos the answer reports as `success`.
        success: u64,
        /// Photos it reports as `duplicate` (already stored: the server did no new work).
        duplicate: u64,
        /// Photos it reports as `failed`, by their `error_kind` (`unknown` when absent).
        failed_by_kind: BTreeMap<String, u64>,
        /// False when the answer carried no per-photo `results` (a sink, or another body): the
        /// request's photos are then counted as unread, not as success.
        read: bool,
    },
    /// No usable answer, or a refusal, under a key a reader can group by: `http:<status>`,
    /// `client:timeout`, `client:connect`, `client:send`, `client:response_body`,
    /// `protocol:results_count`.
    Failed {
        /// The group key.
        key: String,
    },
}

/// One request, from the instant it was handed to the connection pool to its last response byte.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestRecord {
    /// The request's number in the run, from 0.
    pub seq: u64,
    /// Which pack of the list it carried.
    pub pack: usize,
    /// Photos in the request.
    pub photos: usize,
    /// Summed size of the photos, in bytes.
    pub photo_bytes: u64,
    /// The whole body's length, in bytes.
    pub body_bytes: u64,
    /// Milliseconds from the loop's start to the request's start.
    pub started_ms: f64,
    /// Milliseconds from the loop's start to the end of the answer (or of the failure).
    pub finished_ms: f64,
    /// Per photo part, milliseconds from the request's start to the instant the connection took
    /// the part's last frame (a client-side bound; fewer entries than photos when it failed early).
    pub parts_taken_ms: Vec<f64>,
    /// Milliseconds from the request's start to its end.
    pub response_ms: f64,
    /// How it ended.
    pub outcome: Outcome,
}

/// One line of the stamps file.
#[derive(Debug, Serialize)]
struct StampLine<'a> {
    seq: u64,
    pack: usize,
    photos: usize,
    photo_bytes: u64,
    body_bytes: u64,
    phase: &'a str,
    started_ms: f64,
    finished_ms: f64,
    response_ms: f64,
    outcome: &'a str,
    parts_taken_ms: &'a [f64],
}

/// Write one JSON line per request to `path` (replacing the file): the per-part stamps, the
/// response time and the outcome, with the phase each request finished in (`warmup`, `window` or
/// `after`, as `phase_of` says for its finish time).
///
/// # Errors
///
/// Returns `Err` when the file cannot be created or written.
pub fn write_stamps(
    path: &Path,
    records: &[RequestRecord],
    phase_of: impl Fn(f64) -> &'static str,
) -> io::Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    for r in records {
        let outcome = match &r.outcome {
            Outcome::Answered { .. } => "ok",
            Outcome::Failed { key } => key.as_str(),
        };
        let line = StampLine {
            seq: r.seq,
            pack: r.pack,
            photos: r.photos,
            photo_bytes: r.photo_bytes,
            body_bytes: r.body_bytes,
            phase: phase_of(r.finished_ms),
            started_ms: r.started_ms,
            finished_ms: r.finished_ms,
            response_ms: r.response_ms,
            outcome,
            parts_taken_ms: &r.parts_taken_ms,
        };
        serde_json::to_writer(&mut out, &line).map_err(io::Error::from)?;
        out.write_all(b"\n")?;
    }
    out.flush()
}
