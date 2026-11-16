//! One result row: a single JSON line holding the number and what a reader needs to reproduce it
//! (plan 17 section 5, "Evidence for the paper": the set, the hardware, the load on the host, the
//! exact command, the bound that remains).

use serde::Serialize;

use frame_engine_loadgen::cgroup::{self, Cgroup, CpuStat};
use frame_engine_loadgen::row::{CpuEvidence, HostEvidence, ListEvidence, Loadgen};

use crate::multipart::FieldOrder;
use crate::process::ProcessWindow;
use crate::summary::Summary;

/// The row's schema name; a change of any field's meaning changes it.
pub const SCHEMA: &str = "op-upload-loadgen/1";

/// What the stamps mean and how far they can be trusted: copied into every row so a number never
/// travels without its bound.
pub const BOUNDS: [&str; 3] = [
    "part_taken_ms: milliseconds from the request's start to the instant hyper's HTTP/1 connection asked the body for the frame after a photo part's last frame; the connection buffers up to 16 chunks (and the kernel its socket buffers), so this is a client-side bound that is earlier than the server receiving the part, not a server-side stamp",
    "request_ms: request start to the end of the answer's body, read on the same clock; client side, includes the network and the server's whole handling",
    "tail_after_last_part_ms: request_ms minus the last part's stamp: the server's work after the last photo was handed over, plus the transfer of the bytes still buffered",
];

/// The upload settings the row was measured with.
#[derive(Debug, Serialize)]
pub struct UploadEvidence {
    /// The upload route's URL.
    pub url: String,
    /// The HTTP version spoken.
    pub http_version: &'static str,
    /// Idle connections the pool may keep to the host (the in-flight count).
    pub pool_max_idle_per_host: usize,
    /// `browser` or `legacy`.
    pub field_order: &'static str,
    /// The `source` field.
    pub source: String,
    /// The `run_id` field (not sent in the legacy order).
    pub run_id: String,
    /// The most photos in one request.
    pub max_images: usize,
    /// The most photo bytes in one request.
    pub max_bytes: u64,
    /// The largest frame of a photo part, in bytes.
    pub chunk_bytes: usize,
    /// One request's time limit, in seconds.
    pub timeout_s: f64,
    /// Requests that make one pass over the list.
    pub packs_per_pass: usize,
    /// Mean photos per request over one pass.
    pub mean_photos_per_pack: f64,
}

/// A named container's CPU: the cgroup's `cpu.stat` read at the window's edges.
#[derive(Debug, Serialize)]
pub struct NamedCpu {
    /// The name given on the command line.
    pub name: String,
    /// The cgroup's readings.
    #[serde(flatten)]
    pub evidence: CpuEvidence,
}

/// Where the stamps went and what they mean.
#[derive(Debug, Serialize)]
pub struct StampsEvidence {
    /// The per-request, per-part stamps file (JSON lines).
    pub path: String,
    /// Why the figures are bounds; see [`BOUNDS`].
    pub bounds: [&'static str; 3],
}

/// One result row: one JSON line per measured run.
#[derive(Debug, Serialize)]
pub struct Row {
    /// The row's schema name, [`SCHEMA`].
    pub schema: &'static str,
    /// The arm's name from `--label`; empty when not given.
    pub label: String,
    /// The command line the run was started with.
    pub command: Vec<String>,
    /// The load generator that wrote the row.
    pub loadgen: Loadgen,
    /// UTC time the run started, ISO 8601 with milliseconds.
    pub started_at: String,
    /// UTC time the run ended.
    pub ended_at: String,
    /// Unix time in milliseconds when the window opened.
    pub window_start_unix_ms: u64,
    /// Unix time in milliseconds when the window closed.
    pub window_end_unix_ms: u64,
    /// The host the run was measured on.
    pub host: HostEvidence,
    /// The photo list sent.
    pub list: ListEvidence,
    /// How the uploads were sent.
    pub upload: UploadEvidence,
    /// Requests outstanding at all times.
    pub in_flight: usize,
    /// Seconds of unmeasured warm-up asked for.
    pub warmup_s: f64,
    /// Seconds of measured window asked for.
    pub window_s: f64,
    /// Requests started in the whole run.
    pub requests_started: u64,
    /// Photos sent in the whole run, divided by the list's photos: above 1 means photos were sent
    /// more than once, and a server that deduplicates by content answers the repeats `duplicate`.
    pub list_cycles_sent: f64,
    /// The window's figures.
    #[serde(flatten)]
    pub summary: Summary,
    /// The per-part stamps file.
    pub stamps: StampsEvidence,
    /// This process's CPU in the window; null when `/proc/self/stat` was unreadable.
    pub loadgen_cpu: Option<ProcessWindow>,
    /// The named containers' CPU, in the order given.
    pub cpu: Vec<NamedCpu>,
    /// Free text from `--server-note`: the server's build, image digests, settings, as given.
    pub server_note: Option<String>,
}

/// The name of a field order as it is written in a row and on the command line.
pub fn order_name(order: FieldOrder) -> &'static str {
    match order {
        FieldOrder::Browser => "browser",
        FieldOrder::Legacy => "legacy",
    }
}

/// A cgroup's CPU in the window from its two reads.
pub fn cpu_evidence(
    cg: &Cgroup,
    before: &Result<CpuStat, String>,
    after: &Result<CpuStat, String>,
    seconds: f64,
    photos: usize,
) -> CpuEvidence {
    let (window, error) = match (before, after) {
        (Ok(a), Ok(b)) => match cgroup::window(a, b, seconds, photos) {
            Some(w) => (Some(w), None),
            None => (
                None,
                Some("cpu.stat went backwards: the cgroup was recreated".to_string()),
            ),
        },
        (Err(e), _) | (_, Err(e)) => (None, Some(e.clone())),
    };
    CpuEvidence {
        cgroup_stat_path: cg.path().display().to_string(),
        window,
        memory_at_end: cg.read_memory(),
        error,
    }
}
