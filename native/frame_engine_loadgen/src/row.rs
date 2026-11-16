//! One result row: a single JSON line holding the number and everything a reader needs to
//! reproduce it (plan 17 section 5, "Evidence for the paper": the set, the hardware, the load
//! on the host, the exact command, the models' counters).

use std::collections::BTreeMap;

use serde::Serialize;

use crate::arith::ModelWindow;
use crate::cgroup::{CpuWindow, MemoryRead};
use crate::gpu::{GpuWindow, Identity, ProcessLabel};
use crate::host::CpuInfo;

/// The row's schema name; a change of any field's meaning changes it.
pub const SCHEMA: &str = "op-fe-loadgen/1";

/// The tool and build that wrote the row.
#[derive(Debug, Serialize)]
pub struct Loadgen {
    /// The crate version of the load generator.
    pub version: &'static str,
    /// `OP_BUILD_SHA` at build time (the image build passes the commit), when set.
    pub build_sha: Option<&'static str>,
}

/// The host the row was measured on.
#[derive(Debug, Serialize)]
pub struct HostEvidence {
    /// The host name; empty when unreadable.
    pub hostname: String,
    /// The host's CPU.
    pub cpu: CpuInfo,
    /// 1-minute load average when the run started (before the warm-up) and when it ended.
    pub load1_start: Option<f64>,
    /// 1-minute load average when the run ended; null when unreadable.
    pub load1_end: Option<f64>,
}

/// The photo list that was sent.
#[derive(Debug, Serialize)]
pub struct ListEvidence {
    /// The list file's path as given.
    pub path: String,
    /// SHA-256 of the list file, lowercase hexadecimal.
    pub sha256: String,
    /// Photos in the list.
    pub photos: usize,
    /// Summed size of the photo files, in bytes.
    pub bytes: u64,
}

/// The plan every photo carried.
#[derive(Debug, Serialize)]
pub struct PlanEvidence {
    /// The plan file's path as given.
    pub path: String,
    /// SHA-256 of `text`, lowercase hexadecimal.
    pub sha256: String,
    /// The plan JSON exactly as sent.
    pub text: String,
}

/// How the engine was reached.
#[derive(Debug, Serialize)]
pub struct EngineEvidence {
    /// The engine's gRPC address as given.
    pub address: String,
    /// HTTP/2 connections opened to the engine.
    pub channels: usize,
    /// Seconds each photo may take before it counts as `client:timeout`.
    pub timeout_s: f64,
}

/// The measured process's CPU and memory, from its cgroup.
#[derive(Debug, Serialize)]
pub struct CpuEvidence {
    /// The `cpu.stat` file read: the process tree charged.
    pub cgroup_stat_path: String,
    /// CPU used inside the window; null when `error` is set.
    pub window: Option<CpuWindow>,
    /// The cgroup's memory counters when the window ended.
    pub memory_at_end: MemoryRead,
    /// Why there is no window (an unreadable file, a recreated cgroup).
    pub error: Option<String>,
}

/// The card's identity and what it did inside the window.
#[derive(Debug, Serialize)]
pub struct GpuEvidence {
    /// The card and its PCIe link.
    pub identity: Identity,
    /// What the card did inside the window; null when no sample fell inside it.
    pub window: Option<GpuWindow>,
    /// NVML's host process ids of the window's processes, with their command and cgroup.
    pub process_labels: Vec<ProcessLabel>,
    /// Sampler reads that failed during the run, as NVML's messages; empty when all succeeded.
    pub read_errors: Vec<String>,
}

/// The model server's per-model counters across the window.
#[derive(Debug, Serialize)]
pub struct ModelServerEvidence {
    /// The model server's gRPC address as given.
    pub url: String,
    /// Every model that executed inside the window, by name.
    pub models: BTreeMap<String, ModelWindow>,
    /// Models whose counters went backwards inside the window (a restart): not reported.
    pub reset_models: Vec<String>,
    /// Why the counters are missing (a failed read); null when they were read.
    pub error: Option<String>,
}

/// What the models-alone mode replayed.
#[derive(Debug, Serialize)]
pub struct ZEvidence {
    /// Path of the crop-count file; null when the plan embeds no crops.
    pub crop_counts_path: Option<String>,
    /// SHA-256 of the list the crop counts were recorded for; null with `crop_counts_path`.
    pub crop_counts_list_sha256: Option<String>,
    /// Size of the CUDA input region, in MiB.
    pub region_mib: usize,
    /// Name the region was registered under on the model server.
    pub region_name: String,
    /// Model calls per photo on average (detector + frame encoder + crop chunks).
    pub mean_calls_per_photo: f64,
    /// Crops per photo on average across the list.
    pub mean_crops_per_photo: f64,
}

/// One result row: one JSON line per measured run.
#[derive(Debug, Serialize)]
pub struct Row {
    /// The row's schema name, `SCHEMA`.
    pub schema: &'static str,
    /// `engine` for a measured engine, `z` for the models alone.
    pub mode: &'static str,
    /// The arm's name from `--label`; empty when not given.
    pub label: String,
    /// The command line the run was started with.
    pub command: Vec<String>,
    /// The load generator that wrote the row.
    pub loadgen: Loadgen,
    /// UTC time the run started (before the warm-up), ISO 8601 with milliseconds.
    pub started_at: String,
    /// UTC time the run ended, ISO 8601 with milliseconds.
    pub ended_at: String,
    /// Unix time in milliseconds when the window opened.
    pub window_start_unix_ms: u64,
    /// Unix time in milliseconds when the window closed.
    pub window_end_unix_ms: u64,
    /// The host the run was measured on.
    pub host: HostEvidence,
    /// The photo list sent.
    pub list: ListEvidence,
    /// The plan every photo carried.
    pub plan: PlanEvidence,
    /// The engine reached; null in `z` mode.
    pub engine: Option<EngineEvidence>,
    /// The models-alone replay; null in `engine` mode.
    pub z: Option<ZEvidence>,
    /// Photos outstanding at all times.
    pub in_flight: usize,
    /// Seconds of unmeasured warm-up asked for.
    pub warmup_s: f64,
    /// Seconds of measured window asked for.
    pub window_s: f64,
    /// Photos answered `ok` inside the window.
    pub photos: usize,
    /// Photos that failed inside the window; the sum of `failures_by_status`.
    pub failed: u64,
    /// Failures inside the window by key: `grpc:<CODE>`, `status:<photo status>`, `branch:<name>:<reason>`, `protocol:<what>` or `client:timeout`.
    pub failures_by_status: BTreeMap<String, u64>,
    /// Photos that failed during the warm-up (reported, not measured).
    pub warmup_failed: u64,
    /// Photos that completed after the window closed (not counted in the rate).
    pub after_window: u64,
    /// The window's length on the loop's own clock, in seconds; the rate is `photos` over this.
    pub window_seconds_measured: f64,
    /// Photos answered `ok` per second of the measured window.
    pub photos_per_s: f64,
    /// Median latency of the `ok` photos, in milliseconds.
    pub p50_ms: f64,
    /// 99th percentile latency, in milliseconds.
    pub p99_ms: f64,
    /// Mean latency, in milliseconds.
    pub mean_ms: f64,
    /// Largest latency, in milliseconds.
    pub max_ms: f64,
    /// `ok` photos by the decode route the engine reported.
    pub decode_routes: BTreeMap<String, u64>,
    /// Mean size of an `ok` answer on the wire, in bytes.
    pub mean_response_bytes: f64,
    /// The measured process's CPU; null without `--cpu-stat`.
    pub cpu: Option<CpuEvidence>,
    /// The card sampled; null with `--no-gpu`.
    pub gpu: Option<GpuEvidence>,
    /// The model server's counters; null with `--no-stats`.
    pub model_server: Option<ModelServerEvidence>,
}
