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

#[derive(Debug, Serialize)]
pub struct Loadgen {
    pub version: &'static str,
    /// `OP_BUILD_SHA` at build time (the image build passes the commit), when set.
    pub build_sha: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct HostEvidence {
    pub hostname: String,
    pub cpu: CpuInfo,
    /// 1-minute load average when the run started (before the warm-up) and when it ended.
    pub load1_start: Option<f64>,
    pub load1_end: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct ListEvidence {
    pub path: String,
    pub sha256: String,
    pub photos: usize,
    pub bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct PlanEvidence {
    pub path: String,
    pub sha256: String,
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct EngineEvidence {
    pub address: String,
    pub channels: usize,
    pub timeout_s: f64,
}

#[derive(Debug, Serialize)]
pub struct CpuEvidence {
    /// The `cpu.stat` file read: the process tree charged.
    pub cgroup_stat_path: String,
    pub window: Option<CpuWindow>,
    pub memory_at_end: MemoryRead,
    /// Why there is no window (an unreadable file, a recreated cgroup).
    pub error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GpuEvidence {
    pub identity: Identity,
    pub window: Option<GpuWindow>,
    /// NVML's host process ids of the window's processes, with their command and cgroup.
    pub process_labels: Vec<ProcessLabel>,
    pub read_errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelServerEvidence {
    pub url: String,
    pub models: BTreeMap<String, ModelWindow>,
    /// Models whose counters went backwards inside the window (a restart): not reported.
    pub reset_models: Vec<String>,
    pub error: Option<String>,
}

/// What the models-alone mode replayed.
#[derive(Debug, Serialize)]
pub struct ZEvidence {
    pub crop_counts_path: Option<String>,
    pub crop_counts_list_sha256: Option<String>,
    pub region_mib: usize,
    pub region_name: String,
    /// Model calls per photo on average (detector + frame encoder + crop chunks).
    pub mean_calls_per_photo: f64,
    pub mean_crops_per_photo: f64,
}

#[derive(Debug, Serialize)]
pub struct Row {
    pub schema: &'static str,
    pub mode: &'static str,
    pub label: String,
    pub command: Vec<String>,
    pub loadgen: Loadgen,
    pub started_at: String,
    pub ended_at: String,
    pub window_start_unix_ms: u64,
    pub window_end_unix_ms: u64,
    pub host: HostEvidence,
    pub list: ListEvidence,
    pub plan: PlanEvidence,
    pub engine: Option<EngineEvidence>,
    pub z: Option<ZEvidence>,
    pub in_flight: usize,
    pub warmup_s: f64,
    pub window_s: f64,
    /// Photos answered `ok` inside the window.
    pub photos: usize,
    /// Photos that failed inside the window; the sum of `failures_by_status`.
    pub failed: u64,
    pub failures_by_status: BTreeMap<String, u64>,
    pub warmup_failed: u64,
    /// Photos that completed after the window closed (not counted in the rate).
    pub after_window: u64,
    pub window_seconds_measured: f64,
    pub photos_per_s: f64,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub max_ms: f64,
    pub decode_routes: BTreeMap<String, u64>,
    pub mean_response_bytes: f64,
    pub cpu: Option<CpuEvidence>,
    pub gpu: Option<GpuEvidence>,
    pub model_server: Option<ModelServerEvidence>,
}
