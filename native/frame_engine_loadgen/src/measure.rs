//! One measured row, whichever exchange fills it: the closed loop, the counters read at the
//! window's edges, and the row built from both.
//!
//! At the window's open the model server's statistics are read, then `cpu.stat`, then the
//! instant is marked; at its close the instant is marked, then `cpu.stat`, then the statistics.
//! So the CPU window lies inside the rate's window and the statistics window outside it by one
//! request's latency (milliseconds against a window of tens of seconds).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::arith::{self, Counters};
use crate::cgroup::{self, Cgroup, CpuStat};
use crate::client::Client;
use crate::exchange::{Exchange, LoopConfig, LoopResult, Probe, run};
use crate::gpu::{self, Identity, Sampler};
use crate::host;
use crate::list::PhotoList;
use crate::plan::PlanFile;
use crate::row::{
    CpuEvidence, EngineEvidence, GpuEvidence, HostEvidence, ListEvidence, Loadgen,
    ModelServerEvidence, PlanEvidence, Row, SCHEMA, ZEvidence,
};
use crate::statistics;

/// The card's sampler and what names the card.
pub struct Gpu {
    pub sampler: Sampler,
    pub identity: Identity,
}

/// Everything a row is made of besides the loop's result.
pub struct Setup {
    pub mode: &'static str,
    pub label: String,
    pub command: Vec<String>,
    pub list: PhotoList,
    pub plan: PlanFile,
    pub plan_path: String,
    pub in_flight: usize,
    pub warmup: Duration,
    pub window: Duration,
    /// The cgroup of the measured process: the engine, or in `z` mode the model server.
    pub cgroup: Option<Cgroup>,
    /// The model server's address and a channel to it, for the statistics deltas.
    pub stats: Option<(String, Client)>,
    pub gpu: Option<Gpu>,
    pub engine: Option<EngineEvidence>,
    pub z: Option<ZEvidence>,
}

const STATS_TIMEOUT: Duration = Duration::from_secs(10);

/// The counters read at one edge.
#[derive(Default)]
struct Reads {
    at: Option<Instant>,
    unix_ms: u64,
    cpu: Option<Result<CpuStat, String>>,
    counters: Option<Result<BTreeMap<String, Counters>, String>>,
}

struct Edges<'a> {
    cgroup: Option<&'a Cgroup>,
    stats: Option<Client>,
    open: Reads,
    close: Reads,
}

impl Edges<'_> {
    fn mark(r: &mut Reads) {
        r.at = Some(Instant::now());
        r.unix_ms = host::unix_ms();
    }

    fn read_cpu(&self, r: &mut Reads) {
        r.cpu = self.cgroup.map(|c| {
            c.read_cpu()
                .map_err(|e| format!("{}: {e}", c.path().display()))
        });
    }

    async fn read_stats(&mut self, r: &mut Reads) {
        if let Some(client) = &mut self.stats {
            r.counters = Some(statistics::fetch(client, STATS_TIMEOUT).await);
        }
    }
}

impl Probe for Edges<'_> {
    async fn window_open(&mut self) {
        let mut r = Reads::default();
        self.read_stats(&mut r).await;
        self.read_cpu(&mut r);
        Self::mark(&mut r);
        self.open = r;
    }

    async fn window_close(&mut self) {
        let mut r = Reads::default();
        Self::mark(&mut r);
        self.read_cpu(&mut r);
        self.read_stats(&mut r).await;
        self.close = r;
    }
}

/// Run the loop and build the row.
pub async fn measure<E: Exchange>(setup: Setup, exchange: Arc<E>) -> Row {
    let started = host::unix_ms();
    let load_start = host::load1();
    let mut edges = Edges {
        cgroup: setup.cgroup.as_ref(),
        stats: setup.stats.as_ref().map(|(_, c)| c.clone()),
        open: Reads::default(),
        close: Reads::default(),
    };
    let config = LoopConfig {
        in_flight: setup.in_flight,
        warmup: setup.warmup,
        window: setup.window,
        list_len: setup.list.photos.len(),
    };
    let result = run(exchange, config, &mut edges).await;
    let load_end = host::load1();
    let ended = host::unix_ms();
    let (open, close) = (edges.open, edges.close);
    build_row(
        &setup,
        &result,
        &open,
        &close,
        (load_start, load_end),
        (started, ended),
    )
}

fn gpu_evidence(g: &Gpu, open: &Reads, close: &Reads) -> GpuEvidence {
    let window = open
        .at
        .zip(close.at)
        .and_then(|(from, to)| g.sampler.window(from, to));
    let process_labels = window
        .iter()
        .flat_map(|w| w.processes.iter().map(|p| gpu::label(p.pid)))
        .collect();
    GpuEvidence {
        identity: g.identity.clone(),
        window,
        process_labels,
        read_errors: g.sampler.read_errors(),
    }
}

fn host_evidence(load: (Option<f64>, Option<f64>)) -> HostEvidence {
    HostEvidence {
        hostname: host::hostname(),
        cpu: host::cpuinfo(),
        load1_start: load.0,
        load1_end: load.1,
    }
}

fn list_evidence(setup: &Setup) -> ListEvidence {
    ListEvidence {
        path: setup.list.path.display().to_string(),
        sha256: setup.list.sha256.clone(),
        photos: setup.list.photos.len(),
        bytes: setup.list.total_bytes,
    }
}

fn plan_evidence(setup: &Setup) -> PlanEvidence {
    PlanEvidence {
        path: setup.plan_path.clone(),
        sha256: setup.plan.sha256.clone(),
        text: setup.plan.text.clone(),
    }
}

fn engine_evidence(e: &EngineEvidence) -> EngineEvidence {
    EngineEvidence {
        address: e.address.clone(),
        channels: e.channels,
        timeout_s: e.timeout_s,
    }
}

fn z_evidence(z: &ZEvidence) -> ZEvidence {
    ZEvidence {
        crop_counts_path: z.crop_counts_path.clone(),
        crop_counts_list_sha256: z.crop_counts_list_sha256.clone(),
        region_mib: z.region_mib,
        region_name: z.region_name.clone(),
        mean_calls_per_photo: z.mean_calls_per_photo,
        mean_crops_per_photo: z.mean_crops_per_photo,
    }
}

fn build_row(
    setup: &Setup,
    result: &LoopResult,
    open: &Reads,
    close: &Reads,
    load: (Option<f64>, Option<f64>),
    run_ms: (u64, u64),
) -> Row {
    let mut latencies = result.latencies_ms.clone();
    let ok = result.ok();
    let seconds = result.window_seconds;
    let mean = if ok == 0 {
        0.0
    } else {
        latencies.iter().sum::<f64>() / ok as f64
    };
    let p50 = arith::percentile(&mut latencies, 50.0);
    let p99 = arith::percentile(&mut latencies, 99.0);
    let max = latencies.last().copied().unwrap_or(0.0);
    let cpu = setup
        .cgroup
        .as_ref()
        .map(|cg| cpu_evidence(cg, open, close, seconds, ok));
    let gpu = setup.gpu.as_ref().map(|g| gpu_evidence(g, open, close));
    let model_server = setup
        .stats
        .as_ref()
        .map(|(url, _)| server_evidence(url, open, close));
    Row {
        schema: SCHEMA,
        mode: setup.mode,
        label: setup.label.clone(),
        command: setup.command.clone(),
        loadgen: Loadgen {
            version: env!("CARGO_PKG_VERSION"),
            build_sha: option_env!("OP_BUILD_SHA"),
        },
        started_at: host::iso_utc(run_ms.0),
        ended_at: host::iso_utc(run_ms.1),
        window_start_unix_ms: open.unix_ms,
        window_end_unix_ms: close.unix_ms,
        host: host_evidence(load),
        list: list_evidence(setup),
        plan: plan_evidence(setup),
        engine: setup.engine.as_ref().map(engine_evidence),
        z: setup.z.as_ref().map(z_evidence),
        in_flight: setup.in_flight,
        warmup_s: setup.warmup.as_secs_f64(),
        window_s: setup.window.as_secs_f64(),
        photos: ok,
        failed: result.failed(),
        failures_by_status: result.failures.clone(),
        warmup_failed: result.warmup_failures,
        after_window: result.after_window,
        window_seconds_measured: seconds,
        photos_per_s: arith::rate(ok, seconds),
        p50_ms: p50,
        p99_ms: p99,
        mean_ms: mean,
        max_ms: max,
        decode_routes: result.routes.clone(),
        mean_response_bytes: if ok == 0 {
            0.0
        } else {
            result.response_bytes as f64 / ok as f64
        },
        cpu,
        gpu,
        model_server,
    }
}

fn cpu_evidence(
    cg: &Cgroup,
    open: &Reads,
    close: &Reads,
    seconds: f64,
    photos: usize,
) -> CpuEvidence {
    let (window, error) = match (&open.cpu, &close.cpu) {
        (Some(Ok(a)), Some(Ok(b))) => match cgroup::window(a, b, seconds, photos) {
            Some(w) => (Some(w), None),
            None => (
                None,
                Some("cpu.stat went backwards: the cgroup was recreated".to_string()),
            ),
        },
        (Some(Err(e)), _) | (_, Some(Err(e))) => (None, Some(e.clone())),
        _ => (None, Some("cpu.stat was not read".to_string())),
    };
    CpuEvidence {
        cgroup_stat_path: cg.path().display().to_string(),
        window,
        memory_at_end: cg.read_memory(),
        error,
    }
}

fn server_evidence(url: &str, open: &Reads, close: &Reads) -> ModelServerEvidence {
    match (&open.counters, &close.counters) {
        (Some(Ok(a)), Some(Ok(b))) => {
            let (models, reset_models) = arith::windows(a, b);
            ModelServerEvidence {
                url: url.to_string(),
                models,
                reset_models,
                error: None,
            }
        }
        (Some(Err(e)), _) | (_, Some(Err(e))) => ModelServerEvidence {
            url: url.to_string(),
            models: BTreeMap::new(),
            reset_models: Vec::new(),
            error: Some(e.clone()),
        },
        _ => ModelServerEvidence {
            url: url.to_string(),
            models: BTreeMap::new(),
            reset_models: Vec::new(),
            error: Some("statistics were not read".into()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::Counters;

    fn setup() -> Setup {
        Setup {
            mode: "engine",
            label: "arm-a".into(),
            command: vec!["op-fe-loadgen".into(), "engine".into()],
            list: PhotoList {
                path: "/lists/coco.txt".into(),
                sha256: "ab".repeat(32),
                photos: vec![bytes::Bytes::from_static(b"x"); 3],
                total_bytes: 3,
            },
            plan: PlanFile::parse(
                r#"{"version": 1, "branches": ["detect"], "detector": {"model": "d", "input_size": 640}}"#,
            )
            .unwrap(),
            plan_path: "/plans/frame.json".into(),
            in_flight: 32,
            warmup: Duration::from_secs(5),
            window: Duration::from_secs(20),
            cgroup: Some(Cgroup::new("/nonexistent/cpu.stat".into())),
            stats: None,
            gpu: None,
            engine: Some(EngineEvidence {
                address: "127.0.0.1:4621".into(),
                channels: 4,
                timeout_s: 120.0,
            }),
            z: None,
        }
    }

    fn reads(cpu_usec: u64, inf: u64, exec: u64) -> Reads {
        Reads {
            at: None,
            unix_ms: 1_794_745_845_000,
            cpu: Some(Ok(CpuStat {
                usage_usec: cpu_usec,
                ..CpuStat::default()
            })),
            counters: Some(Ok(BTreeMap::from([(
                "det".to_string(),
                Counters {
                    inference_count: inf,
                    execution_count: exec,
                    queue_count: inf,
                    queue_ns: inf * 1_000_000,
                    compute_infer_count: exec,
                    compute_infer_ns: exec * 4_000_000,
                },
            )]))),
        }
    }

    fn result() -> LoopResult {
        LoopResult {
            latencies_ms: vec![10.0, 20.0, 30.0, 40.0],
            failures: BTreeMap::from([("grpc:UNAVAILABLE".to_string(), 1)]),
            warmup_failures: 2,
            after_window: 3,
            routes: BTreeMap::from([("gpu".to_string(), 4)]),
            response_bytes: 4_000,
            window_seconds: 2.0,
        }
    }

    #[test]
    fn a_row_carries_every_field_the_evidence_clause_asks_for() {
        let mut s = setup();
        s.stats = None;
        let row = build_row(
            &s,
            &result(),
            &reads(1_000_000, 100, 10),
            &reads(3_000_000, 500, 20),
            (Some(1.5), Some(2.5)),
            (1_794_745_800_000, 1_794_745_900_000),
        );
        let v = serde_json::to_value(&row).unwrap();
        for key in [
            "schema",
            "mode",
            "label",
            "command",
            "loadgen",
            "started_at",
            "ended_at",
            "window_start_unix_ms",
            "window_end_unix_ms",
            "host",
            "list",
            "plan",
            "engine",
            "in_flight",
            "warmup_s",
            "window_s",
            "photos",
            "failed",
            "failures_by_status",
            "warmup_failed",
            "after_window",
            "photos_per_s",
            "p50_ms",
            "p99_ms",
            "decode_routes",
            "cpu",
            "gpu",
            "model_server",
            "z",
        ] {
            assert!(v.get(key).is_some(), "the row lacks {key}");
        }
        assert_eq!(v["list"]["sha256"], "ab".repeat(32));
        assert_eq!(v["host"]["load1_start"], 1.5);
        assert_eq!(v["engine"]["address"], "127.0.0.1:4621");
    }

    #[test]
    fn the_numbers_are_the_loops_counters_over_the_window() {
        let row = build_row(
            &setup(),
            &result(),
            &reads(1_000_000, 100, 10),
            &reads(3_000_000, 500, 20),
            (None, None),
            (0, 0),
        );
        // 4 photos in 2 s
        assert_eq!(row.photos_per_s, 2.0);
        assert_eq!(
            (row.photos, row.failed, row.warmup_failed, row.after_window),
            (4, 1, 2, 3)
        );
        assert!((row.p50_ms - 25.0).abs() < 1e-9);
        assert_eq!(row.max_ms, 40.0);
        assert_eq!(row.mean_response_bytes, 1000.0);
        // 2 CPU seconds over 2 s of wall time and 4 photos: one core, 500 ms a photo
        let w = row.cpu.unwrap().window.unwrap();
        assert!((w.cores_busy - 1.0).abs() < 1e-9);
        assert!((w.cpu_ms_per_photo - 500.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn model_server_deltas_come_from_the_two_reads() {
        let mut s = setup();
        s.stats = Some(("127.0.0.1:4620".into(), unreachable_client()));
        let row = build_row(
            &s,
            &result(),
            &reads(0, 100, 10),
            &reads(0, 500, 20),
            (None, None),
            (0, 0),
        );
        let m = row.model_server.unwrap();
        assert_eq!(m.url, "127.0.0.1:4620");
        let det = m.models["det"];
        assert_eq!((det.inferences, det.executions), (400, 10));
        assert!((det.mean_batch - 40.0).abs() < 1e-9);
        assert!((det.queue_ms_per_request - 1.0).abs() < 1e-9);
        assert!((det.compute_infer_ms_per_execution - 4.0).abs() < 1e-9);
    }

    #[test]
    fn a_cgroup_that_cannot_be_read_is_reported_not_hidden() {
        let mut open = reads(0, 0, 0);
        open.cpu = Some(Err("/x/cpu.stat: not found".into()));
        let row = build_row(
            &setup(),
            &result(),
            &open,
            &reads(0, 0, 0),
            (None, None),
            (0, 0),
        );
        let cpu = row.cpu.unwrap();
        assert!(cpu.window.is_none());
        assert!(cpu.error.unwrap().contains("not found"));
    }

    /// A client whose channel is never used: the row builder reads counters, not the network.
    fn unreachable_client() -> Client {
        let channel = tonic::transport::Endpoint::from_static("http://127.0.0.1:1").connect_lazy();
        Client::new(channel)
    }
}
