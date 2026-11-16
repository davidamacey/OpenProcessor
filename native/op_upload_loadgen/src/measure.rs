//! One measured row: the loop, the counters read at the window's edges, and the row built from
//! both. At the window's open the cgroups and this process are read, then the instant is marked;
//! at its close the instant is marked, then they are read, so the CPU window lies inside the
//! rate's window.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use frame_engine_loadgen::cgroup::{Cgroup, CpuStat};
use frame_engine_loadgen::exchange::Probe;
use frame_engine_loadgen::host;
use frame_engine_loadgen::list::PhotoList;
use frame_engine_loadgen::row::{HostEvidence, ListEvidence, Loadgen};

use crate::closed_loop::{self, LoopConfig};
use crate::multipart::FieldOrder;
use crate::process::{self, CpuSeconds};
use crate::record;
use crate::row::{
    BOUNDS, NamedCpu, Row, SCHEMA, StampsEvidence, UploadEvidence, cpu_evidence, order_name,
};
use crate::send::Sender;
use crate::summary::summarize;

/// A named cgroup to read at the window's edges.
pub struct NamedCgroup {
    /// The name given on the command line.
    pub name: String,
    /// The cgroup.
    pub cgroup: Cgroup,
}

/// Everything a row is made of besides the loop's result.
pub struct Setup {
    /// The arm's name.
    pub label: String,
    /// The command line as given.
    pub command: Vec<String>,
    /// The list, for its evidence.
    pub list: PhotoList,
    /// The settings recorded in the row.
    pub upload: UploadEvidence,
    /// The loop's shape.
    pub config: LoopConfig,
    /// Where the stamps file goes.
    pub stamps_path: PathBuf,
    /// The containers whose CPU is read.
    pub cgroups: Vec<NamedCgroup>,
    /// Free text recorded as given.
    pub server_note: Option<String>,
}

#[derive(Default)]
struct Reads {
    unix_ms: u64,
    cgroups: Vec<Result<CpuStat, String>>,
    own: Option<CpuSeconds>,
}

struct Edges<'a> {
    cgroups: &'a [NamedCgroup],
    open: Reads,
    close: Reads,
}

impl Edges<'_> {
    fn read(&self) -> Reads {
        Reads {
            unix_ms: 0,
            cgroups: self
                .cgroups
                .iter()
                .map(|n| {
                    n.cgroup
                        .read_cpu()
                        .map_err(|e| format!("{}: {e}", n.cgroup.path().display()))
                })
                .collect(),
            own: process::read(),
        }
    }
}

impl Probe for Edges<'_> {
    async fn window_open(&mut self) {
        let mut reads = self.read();
        reads.unix_ms = host::unix_ms();
        self.open = reads;
    }

    async fn window_close(&mut self) {
        let unix_ms = host::unix_ms();
        let mut reads = self.read();
        reads.unix_ms = unix_ms;
        self.close = reads;
    }
}

/// Run the loop, write the stamps file and build the row.
///
/// # Errors
///
/// Returns `Err` when the stamps file cannot be written (the run is not reported without it).
pub async fn measure(setup: Setup, sender: Arc<Sender>) -> Result<Row, String> {
    let started = host::unix_ms();
    let load_start = host::load1();
    let mut edges = Edges {
        cgroups: &setup.cgroups,
        open: Reads::default(),
        close: Reads::default(),
    };
    let output = closed_loop::run(Arc::clone(&sender), setup.config, &mut edges).await;
    let (load_end, ended) = (host::load1(), host::unix_ms());
    let summary = summarize(&output.records, output.window);
    record::write_stamps(&setup.stamps_path, &output.records, |ms| {
        output.window.phase_of(ms)
    })
    .map_err(|e| format!("{}: {e}", setup.stamps_path.display()))?;
    let seconds = output.window.seconds();
    let answered =
        (summary.photos_success + summary.photos_duplicate + summary.photos_unread) as usize;
    let cpu = setup
        .cgroups
        .iter()
        .enumerate()
        .map(|(i, n)| NamedCpu {
            name: n.name.clone(),
            evidence: cpu_evidence(
                &n.cgroup,
                &edges.open.cgroups[i],
                &edges.close.cgroups[i],
                seconds,
                answered,
            ),
        })
        .collect();
    let sent_photos: u64 = output.records.iter().map(|r| r.photos as u64).sum();
    let list = &setup.list;
    Ok(Row {
        schema: SCHEMA,
        label: setup.label,
        command: setup.command,
        loadgen: Loadgen {
            version: env!("CARGO_PKG_VERSION"),
            build_sha: option_env!("OP_BUILD_SHA"),
        },
        started_at: host::iso_utc(started),
        ended_at: host::iso_utc(ended),
        window_start_unix_ms: edges.open.unix_ms,
        window_end_unix_ms: edges.close.unix_ms,
        host: HostEvidence {
            hostname: host::hostname(),
            cpu: host::cpuinfo(),
            load1_start: load_start,
            load1_end: load_end,
        },
        list: ListEvidence {
            path: list.path.display().to_string(),
            sha256: list.sha256.clone(),
            photos: list.photos.len(),
            bytes: list.total_bytes,
        },
        upload: setup.upload,
        in_flight: setup.config.in_flight,
        warmup_s: setup.config.warmup.as_secs_f64(),
        window_s: setup.config.window.as_secs_f64(),
        requests_started: output.requests_started,
        list_cycles_sent: sent_photos as f64 / list.photos.len() as f64,
        summary,
        stamps: StampsEvidence {
            path: setup.stamps_path.display().to_string(),
            bounds: BOUNDS,
        },
        loadgen_cpu: edges
            .open
            .own
            .zip(edges.close.own)
            .and_then(|(a, b)| process::window(a, b, seconds)),
        cpu,
        server_note: setup.server_note,
    })
}

/// The settings the row records for a run.
pub struct UploadSettings<'a> {
    /// The upload route's URL.
    pub url: &'a str,
    /// The field order.
    pub order: FieldOrder,
    /// The `source` field.
    pub source: &'a str,
    /// The `run_id` field.
    pub run_id: &'a str,
    /// The most photos in one request.
    pub max_images: usize,
    /// The most photo bytes in one request.
    pub max_bytes: u64,
    /// The largest frame of a photo part.
    pub chunk_bytes: usize,
    /// One request's time limit.
    pub timeout: Duration,
}

/// The row's upload evidence from the settings and the packs.
pub fn upload_evidence(
    s: &UploadSettings<'_>,
    in_flight: usize,
    packs: &[std::ops::Range<usize>],
) -> UploadEvidence {
    let photos: usize = packs.iter().map(ExactSizeIterator::len).sum();
    UploadEvidence {
        url: s.url.to_string(),
        http_version: "HTTP/1.1",
        pool_max_idle_per_host: in_flight,
        field_order: order_name(s.order),
        source: s.source.to_string(),
        run_id: s.run_id.to_string(),
        max_images: s.max_images,
        max_bytes: s.max_bytes,
        chunk_bytes: s.chunk_bytes,
        timeout_s: s.timeout.as_secs_f64(),
        packs_per_pass: packs.len(),
        mean_photos_per_pack: photos as f64 / packs.len() as f64,
    }
}
