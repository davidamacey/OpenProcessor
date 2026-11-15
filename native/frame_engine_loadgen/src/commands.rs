//! The subcommands: what the command line means, and the wiring of the sources a row reads.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Args, Subcommand};

use crate::cgroup::Cgroup;
use crate::client::{self, Client, EngineClient};
use crate::gpu::{NvmlSource, SAMPLE_INTERVAL, Sampler};
use crate::list::PhotoList;
use crate::measure::{Gpu, Setup, measure};
use crate::plan::PlanFile;
use crate::row::{EngineEvidence, Row};

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Measure an engine at an address: a closed loop of photos in flight, protocol version 1.
    Engine(Box<EngineArgs>),
    /// Exit 0 when the engine answers ServerReady and ModelReady for `frame_engine`.
    Ready(ReadyArgs),
}

/// Arguments of every measured row.
#[derive(Debug, Args)]
pub struct Common {
    /// The photo list: one path per line.
    #[arg(long)]
    pub list: PathBuf,
    /// The plan JSON every photo carries (written by `gpu_prep_parity.py plan`).
    #[arg(long)]
    pub plan_file: PathBuf,
    /// Photos in flight.
    #[arg(long)]
    pub in_flight: usize,
    #[arg(long, default_value_t = 5.0)]
    pub warmup_s: f64,
    #[arg(long, default_value_t = 20.0)]
    pub window_s: f64,
    /// Name of this arm, recorded in the row.
    #[arg(long, default_value = "")]
    pub label: String,
    /// Append the row to this file as one JSON line (it is also printed).
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// The cgroup v2 `cpu.stat` of the process measured: the engine, or in `z` mode the model
    /// server. Memory files beside it are read too.
    #[arg(long)]
    pub cpu_stat: Option<PathBuf>,
    /// The model server's gRPC address, for the statistics deltas.
    #[arg(
        long,
        conflicts_with = "no_stats",
        required_unless_present = "no_stats"
    )]
    pub stats_url: Option<String>,
    #[arg(long)]
    pub no_stats: bool,
    /// The card to sample (its PCI bus id as NVML names it, for example `00000000:C1:00.0`).
    #[arg(long, conflicts_with = "no_gpu", required_unless_present = "no_gpu")]
    pub gpu_bus_id: Option<String>,
    #[arg(long)]
    pub no_gpu: bool,
    /// Exit 0 even when photos failed (fault runs); the row still counts them.
    #[arg(long)]
    pub allow_failures: bool,
}

#[derive(Debug, Args)]
pub struct EngineArgs {
    #[command(flatten)]
    pub common: Common,
    /// The engine's gRPC address.
    #[arg(long)]
    pub url: String,
    /// HTTP/2 connections to the engine.
    #[arg(long, default_value_t = 4)]
    pub channels: usize,
    /// One photo's timeout (the API's is 120 s).
    #[arg(long, default_value_t = 120.0)]
    pub timeout_s: f64,
}

#[derive(Debug, Args)]
pub struct ReadyArgs {
    #[arg(long)]
    pub url: String,
    #[arg(long, default_value_t = 5.0)]
    pub timeout_s: f64,
}

/// An error in the command line's meaning or a source that cannot be opened: nothing measured.
pub type Failure = String;

pub fn secs(label: &str, v: f64) -> Result<Duration, Failure> {
    if v.is_finite() && v > 0.0 {
        Ok(Duration::from_secs_f64(v))
    } else {
        Err(format!("{label} must be a positive number of seconds"))
    }
}

pub async fn run(command: Command, argv: Vec<String>) -> Result<ExitCode, Failure> {
    match command {
        Command::Engine(a) => engine(*a, argv).await,
        Command::Ready(a) => ready(&a).await,
    }
}

async fn ready(a: &ReadyArgs) -> Result<ExitCode, Failure> {
    let timeout = secs("--timeout-s", a.timeout_s)?;
    let mut clients = client::connect(&a.url, 1, timeout).await?;
    let ready = client::is_ready(&mut clients[0], timeout).await?;
    println!("{}", if ready { "ready" } else { "not ready" });
    Ok(if ready {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// The inputs every mode shares, read before any clock starts.
pub struct Prepared {
    pub list: PhotoList,
    pub plan: PlanFile,
    pub warmup: Duration,
    pub window: Duration,
    pub cgroup: Option<Cgroup>,
    pub stats: Option<(String, Client)>,
    pub gpu: Option<Gpu>,
}

pub async fn prepare(c: &Common) -> Result<Prepared, Failure> {
    if c.in_flight == 0 {
        return Err("--in-flight must be at least 1".into());
    }
    let warmup = secs("--warmup-s", c.warmup_s)?;
    let window = secs("--window-s", c.window_s)?;
    let list = PhotoList::load(&c.list).map_err(|e| format!("list: {e}"))?;
    let plan = PlanFile::load(&c.plan_file).map_err(|e| format!("plan: {e}"))?;
    let stats = match &c.stats_url {
        Some(url) => {
            let mut clients = client::connect(url, 1, Duration::from_secs(10)).await?;
            Some((url.clone(), clients.remove(0)))
        }
        None => None,
    };
    let gpu = match &c.gpu_bus_id {
        Some(bus_id) => {
            let (source, identity) = NvmlSource::open(bus_id)?;
            Some(Gpu {
                sampler: Sampler::start(source, SAMPLE_INTERVAL),
                identity,
            })
        }
        None => None,
    };
    Ok(Prepared {
        list,
        plan,
        warmup,
        window,
        cgroup: c.cpu_stat.clone().map(Cgroup::new),
        stats,
        gpu,
    })
}

async fn engine(a: EngineArgs, argv: Vec<String>) -> Result<ExitCode, Failure> {
    let timeout = secs("--timeout-s", a.timeout_s)?;
    let p = prepare(&a.common).await?;
    let mut clients = client::connect(&a.url, a.channels, timeout).await?;
    if !client::is_ready(&mut clients[0], Duration::from_secs(5)).await? {
        return Err(format!("{} is not ready: nothing measured", a.url));
    }
    let exchange = Arc::new(EngineClient::new(
        clients,
        &p.list.photos,
        &p.plan.text,
        timeout,
    ));
    let setup = Setup {
        mode: "engine",
        label: a.common.label.clone(),
        command: argv,
        list: p.list,
        plan: p.plan,
        plan_path: a.common.plan_file.display().to_string(),
        in_flight: a.common.in_flight,
        warmup: p.warmup,
        window: p.window,
        cgroup: p.cgroup,
        stats: p.stats,
        gpu: p.gpu,
        engine: Some(EngineEvidence {
            address: a.url,
            channels: a.channels,
            timeout_s: a.timeout_s,
        }),
        z: None,
    };
    let row = measure(setup, exchange).await;
    finish(&a.common, row)
}

/// Write the row, say it in one line on stderr, and decide the exit code.
pub fn finish(c: &Common, row: Row) -> Result<ExitCode, Failure> {
    let line = serde_json::to_string(&row).map_err(|e| format!("row: {e}"))?;
    println!("{line}");
    if let Some(path) = &c.out {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        writeln!(file, "{line}").map_err(|e| format!("{}: {e}", path.display()))?;
    }
    eprintln!(
        "{} {}: {:.1} photos/s, p50 {:.1} ms, p99 {:.1} ms, failed {}",
        row.mode, row.label, row.photos_per_s, row.p50_ms, row.p99_ms, row.failed
    );
    let clean = row.failed == 0 && row.photos > 0;
    Ok(if clean || c.allow_failures {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(3)
    })
}
