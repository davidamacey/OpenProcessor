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
use crate::z_mode;

/// The subcommand: which measurement or readiness check to run.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Measure an engine at an address: a closed loop of photos in flight, protocol version 1.
    Engine(Box<EngineArgs>),
    /// Measure the models alone: the same model calls with inputs already resident.
    Z(Box<ZArgs>),
    /// Exit 0 when the server answers ServerReady and ModelReady for a model.
    Ready(ReadyArgs),
    /// List the model server's CUDA shared-memory regions; exit 0 when one carries the prefix.
    Regions(RegionsArgs),
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
    /// Seconds of load before the window opens; those photos are not measured.
    #[arg(long, default_value_t = 5.0)]
    pub warmup_s: f64,
    /// Seconds of the measured window.
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
    /// Take no statistics from the model server: the row has no `model_server` evidence.
    #[arg(long)]
    pub no_stats: bool,
    /// The card to sample (its PCI bus id as NVML names it, for example `00000000:C1:00.0`).
    #[arg(long, conflicts_with = "no_gpu", required_unless_present = "no_gpu")]
    pub gpu_bus_id: Option<String>,
    /// Sample no GPU: the row has no `gpu` evidence.
    #[arg(long)]
    pub no_gpu: bool,
    /// Exit 0 even when photos failed (fault runs); the row still counts them.
    #[arg(long)]
    pub allow_failures: bool,
}

/// Arguments of the `engine` mode.
#[derive(Debug, Args)]
pub struct EngineArgs {
    /// The arguments every measured row takes.
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

/// Arguments of the `z` (models alone) mode.
#[derive(Debug, Args)]
pub struct ZArgs {
    /// The arguments every measured row takes.
    #[command(flatten)]
    pub common: Common,
    /// Each photo's crop count as a parity run recorded it (required when the plan embeds crops).
    #[arg(long)]
    pub crop_counts: Option<PathBuf>,
    /// The card as this process's CUDA numbers it (`CUDA_VISIBLE_DEVICES` applies).
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,
    /// The same card as the model server numbers it.
    #[arg(long, default_value_t = 0)]
    pub server_device_id: i64,
    /// Region size in MiB; by default what the in-flight photos need.
    #[arg(long)]
    pub region_mib: Option<usize>,
    /// Channels to the model server.
    #[arg(long, default_value_t = 4)]
    pub channels: usize,
}

/// Arguments of the `ready` check.
#[derive(Debug, Args)]
pub struct ReadyArgs {
    /// The gRPC address to ask.
    #[arg(long)]
    pub url: String,
    /// The model asked after: `frame_engine` for an engine, a model's name for the model server.
    #[arg(long, default_value = client::MODEL)]
    pub model: String,
    /// Seconds to wait for each call.
    #[arg(long, default_value_t = 5.0)]
    pub timeout_s: f64,
}

/// Arguments of the `regions` check.
#[derive(Debug, Args)]
pub struct RegionsArgs {
    /// The model server's gRPC address.
    #[arg(long)]
    pub url: String,
    /// The prefix of the engine's regions (`op-fe-`).
    #[arg(long, default_value = "op-fe-")]
    pub prefix: String,
    /// Seconds to wait for each call.
    #[arg(long, default_value_t = 5.0)]
    pub timeout_s: f64,
}

/// An error in the command line's meaning or a source that cannot be opened: nothing measured.
pub type Failure = String;

/// A positive, finite number of seconds as a `Duration`; `label` names the option in the error.
///
/// # Errors
///
/// Returns `Err` when `v` is zero, negative, infinite or NaN.
pub fn secs(label: &str, v: f64) -> Result<Duration, Failure> {
    if v.is_finite() && v > 0.0 {
        Ok(Duration::from_secs_f64(v))
    } else {
        Err(format!("{label} must be a positive number of seconds"))
    }
}

/// Run one subcommand and give the process exit code: 0 on success (for a measured row, no failed photo unless `--allow-failures`), 1 for a not-ready server or a missing region prefix, 3 for failed photos.
///
/// # Errors
///
/// Returns `Err` when an option is invalid, an input cannot be opened, the server cannot be reached or is not ready for a measured mode, or the row cannot be written.
pub async fn run(command: Command, argv: Vec<String>) -> Result<ExitCode, Failure> {
    match command {
        Command::Engine(a) => engine(*a, argv).await,
        Command::Z(a) => z_mode::run(*a, argv).await,
        Command::Ready(a) => ready(&a).await,
        Command::Regions(a) => regions(&a).await,
    }
}

async fn ready(a: &ReadyArgs) -> Result<ExitCode, Failure> {
    let timeout = secs("--timeout-s", a.timeout_s)?;
    let mut clients = client::connect(&a.url, 1, timeout).await?;
    let ready = client::is_ready(&mut clients[0], &a.model, timeout).await?;
    println!("{}", if ready { "ready" } else { "not ready" });
    Ok(if ready {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// Whether any of `names` carries `prefix` (an empty prefix matches nothing).
pub fn holds_prefix(names: &[String], prefix: &str) -> bool {
    !prefix.is_empty() && names.iter().any(|n| n.starts_with(prefix))
}

async fn regions(a: &RegionsArgs) -> Result<ExitCode, Failure> {
    let timeout = secs("--timeout-s", a.timeout_s)?;
    let mut clients = client::connect(&a.url, 1, timeout).await?;
    let names = client::cuda_regions(&mut clients[0], timeout).await?;
    for name in &names {
        println!("{name}");
    }
    Ok(if holds_prefix(&names, &a.prefix) {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}

/// The inputs every mode shares, read before any clock starts.
pub struct Prepared {
    /// The photos, held in memory, with the hash of their list file.
    pub list: PhotoList,
    /// The plan every photo carries.
    pub plan: PlanFile,
    /// How long the loop runs before the window opens.
    pub warmup: Duration,
    /// How long the measured window lasts.
    pub window: Duration,
    /// The measured process's cgroup, when `--cpu-stat` is given.
    pub cgroup: Option<Cgroup>,
    /// The model server's address and one channel to it, when `--stats-url` is given.
    pub stats: Option<(String, Client)>,
    /// The sampled card and the sampler that reads it, when `--gpu-bus-id` is given.
    pub gpu: Option<Gpu>,
}

/// Read the list and the plan, connect the statistics channel and start the GPU sampler, all before any clock starts.
///
/// # Errors
///
/// Returns `Err` when `--in-flight` is 0, a duration is not positive, the list or the plan cannot be loaded, the model server cannot be connected, or the card cannot be opened.
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
    if !client::is_ready(&mut clients[0], client::MODEL, Duration::from_secs(5)).await? {
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
///
/// # Errors
///
/// Returns `Err` when the row cannot be serialised or the `--out` file cannot be opened or written.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_region_is_the_engines_when_its_name_starts_with_the_prefix() {
        let names = ["op-fe-4242".to_string(), "other".to_string()];
        assert!(holds_prefix(&names, "op-fe-"));
        assert!(!holds_prefix(&names, "op-fz-"));
        assert!(!holds_prefix(&names, ""), "an empty prefix matches nothing");
        assert!(!holds_prefix(&[], "op-fe-"));
    }

    #[test]
    fn a_duration_must_be_a_positive_number_of_seconds() {
        assert_eq!(secs("x", 1.5).unwrap(), Duration::from_millis(1500));
        assert!(secs("x", 0.0).is_err());
        assert!(secs("x", -1.0).is_err());
        assert!(secs("x", f64::NAN).is_err());
    }
}
