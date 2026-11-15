//! `op-frame-engine`: the frame engine process (plan 17 section 5 N0).
//!
//! Start, serve, drain on SIGTERM (readiness turns false, photos in flight get
//! `OP_FRAME_ENGINE_DRAIN_SECONDS`), exit 0; a lost CUDA context ends the process with 1 for the
//! restart policy. The pipeline that makes the answers is chosen by `--pipeline`: `stub` needs no
//! GPU. The pixel pipeline joins here as another variant of [`PipelineChoice`].

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use frame_engine::front::{Front, FrontConfig, ServeConfig, serve};
use frame_engine::health;
use frame_engine::metrics::{self, Metrics};
use frame_engine::pipeline::Pipeline;
use frame_engine::settings::Settings;
use frame_engine::stub::StubPipeline;
use tokio::net::TcpListener;
use tokio::signal::unix::{SignalKind, signal};

/// Threads of the network runtime (the tonic front, the metrics endpoint, the health tick).
/// The throwaway pipeline used 4; N0's sweep by thread role decides (plan 17 decision 4.2).
const NETWORK_THREADS: usize = 4;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PipelineChoice {
    /// Answers every branch with deterministic tensors; no CUDA, no model server.
    Stub,
}

#[derive(Debug, Parser)]
#[command(version, about = "The OpenProcessor frame engine, protocol version 1")]
struct Args {
    #[arg(long, value_enum)]
    pipeline: PipelineChoice,
    /// The stub is not ready while this file exists (read by the health tick).
    #[arg(long)]
    stub_not_ready_file: Option<PathBuf>,
}

fn main() -> ExitCode {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
    let settings = match Settings::from_env() {
        Ok(settings) => settings,
        Err(refusal) => {
            tracing::error!("{refusal}");
            return ExitCode::FAILURE;
        }
    };
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(NETWORK_THREADS)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            tracing::error!("cannot start the runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let pipeline = match args.pipeline {
        PipelineChoice::Stub => Arc::new(StubPipeline::new(args.stub_not_ready_file)),
    };
    match runtime.block_on(run(settings, pipeline)) {
        Ok(code) => code,
        Err(e) => {
            tracing::error!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// Serves until SIGTERM or SIGINT (exit 0) or a lost CUDA context (exit 1).
async fn run<P: Pipeline>(
    settings: Settings,
    pipeline: Arc<P>,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let metrics = Metrics::new();
    let scrape = TcpListener::bind(("0.0.0.0", settings.metrics_port)).await?;
    let listener = TcpListener::bind(("0.0.0.0", settings.port)).await?;
    let mut term = signal(SignalKind::terminate())?;
    let mut int = signal(SignalKind::interrupt())?;
    let (stop_metrics, metrics_stopped) = tokio::sync::oneshot::channel::<()>();
    let metrics_server = tokio::spawn(metrics::serve(scrape, Arc::clone(&metrics), async {
        let _ = metrics_stopped.await;
    }));
    let config = FrontConfig {
        max_in_flight: settings.max_in_flight,
        request_deadline_seconds: settings.request_deadline_seconds,
    };
    let (front, handle) = Front::new(Arc::clone(&pipeline), Arc::clone(&metrics), config);
    // the health tick and the signals both end the serving: one stop for the front
    let (stop, stopped) = tokio::sync::oneshot::channel::<bool>();
    let tick = tokio::spawn({
        let pipeline = Arc::clone(&pipeline);
        let every = Duration::from_secs_f64(settings.health_seconds);
        async move {
            let ended = health::run(&*pipeline, every, std::future::pending()).await;
            let _ = stop.send(ended == health::Ended::ContextLost);
        }
    });
    tracing::info!(
        "frame engine serving on {} (metrics {})",
        settings.port,
        settings.metrics_port
    );
    let lost = std::cell::Cell::new(false);
    let shutdown = async {
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
            ended = stopped => lost.set(ended.unwrap_or(false)),
        }
    };
    let serve_config = ServeConfig {
        max_message_mib: settings.max_message_mib,
        drain: Duration::from_secs_f64(settings.drain_seconds),
    };
    serve(listener, front, handle, serve_config, shutdown).await?;
    tick.abort();
    let _ = stop_metrics.send(());
    let _ = metrics_server.await;
    if lost.get() {
        tracing::error!("stopped after a lost CUDA context: exiting 1 for a restart");
        return Ok(ExitCode::FAILURE);
    }
    tracing::info!("stopped");
    Ok(ExitCode::SUCCESS)
}
