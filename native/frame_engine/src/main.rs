//! `op-frame-engine`: the frame engine process (plan 17 section 5 N0).
//!
//! Start, serve, drain on SIGTERM (readiness turns false, photos in flight get
//! `OP_FRAME_ENGINE_DRAIN_SECONDS`), exit 0; a lost CUDA context ends the process with 1 for the
//! restart policy. The pipeline that makes the answers is chosen by `--pipeline`: `stub` needs no
//! GPU, `gpu` is the engine (device 0 as `CUDA_VISIBLE_DEVICES` numbers it). Two subcommands:
//! `health` (the container's health check: the running engine answers `ModelReady`) and
//! `self-check` (one photo through plans F and C on the card, one line per stage).

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand, ValueEnum};
use frame_engine::engine::GpuPipeline;
use frame_engine::front::{Front, FrontConfig, ServeConfig, serve};
use frame_engine::health;
use frame_engine::metrics::{self, Metrics};
use frame_engine::photo::read::KeyedRoots;
use frame_engine::pipeline::Pipeline;
use frame_engine::self_check::{self, Models};
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
    /// Decodes on the card and calls the models over CUDA shared memory.
    Gpu,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Exit 0 when the engine on OP_FRAME_ENGINE_PORT answers ModelReady, 1 otherwise.
    Health {
        #[arg(long, default_value_t = 2.0)]
        timeout_seconds: f64,
    },
    /// Start the GPU pipeline against the model server at OP_FRAME_ENGINE_TRITON_URL and run one
    /// photo through plan F and plan C, one line per stage; exit 1 on any failure.
    SelfCheck {
        /// A JPEG to use instead of the built-in pattern (which has no objects to crop).
        #[arg(long)]
        photo: Option<PathBuf>,
        #[arg(long, default_value = "yolov11_small_trt_end2end")]
        detector: String,
        #[arg(long, default_value = "pe_image_encoder")]
        frame_encoder: String,
        #[arg(long, default_value = "pe_crop_encoder")]
        crop_encoder: String,
        /// Seconds to wait for the model server and the region's registration.
        #[arg(long, default_value_t = 30.0)]
        ready_seconds: f64,
    },
}

#[derive(Debug, Parser)]
#[command(
    version,
    about = "The OpenProcessor frame engine, protocol version 1",
    subcommand_negates_reqs = true
)]
struct Args {
    #[arg(long, value_enum, required = true)]
    pipeline: Option<PipelineChoice>,
    /// The stub is not ready while this file exists (read by the health tick).
    #[arg(long)]
    stub_not_ready_file: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
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
    let outcome = runtime.block_on(async {
        match (args.command, args.pipeline) {
            (Some(Command::Health { timeout_seconds }), _) => {
                let wait = Duration::from_secs_f64(timeout_seconds);
                match health::probe(settings.port, wait).await {
                    Ok(true) => Ok(ExitCode::SUCCESS),
                    Ok(false) => {
                        tracing::error!("the engine on port {} is not ready", settings.port);
                        Ok(ExitCode::FAILURE)
                    }
                    Err(e) => Err(e.into()),
                }
            }
            (
                Some(Command::SelfCheck {
                    photo,
                    detector,
                    frame_encoder,
                    crop_encoder,
                    ready_seconds,
                }),
                _,
            ) => {
                let models = Models {
                    detector,
                    frame_encoder,
                    crop_encoder,
                };
                let wait = Duration::from_secs_f64(ready_seconds);
                Ok(if self_check::run(&settings, &models, photo, wait).await {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::FAILURE
                })
            }
            (None, Some(PipelineChoice::Gpu)) => {
                let metrics = Metrics::new();
                let pipeline = GpuPipeline::start(&settings, Arc::clone(&metrics))?;
                run(settings, metrics, Arc::new(pipeline)).await
            }
            (None, Some(PipelineChoice::Stub) | None) => {
                let roots = KeyedRoots::new(
                    settings.landing_root.as_deref(),
                    settings.source_root.as_deref(),
                    u64::from(settings.max_message_mib) << 20,
                )?;
                let stub = StubPipeline::new(args.stub_not_ready_file).with_roots(roots);
                run(settings, Metrics::new(), Arc::new(stub)).await
            }
        }
    });
    match outcome {
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
    metrics: Arc<Metrics>,
    pipeline: Arc<P>,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
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
    // the tick holds the pipeline: wait for its end so the close below can own it
    let _ = tick.await;
    let _ = stop_metrics.send(());
    let _ = metrics_server.await;
    if lost.get() {
        tracing::error!("stopped after a lost CUDA context: exiting 1 for a restart");
        return Ok(ExitCode::FAILURE);
    }
    match Arc::try_unwrap(pipeline) {
        Ok(pipeline) => pipeline.close().await,
        Err(_) => {
            tracing::warn!("the pipeline is still held: what it holds is left to process exit")
        }
    }
    tracing::info!("stopped");
    Ok(ExitCode::SUCCESS)
}
