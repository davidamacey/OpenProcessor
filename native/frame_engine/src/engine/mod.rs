//! The GPU pipeline (`frame_engine/engine.py` and `__main__.py`'s start): one card, its decode
//! lanes, launch and finish pools, the region registered with the model server, and the photo
//! flow over them, behind the front's [`Pipeline`] seam.
//!
//! Start, in order, each step logged with its time: the card's name and PCI bus id, read without
//! a context (refused when `OP_FRAME_ENGINE_EXPECT_BUS_ID` names another card), the context, the
//! frame pool under `OP_FRAME_ENGINE_POOL_MIB`, the region, the model-server channels (opened
//! lazily: the server may start later), the lanes, the kernels (NVRTC) and the pools. The region
//! is registered by the first health check, after the purge of this card's prefix.
//!
//! Readiness (`transport.ready() and not engine.lost`): the model server answers ready, the region
//! is registered, and the context is not lost; one atomic read. A lost context is a sticky CUDA
//! code from any call (`gpu_frame`'s table): every photo is refused, readiness turns false and the
//! health tick ends the process for its restart policy. At a clean stop the region is unregistered,
//! then freed (`ModelServer::close`).

mod gpu;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use gpu_frame::model_server::RegistrationCause;
use gpu_frame::{Context, DeviceCap, FramePool, ModelServer, ModelServerConfig, Region};
use tokio::sync::Notify;

pub use self::gpu::{Frame, Gpu, GpuConfig};
use crate::metrics::{ByCause, ByCode, Metrics};
use crate::photo::budget::GpuBudget;
use crate::photo::{PhotoFlow, PhotoSettings};
use crate::pipeline::{Photo, PhotoAnswer, Pipeline, PipelineError};
use crate::settings::{Settings, parse_bus_id};

const MIB: u64 = 1 << 20;
/// Region names start with this, then the card and the instance (`transport.REGION_PREFIX`).
const REGION_PREFIX: &str = "op-fe-";

/// `transport.region_prefix`: every region of this card's engine `instance` carries it.
pub fn region_prefix(card: &str, instance: &str) -> String {
    let safe: String = card
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect();
    format!("{REGION_PREFIX}{safe}-{instance}-")
}

/// `transport.process_identity`: the host name (the container id under Docker) and the process id.
pub fn process_identity() -> String {
    let host = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|h| h.trim().to_owned())
        .unwrap_or_else(|_| "host".into());
    format!("{host}-{}", std::process::id())
}

/// Launch threads per card before N0's measurement by thread role sets them (plan 17 section
/// 2.1): the physical cores over 6, at least 2.
pub fn launch_threads(physical_cores: usize) -> usize {
    (physical_cores / 6).max(2)
}

/// The card's name and bus id, refused when the expected card is another one.
///
/// # Errors
///
/// There is no CUDA device 0, or `expect` is set and names another PCI bus id.
pub fn identify(expect: Option<&str>) -> Result<(String, String), String> {
    let (name, bus) = gpu_frame::device_identity(0).map_err(|e| format!("no CUDA device: {e}"))?;
    tracing::info!("device 0: {name}, PCI bus id {bus}");
    if let Some(want) = expect
        && parse_bus_id(want) != parse_bus_id(&bus)
    {
        return Err(format!(
            "OP_FRAME_ENGINE_EXPECT_BUS_ID is {want} but device 0 is {bus} ({name}); \
             host runs set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES"
        ));
    }
    Ok((name, bus))
}

/// The GPU pipeline.
pub struct GpuPipeline {
    flow: PhotoFlow<Gpu, GpuBudget>,
    gpu: Arc<Gpu>,
    ready: AtomicBool,
    lost_counted: AtomicBool,
    registrations: [AtomicU64; 3],
}

const CAUSES: [RegistrationCause; 3] = [
    RegistrationCause::Start,
    RegistrationCause::ServerLostRegion,
    RegistrationCause::RegionUnknown,
];

fn step<T, E: std::fmt::Display>(
    what: &str,
    f: impl FnOnce() -> Result<T, E>,
) -> Result<T, String> {
    let started = Instant::now();
    let out = f().map_err(|e| format!("{what}: {e}"))?;
    tracing::info!(
        "start: {what} in {:.0} ms",
        started.elapsed().as_secs_f64() * 1000.0
    );
    Ok(out)
}

impl GpuPipeline {
    /// Everything up to the first health check. Call inside the tokio runtime (the model-server
    /// channels are tonic's); it blocks while the kernels compile.
    ///
    /// # Errors
    ///
    /// Any start step fails (device identity, CUDA context, pools, region, model-server channels, kernels, registration); the text names the step.
    pub fn start(s: &Settings, metrics: Arc<Metrics>) -> Result<Self, String> {
        let (_, bus) = identify(s.expect_bus_id.as_deref())?;
        let ctx = step("context", || Context::new(0))?;
        let cap =
            DeviceCap::new(usize::try_from(u64::from(s.pool_mib) * MIB).unwrap_or(usize::MAX));
        let frames = step("frame pool", || FramePool::new(&ctx, cap))?;
        let region_bytes = usize::try_from(u64::from(s.region_mib) * MIB).unwrap_or(usize::MAX);
        let region = step("region", || Region::new(&ctx, region_bytes))?;
        let prefix = region_prefix(&bus, &s.instance);
        let server = step("model-server channels", || {
            ModelServer::new(
                ModelServerConfig {
                    url: s.triton_url.clone(),
                    channels: s.triton_channels as usize,
                    call_timeout: Duration::from_secs_f64(s.triton_timeout_seconds),
                    region_name: format!("{prefix}{}", process_identity()),
                    region_prefix: prefix.clone(),
                    device_id: i64::from(s.triton_device_id),
                },
                region,
            )
        })?;
        let cfg = GpuConfig {
            lanes: s.lanes as usize,
            launch_threads: launch_threads(crate::cores::physical_cores()),
            finish_threads: s.encode_threads as usize,
            jpeg_quality: s.jpeg_quality,
            crop_max_side: s.crop_max_side,
            stage_sync: s.stage_sync,
        };
        let gpu = Arc::new(step("lanes, kernels and pools", || {
            Gpu::start(&ctx, frames, server, &cfg)
        })?);
        let (major, minor, patch) = gpu.nvjpeg_version();
        tracing::info!(
            "frame engine on {bus}: {} decode lanes (nvJPEG {major}.{minor}.{patch}), {} launch \
             and {} finish threads, region {} MiB ({}), budget {} MiB, pool {} MiB, workers {}",
            gpu.lane_count(),
            cfg.launch_threads,
            cfg.finish_threads,
            s.region_mib,
            gpu.server.region_name(),
            s.budget_mib,
            s.pool_mib,
            s.workers,
        );
        let budget_bytes = u64::from(s.budget_mib) * MIB;
        let flow = PhotoFlow {
            device: Arc::clone(&gpu),
            budget: GpuBudget::new(
                s.workers,
                budget_bytes,
                s.budget_wait_seconds,
                metrics.budget_in_use.clone(),
            ),
            settings: PhotoSettings {
                min_megapixels: s.min_megapixels,
                max_megapixels: s.max_megapixels,
                budget_bytes,
                budget_wait: Duration::from_secs_f64(s.budget_wait_seconds),
                slab_wait: Duration::from_secs_f64(s.slab_wait_seconds),
                window: region_bytes / (s.workers as usize).max(1),
            },
            metrics,
        };
        Ok(Self {
            flow,
            gpu,
            ready: AtomicBool::new(false),
            lost_counted: AtomicBool::new(false),
            registrations: Default::default(),
        })
    }

    fn metrics(&self) -> &Metrics {
        &self.flow.metrics
    }

    /// Count a lost context once, by the code its message names.
    fn count_lost(&self, why: &str) {
        if self.lost_counted.swap(true, Ordering::AcqRel) {
            return;
        }
        let code = why
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .find(|w| w.starts_with("CUDA_ERROR_"))
            .unwrap_or("unknown");
        self.metrics()
            .context_lost
            .get_or_create(&ByCode { code: code.into() })
            .inc();
        tracing::error!("CUDA context lost ({why}): refusing work, exiting");
    }
}

impl Pipeline for GpuPipeline {
    fn ready(&self) -> bool {
        self.ready.load(Ordering::Acquire) && !self.gpu.context().is_lost()
    }

    fn lost(&self) -> bool {
        self.gpu.context().is_lost()
    }

    fn wake(&self) -> &Notify {
        &self.gpu.wake
    }

    async fn check(&self) {
        let health = self.gpu.server.check().await;
        let lost = self.lost();
        if lost {
            self.count_lost("found by the health check");
        }
        self.ready.store(health.ready() && !lost, Ordering::Release);
        let m = self.metrics();
        for (k, cause) in CAUSES.into_iter().enumerate() {
            let now = self.gpu.server.registrations(cause);
            let seen = self.registrations[k].swap(now, Ordering::AcqRel);
            if now > seen {
                m.registrations
                    .get_or_create(&ByCause {
                        cause: cause.label().into(),
                    })
                    .inc_by(now - seen);
            }
        }
        let usage = self.gpu.server.region().usage();
        let gauge = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        m.slab_in_use.set(gauge(usage.in_use as u64));
        m.slab_peak.set(gauge(usage.peak as u64));
        m.slab_waits.set(gauge(usage.waits));
    }

    async fn process(&self, photo: Photo) -> Result<PhotoAnswer, PipelineError> {
        let answer = self.flow.process(photo).await;
        if let Err(PipelineError::ContextLost(why)) = &answer {
            self.ready.store(false, Ordering::Release);
            self.count_lost(why);
        }
        answer
    }

    async fn close(self) {
        let Self { flow, gpu, .. } = self;
        drop(flow);
        let Ok(gpu) = Arc::try_unwrap(gpu) else {
            tracing::warn!("photos still hold the pipeline: the region is left to process exit");
            return;
        };
        let server = gpu.into_server();
        match Arc::try_unwrap(server) {
            Ok(server) => {
                let freed = server.close().await;
                tracing::info!("region unregistered and freed: {freed}");
            }
            Err(_) => tracing::warn!("a call still holds the region: it is left to process exit"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_region_is_named_as_the_python_transport_names_it() {
        assert_eq!(
            region_prefix("0000:65:00.0", "fe0"),
            "op-fe-0000-65-00-0-fe0-"
        );
        assert_eq!(
            region_prefix("0000:B3:00.0", "fe-1"),
            "op-fe-0000-b3-00-0-fe-1-"
        );
        let id = process_identity();
        assert!(id.ends_with(&format!("-{}", std::process::id())));
    }

    #[test]
    fn launch_threads_derive_from_the_cores_with_a_floor_of_two() {
        assert_eq!(launch_threads(24), 4);
        assert_eq!(launch_threads(8), 2);
        assert_eq!(launch_threads(1), 2);
    }
}
