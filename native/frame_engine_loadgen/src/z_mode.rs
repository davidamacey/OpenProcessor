//! The models alone (plan 17 N0, "Z rows"): the model calls the engine makes for the plan, through
//! `gpu_frame`'s `ModelCalls` interface and a registered CUDA region, with inputs that are already
//! resident (the region is filled once with flat grey and never written again). No decode, no
//! kernels, no gRPC front: what is left is the models' own rate at this many photos in flight,
//! the number the card's ceiling is built from (plan 17 section 0.2, G1).
//!
//! For a plan with crops each photo replays the crop count a parity run recorded, so the calls
//! are the real ones in number and shape.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use gpu_frame::slab::{ALIGN, aligned};
use gpu_frame::{
    CallError, Context, InferCall, InputTensor, ModelCalls, ModelServer, ModelServerConfig, Region,
    Stream,
};
use tokio::time::Instant;

use crate::commands::{Failure, ZArgs, finish, prepare};
use crate::exchange::{Exchange, Outcome};
use crate::measure::{Setup, measure};
use crate::row::ZEvidence;
use crate::z_plan::{CallSpec, CropCounts, photo_calls, region_bytes};

/// The prefix of this tool's regions. Not the engine's `op-fe-`: a purge at the start of either
/// would unregister the other's region.
const REGION_PREFIX: &str = "op-fz-";
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
const SLICE_WAIT: Duration = Duration::from_secs(60);
/// The value the region is filled with once: a flat grey frame.
const FILL: u8 = 128;

struct Models {
    server: ModelServer,
    plan: crate::plan::PlanFile,
    crops: Vec<usize>,
}

impl Models {
    async fn call(&self, spec: &CallSpec) -> Result<(), String> {
        let slice = self
            .server
            .region()
            .slice(spec.bytes(), SLICE_WAIT)
            .await
            .map_err(|e| format!("z:region:{e}"))?;
        let inputs = [InputTensor {
            name: "images",
            datatype: "UINT8",
            shape: &spec.shape,
            slice: &slice,
        }];
        let outputs = [spec.output];
        let answer = self
            .server
            .infer(InferCall {
                model: &spec.model,
                inputs: &inputs,
                outputs: &outputs,
                deadline: Instant::now() + CALL_TIMEOUT,
            })
            .await;
        answer.map(|_| ()).map_err(|e| match e {
            CallError::Unavailable(_) => "z:unavailable".to_string(),
            CallError::Deadline => "z:deadline".to_string(),
            CallError::Model { code, .. } => format!("z:model:{code}"),
            CallError::RegionUnknown(_) => "z:region_unknown".to_string(),
        })
    }

    async fn round(&self, specs: &[CallSpec]) -> Result<(), String> {
        for answer in join_all(specs.iter().map(|s| self.call(s))).await {
            answer?;
        }
        Ok(())
    }
}

impl Exchange for Models {
    async fn exchange(&self, index: usize) -> Outcome {
        let calls = match photo_calls(&self.plan, self.crops[index]) {
            Ok(c) => c,
            Err(key) => {
                return Outcome::Failed {
                    key: format!("z:plan:{key}"),
                };
            }
        };
        let result = match self.round(&calls.round1).await {
            Ok(()) => self.round(&calls.round2).await,
            Err(key) => Err(key),
        };
        match result {
            Ok(()) => Outcome::Ok {
                route: "z".into(),
                bytes: 0,
            },
            Err(key) => Outcome::Failed { key },
        }
    }
}

/// Replay the plan's model calls with resident inputs against the model server and print the row.
///
/// # Errors
///
/// Returns `Err` when `--stats-url` is missing, the crop counts or the region size are invalid, CUDA or the region cannot be set up, or the model server cannot be reached or registered with.
pub async fn run(a: ZArgs, argv: Vec<String>) -> Result<ExitCode, Failure> {
    let p = prepare(&a.common).await?;
    let url = a
        .common
        .stats_url
        .clone()
        .ok_or("z mode needs --stats-url: the model server it calls")?;
    let (crops, crop_counts_path, crop_counts_sha) =
        crop_counts(&a, &p.list.sha256, p.list.photos.len(), &p.plan)?;
    let region_len = region_size(&a, &p.plan, &crops)?;
    let ctx = Context::new(a.cuda_device).map_err(|e| format!("CUDA: {e}"))?;
    let region = Region::new(&ctx, region_len).map_err(|e| format!("CUDA region: {e}"))?;
    fill(&ctx, &region)?;
    let name = format!("{REGION_PREFIX}{}", std::process::id());
    let server = ModelServer::new(
        ModelServerConfig {
            url: url.clone(),
            channels: a.channels,
            call_timeout: CALL_TIMEOUT,
            region_prefix: REGION_PREFIX.to_string(),
            region_name: name.clone(),
            device_id: a.server_device_id,
        },
        region,
    )
    .map_err(|e| e.to_string())?;
    let health = server.check().await;
    if !health.ready() {
        return Err(format!(
            "the model server at {url} is not ready or did not take the region"
        ));
    }
    let (calls, total_crops) = totals(&p.plan, &crops)?;
    let photos = crops.len() as f64;
    let models = Arc::new(Models {
        server,
        plan: p.plan.clone(),
        crops,
    });
    let setup = Setup {
        mode: "z",
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
        engine: None,
        z: Some(ZEvidence {
            crop_counts_path,
            crop_counts_list_sha256: crop_counts_sha,
            region_mib: region_len >> 20,
            region_name: name,
            mean_calls_per_photo: calls as f64 / photos,
            mean_crops_per_photo: total_crops as f64 / photos,
        }),
    };
    let row = measure(setup, models.clone()).await;
    // unregister before the memory goes (the region type frees it only then)
    if let Some(models) = Arc::into_inner(models) {
        let _ = models.server.close().await;
    }
    finish(&a.common, row)
}

type Counts = (Vec<usize>, Option<String>, Option<String>);

fn crop_counts(
    a: &ZArgs,
    list_sha: &str,
    photos: usize,
    plan: &crate::plan::PlanFile,
) -> Result<Counts, Failure> {
    match (&a.crop_counts, plan.runs_crops()) {
        (Some(path), true) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let counts = CropCounts::parse(&text, list_sha, photos)?;
            Ok((
                counts.crops,
                Some(path.display().to_string()),
                Some(counts.list_sha256),
            ))
        }
        (None, true) => Err("the plan embeds crops: --crop-counts is needed".into()),
        (Some(_), false) => Err("--crop-counts given for a plan without crops".into()),
        (None, false) => Ok((vec![0; photos], None, None)),
    }
}

/// The region for the in-flight photos' inputs: the largest photo's calls, each slice aligned.
fn region_size(a: &ZArgs, plan: &crate::plan::PlanFile, crops: &[usize]) -> Result<usize, Failure> {
    if let Some(mib) = a.region_mib {
        return aligned(mib << 20).ok_or_else(|| "--region-mib overflows".to_string());
    }
    let most = crops.iter().copied().max().unwrap_or(0);
    let calls = photo_calls(plan, most)?;
    let sizes: Vec<usize> = calls
        .round1
        .iter()
        .chain(calls.round2.iter())
        .map(CallSpec::bytes)
        .collect();
    Ok(region_bytes(&sizes, ALIGN, a.common.in_flight))
}

fn totals(plan: &crate::plan::PlanFile, crops: &[usize]) -> Result<(usize, usize), Failure> {
    let mut calls = 0;
    for &n in crops {
        let c = photo_calls(plan, n)?;
        calls += c.round1.len() + c.round2.len();
    }
    Ok((calls, crops.iter().sum()))
}

/// Fill the whole region once, on its own stream, and wait for it.
fn fill(ctx: &Arc<Context>, region: &Region) -> Result<(), Failure> {
    let stream = Stream::new(ctx).map_err(|e| format!("CUDA stream: {e}"))?;
    let all = region
        .slice_blocking(region.len(), SLICE_WAIT)
        .map_err(|e| format!("region fill: {e}"))?;
    stream
        .memset(&all, 0, region.len(), FILL)
        .map_err(|e| format!("region fill: {e}"))?;
    stream
        .synchronize()
        .map_err(|e| format!("region fill: {e}"))
}
