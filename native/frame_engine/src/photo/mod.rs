//! One photo through the engine (`FrameEngine._process`, `_Rounds` and the branches of
//! `frame_engine/engine.py`, plan 17 section 2.2), as an async control task on the network
//! runtime that hands each piece of work to the thread that owns it:
//!
//! 1. Route by header (`decode`): the markers, then the end-of-image walk, before any bytes are
//!    reserved. A photo the GPU route does not take answers its status (`corrupt`, `huge`) or,
//!    for the host route N0 does not build, `decoder`. A need above the whole budget is
//!    `over_budget`.
//! 2. Reserve the frame (`budget`): a worker slot and the frame's bytes, first come first served,
//!    bounded by `OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS`, then an overload.
//! 3. Round 1: the slices of the inputs that need only the frame (`letterbox`, `embed_frame`),
//!    then one job on a decode lane: nvJPEG, the orientation, the size check, the fills and
//!    resamples into those slices, one wait for the lane's stream. Every round-1 call goes out.
//! 4. Settle: the detector's boxes, as frame rectangles.
//! 5. Round 2: the crop squares in chunks of the plan's batch, staged within the photo's window
//!    (collecting its own oldest call when the next chunk would pass it), written with the blur
//!    sums by one job on the launch pool; their calls go out; the crop JPEGs start on the finish
//!    pool.
//! 6. Finish: collect, outputs in the wire's order.
//!
//! A model error fails its branch (`model_error`) and what needs it (`dependency_failed`), and the
//! photo goes on; a model server that is gone, a deadline or a lost region fail the photo
//! (`UNAVAILABLE`). A failed allocation anywhere answers `gpu_memory`; a lost context fails the
//! photo and every later one. Cancellation: the request's token is checked by every pool before a
//! job starts, so a photo whose caller left releases its slices, frame and budget instead of
//! starting work; dropping this future cancels its calls.

pub mod budget;
pub mod calls;
pub mod clock;
pub mod device;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use self::calls::{Answer, CallId, PhotoCalls};
use self::clock::{
    BLUR, BUDGET, CROP_COPY, CROP_JPEG, DECODE, DETECT, EMBED_CROPS, EMBED_FRAME, EMBED_PE,
    LETTERBOX, QUEUE, StageClock,
};
use self::device::{DecodeJob, Decoded, Device, LaunchJob, Write};
use crate::admission::FrameBudget;
use crate::branches::{BUILT, detect, embed};
use crate::fault::{GpuFault, job_failed};
use crate::jpeg_header::{Decoder, oriented_size, route};
use crate::metrics::{ByImage, ByOutcome, Metrics};
use crate::pipeline::{Photo, PhotoAnswer, PipelineError, Tensor};
use crate::wire::EnginePlan;
use crate::wire::branch_status::{
    BranchState, DEPENDENCY_FAILED, IMAGE_FAILED, MODEL_ERROR, UNSUPPORTED_BRANCH,
};
use crate::wire::plan::Crops;

/// Why a photo stops before its branches finish: a photo status, or a failure of the request.
#[derive(Debug)]
pub enum Stop {
    Status(&'static str),
    Fail(PipelineError),
}

impl From<GpuFault> for Stop {
    fn from(f: GpuFault) -> Self {
        match f {
            GpuFault::OutOfMemory(_) => Stop::Status("gpu_memory"),
            GpuFault::ContextLost(m) => Stop::Fail(PipelineError::ContextLost(m)),
            GpuFault::Failed(m) => Stop::Fail(PipelineError::Engine(m)),
        }
    }
}

fn engine(m: impl std::fmt::Display) -> Stop {
    Stop::Fail(PipelineError::Engine(m.to_string()))
}

/// What the flow reads from the settings.
#[derive(Debug, Clone)]
pub struct PhotoSettings {
    pub min_megapixels: f64,
    pub max_megapixels: f64,
    /// `OP_FRAME_ENGINE_BUDGET_MIB` in bytes: a photo needing more answers `over_budget`.
    pub budget_bytes: u64,
    pub budget_wait: Duration,
    pub slab_wait: Duration,
    /// Region bytes one photo may hold with calls in flight.
    pub window: usize,
}

/// The photo flow over a device and a frame budget.
pub struct PhotoFlow<D, B> {
    pub device: Arc<D>,
    pub budget: B,
    pub settings: PhotoSettings,
    pub metrics: Arc<Metrics>,
}

/// The answer of a photo that decoded nothing: every reported branch `image_failed`, unbuilt ones
/// `unsupported_branch` (`FrameEngine._failed`).
pub fn refused(plan: &EnginePlan, status: &str) -> PhotoAnswer {
    let mut branches = unsupported(plan);
    for name in plan.reported() {
        branches
            .entry(name)
            .or_insert(BranchState::Error(IMAGE_FAILED));
    }
    PhotoAnswer {
        status: status.to_owned(),
        decode_route: "none".into(),
        branches,
        outputs: Vec::new(),
    }
}

fn unsupported(plan: &EnginePlan) -> BTreeMap<String, BranchState> {
    plan.reported()
        .into_iter()
        .filter(|b| !BUILT.contains(&b.as_str()))
        .map(|b| (b, BranchState::Error(UNSUPPORTED_BRANCH)))
        .collect()
}

/// The built branches the plan runs.
#[derive(Debug, Clone, Copy)]
struct Runs {
    detect: bool,
    embed_frame: bool,
    embed_crops: bool,
    crops: bool,
}

impl Runs {
    fn of(plan: &EnginePlan) -> Self {
        Self {
            detect: plan.runs("detect") && plan.detector.is_some(),
            embed_frame: plan.runs("embed_frame") && plan.embed.is_some(),
            embed_crops: plan.runs("embed_crops") && plan.crop_embed.is_some(),
            crops: plan.crops == Crops::Inline,
        }
    }

    /// Output widths the round-1 and round-2 resamples write, for the scratch they reserve.
    fn widths(&self, plan: &EnginePlan) -> Vec<u32> {
        let mut out = Vec::new();
        if let (true, Some(d)) = (self.detect, &plan.detector) {
            out.push(d.input_size);
        }
        if let (true, Some(e)) = (self.embed_frame, &plan.embed) {
            out.push(e.size);
        }
        if let (true, Some(c)) = (self.embed_crops, &plan.crop_embed) {
            out.push(c.size);
        }
        out
    }
}

/// The photo's branch states and outputs as they settle.
struct Answers {
    states: BTreeMap<String, BranchState>,
    stopped: Vec<&'static str>,
    out: Vec<(&'static str, Tensor)>,
}

impl Answers {
    /// Branch `name` ends with `state`; what needs it ends `dependency_failed` (`_Rounds._stop`).
    fn stop<D: Device>(
        &mut self,
        calls: &mut PhotoCalls<D>,
        name: &'static str,
        state: BranchState,
    ) {
        let dependents: &[&'static str] = if name == "detect" {
            &["embed_crops"]
        } else {
            &[]
        };
        for (branch, st) in std::iter::once((name, state)).chain(
            dependents
                .iter()
                .map(|d| (*d, BranchState::Error(DEPENDENCY_FAILED))),
        ) {
            if self.stopped.contains(&branch) {
                continue;
            }
            self.stopped.push(branch);
            calls.cancel(branch);
            self.states.insert(branch.to_owned(), st);
        }
    }

    fn running(&self, name: &str) -> bool {
        !self.stopped.contains(&name)
    }
}

impl<D: Device, B: FrameBudget> PhotoFlow<D, B>
where
    B::Permit: Sync,
{
    /// The answer for one admitted photo.
    pub async fn process(&self, photo: Photo) -> Result<PhotoAnswer, PipelineError> {
        let mut clock = StageClock::new(Arc::clone(&photo.request), Arc::clone(&self.metrics));
        clock.add(QUEUE, photo.request.arrived().elapsed().as_secs_f64());
        let plan = photo.plan.clone();
        let answer = match self.run(&mut clock, photo).await {
            Ok(answer) => answer,
            Err(Stop::Status(status)) => refused(&plan, status),
            Err(Stop::Fail(e)) => return Err(e),
        };
        self.metrics
            .outcomes
            .get_or_create(&ByImage {
                status: answer.status.clone(),
                decode_route: answer.decode_route.clone(),
            })
            .inc();
        Ok(answer)
    }

    async fn run(&self, clock: &mut StageClock, photo: Photo) -> Result<PhotoAnswer, Stop> {
        let s = &self.settings;
        if self.device.lost() {
            return Err(Stop::Fail(PipelineError::ContextLost(
                "the CUDA context of this engine is lost; it is restarting".into(),
            )));
        }
        if photo.boxes.is_some() {
            return Err(engine(
                "stored boxes are not built in this engine yet (plan 17 N1a)",
            ));
        }
        let plan = &photo.plan;
        let runs = Runs::of(plan);
        let decided = clock.time(DECODE, || {
            route(&photo.image, s.min_megapixels, s.max_megapixels)
        });
        let header = match decided.decoder {
            Decoder::Gpu => decided.header,
            Decoder::Refused(status) => return Err(Stop::Status(status)),
            // the host route is N1a's: the API decodes such a photo itself
            Decoder::Cpu(_) => return Err(Stop::Status("decoder")),
        };
        if budget::need_bytes(&header) > s.budget_bytes {
            return Err(Stop::Status("over_budget"));
        }
        let reserve = budget::reserve_bytes(&header, &runs.widths(plan)).min(s.budget_bytes);
        let permit = clock
            .wait(
                BUDGET,
                self.budget.admit(reserve, Instant::now() + s.budget_wait),
            )
            .await;
        let waited = if permit.is_ok() {
            "admitted"
        } else {
            "timeout"
        };
        self.metrics
            .budget_waits
            .get_or_create(&ByOutcome {
                outcome: waited.into(),
            })
            .inc();
        let permit = permit.map_err(|o| Stop::Fail(PipelineError::Overload(o)))?;

        let (w, h) = oriented_size(&header);
        let request = Arc::clone(&photo.request);
        let cancel = request.cancel_token().clone();
        let deadline = tokio::time::Instant::from_std(request.clock.deadline);
        let mut calls = PhotoCalls::new(
            Arc::clone(&self.device),
            Arc::clone(&self.metrics),
            s.window,
            s.slab_wait,
            deadline,
        );
        let mut answers = Answers {
            states: unsupported(plan),
            stopped: Vec::new(),
            out: Vec::new(),
        };

        // round 1: the inputs cut from the frame alone, written by the decode job
        let mut writes = Vec::new();
        let mut round1 = Vec::new();
        let mut canvas = None;
        if let (true, Some(spec)) = (runs.detect, &plan.detector) {
            let c = detect::canvas(spec, w, h).map_err(engine)?;
            let slice = calls.stage(clock, c.input.bytes(), LETTERBOX).await?;
            writes.push(Write {
                input: slice,
                fill: Some(c.fill),
                resamples: vec![c.resample],
                stage: LETTERBOX,
            });
            round1.push((
                "detect",
                DETECT,
                spec.model.as_str(),
                c.input.clone(),
                detect::OUTPUT,
            ));
            canvas = Some(c);
        }
        if let (true, Some(spec)) = (runs.embed_frame, &plan.embed) {
            let (input, r) = embed::frame_square(spec, w, h).map_err(engine)?;
            let slice = calls.stage(clock, input.bytes(), EMBED_FRAME).await?;
            writes.push(Write {
                input: slice,
                fill: None,
                resamples: vec![r],
                stage: EMBED_FRAME,
            });
            round1.push((
                "embed_frame",
                EMBED_PE,
                spec.model.as_str(),
                input,
                embed::OUTPUT,
            ));
        }
        let job = DecodeJob {
            image: photo.image.slice(..header.end.min(photo.image.len())),
            orientation: header.orientation,
            size: (w, h),
            writes,
            hold: Box::new(permit),
            request: Arc::clone(&request),
        };
        clock.enter(DECODE);
        let started = Instant::now();
        let handle = self
            .device
            .decode(job, &cancel)
            .map_err(|e| Stop::Fail(job_failed(DECODE, &e)))?;
        let decoded = handle
            .wait()
            .await
            .map_err(|e| Stop::Fail(job_failed(DECODE, &e)))?;
        let lane_wait = started.elapsed().as_secs_f64();
        let (frame, inputs) = match decoded? {
            Decoded::Refused { status, stages } => {
                settle_job_time(clock, lane_wait, stages);
                return Err(Stop::Status(status));
            }
            Decoded::Frame {
                frame,
                inputs,
                stages,
            } => {
                settle_job_time(clock, lane_wait, stages);
                (frame, inputs)
            }
        };
        let mut sent: Vec<(&'static str, CallId)> = Vec::new();
        for ((owner, stage, model, input, output), slice) in round1.into_iter().zip(inputs) {
            sent.push((
                owner,
                calls.send(owner, stage, model, input, slice, vec![output]),
            ));
        }
        let call_of = |owner: &str| sent.iter().find(|(o, _)| *o == owner).map(|(_, id)| *id);

        // settle: the detector's answer
        let mut found = None;
        if let (Some(id), Some(c)) = (call_of("detect"), &canvas) {
            match calls.result(clock, id).await? {
                Answer::Outputs(out) => {
                    let d = clock
                        .time(DETECT, || detect::settle(&out, &c.letterbox, w, h))
                        .map_err(engine)?;
                    answers.out.extend(detect::outputs(&d));
                    found = Some(d);
                }
                Answer::Rejected(why) => {
                    tracing::warn!("detect: model call failed: {why}");
                    answers.stop(&mut calls, "detect", BranchState::Error(MODEL_ERROR));
                }
            }
        }
        if found.is_none() && runs.embed_crops {
            answers.stop(
                &mut calls,
                "embed_crops",
                BranchState::Error(DEPENDENCY_FAILED),
            );
        }

        // round 2: crop squares and blur sums on the launch pool, crop JPEGs on the finish pool
        let mut crop_calls = Vec::new();
        let mut picked = Vec::new();
        let mut blur = None;
        let mut jpegs = Vec::new();
        if let Some(d) = &found {
            let (summed, clamped) = detect::blur_rects(d, w, h);
            let mut chunks = Vec::new();
            if let (true, Some(spec)) = (answers.running("embed_crops"), &plan.crop_embed) {
                let cut = embed::crop_plan(
                    spec,
                    plan.embed_classes.as_ref(),
                    &d.rects,
                    Some(&d.classes),
                )
                .map_err(engine)?;
                picked = cut.picked;
                chunks = cut.chunks;
            }
            let model = plan.crop_embed.as_ref().map_or("", |c| c.model.as_str());
            let mut chunks = chunks.into_iter().peekable();
            let mut pending_blur = Some(summed);
            while chunks.peek().is_some() || pending_blur.is_some() {
                let mut batch = Vec::new();
                let mut staged = Vec::new();
                while let Some(next) = chunks.peek() {
                    let bytes = next.input.bytes();
                    if !batch.is_empty() && !calls.within_window(bytes) {
                        break;
                    }
                    let slice = calls.stage(clock, bytes, EMBED_CROPS).await?;
                    if let Some(chunk) = chunks.next() {
                        staged.push(Write {
                            input: slice,
                            fill: Some(chunk.fill),
                            resamples: chunk.resamples.clone(),
                            stage: EMBED_CROPS,
                        });
                        batch.push(chunk);
                    }
                }
                let job = LaunchJob {
                    frame: Arc::clone(&frame),
                    writes: staged,
                    blur: pending_blur.take().unwrap_or_default(),
                    request: Arc::clone(&request),
                };
                let first = blur.is_none();
                clock.enter(if batch.is_empty() { BLUR } else { EMBED_CROPS });
                let started = Instant::now();
                let launched = self
                    .device
                    .launch(job, &cancel)
                    .map_err(|e| Stop::Fail(job_failed(EMBED_CROPS, &e)))?
                    .wait()
                    .await
                    .map_err(|e| Stop::Fail(job_failed(EMBED_CROPS, &e)))??;
                settle_job_time(clock, started.elapsed().as_secs_f64(), launched.stages);
                if first {
                    blur = Some(detect::blur_sums(w, h, &clamped, &launched.blur).map_err(engine)?);
                }
                for (chunk, slice) in batch.into_iter().zip(launched.inputs) {
                    crop_calls.push(calls.send(
                        "embed_crops",
                        EMBED_PE,
                        model,
                        chunk.input,
                        slice,
                        vec![embed::OUTPUT],
                    ));
                }
                if first && runs.crops {
                    for rect in &d.rects {
                        jpegs.push(
                            self.device
                                .crop_jpeg(Arc::clone(&frame), *rect, &cancel)
                                .map_err(|e| Stop::Fail(job_failed(CROP_COPY, &e)))?,
                        );
                    }
                }
            }
        }
        // the photo's own hold on the frame ends here; jobs still running keep it to their wait
        drop(frame);

        // finish
        if let Some(sums) = blur {
            answers.out.push(("blur_sums", sums));
        }
        if let Some(id) = call_of("embed_frame") {
            match calls.result(clock, id).await? {
                Answer::Outputs(out) => {
                    answers.out.push((
                        "whole_embedding",
                        embed::whole_embedding(&out).map_err(engine)?,
                    ));
                }
                Answer::Rejected(why) => {
                    tracing::warn!("embed_frame: model call failed: {why}");
                    answers.stop(&mut calls, "embed_frame", BranchState::Error(MODEL_ERROR));
                }
            }
        }
        if found.is_some() && answers.running("embed_crops") && runs.embed_crops {
            let mut vectors = Vec::with_capacity(crop_calls.len());
            for id in crop_calls {
                match calls.result(clock, id).await? {
                    Answer::Outputs(out) => vectors.push(out),
                    Answer::Rejected(why) => {
                        tracing::warn!("embed_crops: model call failed: {why}");
                        answers.stop(&mut calls, "embed_crops", BranchState::Error(MODEL_ERROR));
                        break;
                    }
                }
            }
            if answers.running("embed_crops") {
                answers
                    .out
                    .extend(embed::crop_outputs(&picked, &vectors).map_err(engine)?);
            }
        }
        if found.is_some() && runs.crops {
            let (mut copy, mut encode) = (0.0, 0.0);
            let mut data = Vec::with_capacity(jpegs.len());
            for job in jpegs {
                let crop = job
                    .wait()
                    .await
                    .map_err(|e| Stop::Fail(job_failed(CROP_JPEG, &e)))??;
                copy += crop.copy_seconds;
                encode += crop.encode_seconds;
                data.push(crop.data);
            }
            clock.add(CROP_COPY, copy);
            if !data.is_empty() {
                clock.add(CROP_JPEG, encode);
            }
            answers.out.push(("crop_jpegs", Tensor::Bytes(data)));
        }

        for name in BUILT {
            let built = match name {
                "detect" => runs.detect,
                "embed_frame" => runs.embed_frame,
                _ => runs.embed_crops,
            };
            if built && answers.running(name) {
                answers.states.insert(name.to_owned(), BranchState::Ok);
            }
        }
        Ok(PhotoAnswer {
            status: "ok".into(),
            decode_route: "gpu".into(),
            branches: answers.states,
            outputs: ordered(plan, (w, h), answers.out),
        })
    }
}

/// A job's own stages, and the rest of the wait (the queue for a free thread) in the first.
fn settle_job_time(clock: &mut StageClock, waited: f64, stages: Vec<(&'static str, f64)>) {
    let inside: f64 = stages.iter().map(|(_, s)| s).sum();
    let first = stages.first().map_or(DECODE, |(s, _)| *s);
    clock.add(first, (waited - inside).max(0.0));
    clock.merge(stages);
}

/// `size`, then the plan's outputs in the wire module's order, those the photo has.
fn ordered(
    plan: &EnginePlan,
    (w, h): (u32, u32),
    mut out: Vec<(&'static str, Tensor)>,
) -> Vec<(String, Tensor)> {
    let size = Tensor::Int32 {
        shape: vec![2],
        data: vec![
            i32::try_from(w).unwrap_or(i32::MAX),
            i32::try_from(h).unwrap_or(i32::MAX),
        ],
    };
    let mut ordered = vec![("size".to_owned(), size)];
    for name in plan.expected_outputs() {
        if let Some(k) = out.iter().position(|(n, _)| *n == name) {
            let (_, t) = out.swap_remove(k);
            ordered.push((name.to_owned(), t));
        }
    }
    ordered
}

#[cfg(test)]
mod fake;
#[cfg(test)]
mod tests;
