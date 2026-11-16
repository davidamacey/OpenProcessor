//! One photo through the engine (`FrameEngine._process`, `_Rounds` and the branches of
//! `frame_engine/engine.py`, plan 17 section 2.2), as an async control task on the network
//! runtime that hands each piece of work to the thread that owns it:
//!
//! 1. Route by header (`decode`): the markers, then the end-of-image walk, before any bytes are
//!    reserved (`source`). The GPU route takes a JPEG nvJPEG decodes; every other readable image
//!    takes the host route (`crate::host`, probed here, decoded on the lane); what neither takes
//!    answers its status (`corrupt`, `huge`, `decoder`). A need above the whole budget is
//!    `over_budget`.
//! 2. Reserve the frame (`budget`): a worker slot and the frame's bytes, first come first served,
//!    bounded by `OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS`, then an overload.
//! 3. Round 1: the slices of the inputs that need only the frame (`letterbox`, `embed_frame`,
//!    the face detector's passes), except those of the branches the plan makes late (`late`),
//!    then one job on a decode lane: nvJPEG, the orientation and the colour step's gather (or the
//!    host decode and its upload), the size check, the fills and resamples into those slices, one
//!    wait for the lane's stream. Every round-1 call goes out.
//! 4. Settle: the detector's boxes as frame rectangles, or the request's stored boxes in their
//!    place (`cut`); the faces (anchor decode and NMS on the finish pool) unless they are late.
//! 5. The late round: each late branch's condition against the detector's classes; those that
//!    hold write their round-1 inputs from the frame by launch jobs and send their calls.
//! 6. Round 2: the crop squares in chunks of the plan's batch, staged within the photo's window
//!    (collecting its own oldest call when the next chunk would pass it), written with the blur
//!    sums by one job on the launch pool; their calls go out; the crop JPEGs start on the finish
//!    pool. Then the late branches' settle, and the face warps into recogniser batches.
//! 7. Finish: collect, outputs in the wire's order.
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
pub mod cut;
pub mod device;
mod faces;
pub mod late;
mod rounds;
pub mod source;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gpu_frame::CancelToken;

use self::calls::PhotoCalls;
use self::clock::{BUDGET, DECODE, QUEUE, StageClock};
use self::device::{DecodeJob, Decoded, Device, Write};
use self::rounds::Work;
use self::source::{Origin, Source};
use crate::admission::FrameBudget;
use crate::branches::BUILT;
use crate::colour::{Colour, ColourError, ColourStep, Table, Upload};
use crate::fault::{GpuFault, job_failed, note_memory_failure};
use crate::jpeg_header::route;
use crate::metrics::{ByImage, ByOutcome, Metrics};
use crate::pipeline::{Photo, PhotoAnswer, PipelineError, Tensor};
use crate::wire::EnginePlan;
use crate::wire::branch_status::{
    BranchState, DEPENDENCY_FAILED, IMAGE_FAILED, UNSUPPORTED_BRANCH,
};
use crate::wire::plan::Crops;

/// Why a photo stops before its branches finish: a photo status, or a failure of the request.
#[derive(Debug)]
pub enum Stop {
    /// A photo status that answers the photo (for example `over_budget` or `huge`) with every branch failed.
    Status(&'static str),
    /// A device allocation was refused: the photo answers `gpu_memory`; the text says what was
    /// asked and which limit refused it.
    Memory(String),
    /// The request itself fails with this error.
    Fail(PipelineError),
}

impl From<GpuFault> for Stop {
    fn from(f: GpuFault) -> Self {
        match f {
            GpuFault::OutOfMemory(detail) => Stop::Memory(detail),
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
    /// Smallest image, in megapixels, that goes to the GPU decoder; smaller ones decode on the host.
    pub min_megapixels: f64,
    /// Largest image, in megapixels, that is decoded at all; a larger one answers `huge`.
    pub max_megapixels: f64,
    /// `OP_FRAME_ENGINE_BUDGET_MIB` in bytes: a photo needing more answers `over_budget`.
    pub budget_bytes: u64,
    /// How long a photo waits for the frame budget, `OP_FRAME_ENGINE_BUDGET_WAIT_SECONDS`.
    pub budget_wait: Duration,
    /// How long a photo waits for room in the shared-memory region, `OP_FRAME_ENGINE_SLAB_WAIT_SECONDS`.
    pub slab_wait: Duration,
    /// Region bytes one photo may hold with calls in flight.
    pub window: usize,
}

/// The photo flow over a device and a frame budget.
pub struct PhotoFlow<D, B> {
    /// The card the jobs run on.
    pub device: Arc<D>,
    /// The decoded-frame budget a photo reserves from before decoding.
    pub budget: B,
    /// The limits the flow works to.
    pub settings: PhotoSettings,
    /// Where the flow's stage and outcome series are written.
    pub metrics: Arc<Metrics>,
    /// The colour step's profile kinds and tables, shared by every photo.
    pub colour: ColourStep,
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
    faces: bool,
    face_embed: bool,
}

impl Runs {
    fn of(plan: &EnginePlan) -> Self {
        Self {
            detect: plan.runs("detect") && plan.detector.is_some(),
            embed_frame: plan.runs("embed_frame") && plan.embed.is_some(),
            embed_crops: plan.runs("embed_crops") && plan.crop_embed.is_some(),
            crops: plan.crops == Crops::Inline,
            faces: plan.runs("faces") && plan.faces.is_some(),
            face_embed: plan.runs("face_embed") && plan.faces.is_some(),
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
        if let (true, Some(f)) = (self.faces, &plan.faces) {
            out.push(f.input_size);
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
        for (branch, st) in std::iter::once((name, state)).chain(
            late::dependents(name)
                .into_iter()
                .map(|d| (d, BranchState::Error(DEPENDENCY_FAILED))),
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

    /// Every built branch the plan runs that did not stop ends `ok`.
    fn mark_built(&mut self, runs: &Runs) {
        for name in BUILT {
            let built = match name {
                "detect" => runs.detect,
                "embed_frame" => runs.embed_frame,
                "embed_crops" => runs.embed_crops,
                "faces" => runs.faces,
                "face_embed" => runs.face_embed,
                _ => false,
            };
            if built && self.running(name) {
                self.states.insert(name.to_owned(), BranchState::Ok);
            }
        }
    }
}

impl<D: Device, B: FrameBudget> PhotoFlow<D, B>
where
    B::Permit: Sync,
{
    /// The answer for one admitted photo.
    ///
    /// # Errors
    ///
    /// The request fails as a whole: an overload, a lost CUDA context, an unreachable model server or an engine fault. A photo that is merely refused is an `Ok` answer with its status.
    pub async fn process(&self, photo: Photo) -> Result<PhotoAnswer, PipelineError> {
        let mut clock = StageClock::new(Arc::clone(&photo.request), Arc::clone(&self.metrics));
        clock.add(QUEUE, photo.request.arrived().elapsed().as_secs_f64());
        let plan = photo.plan.clone();
        let mut size = (0, 0);
        let answer = match self.run(&mut clock, photo, &mut size).await {
            Ok(answer) => answer,
            Err(Stop::Status(status)) => refused(&plan, status),
            Err(Stop::Memory(detail)) => {
                note_memory_failure("the photo's work", size, &detail);
                refused(&plan, "gpu_memory")
            }
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

    /// The route and the reservation of one photo: its source and the permit that holds its frame
    /// budget, or the status that refuses it.
    async fn admit(
        &self,
        clock: &mut StageClock,
        photo: &Photo,
        runs: &Runs,
    ) -> Result<(Source, B::Permit), Stop> {
        let s = &self.settings;
        let source = clock
            .time(DECODE, || {
                let decided = route(&photo.image, s.min_megapixels, s.max_megapixels);
                source::decide(
                    &photo.image,
                    decided,
                    s.max_megapixels,
                    &runs.widths(&photo.plan),
                )
            })
            .map_err(Stop::Status)?;
        if source.need > s.budget_bytes {
            return Err(Stop::Status("over_budget"));
        }
        let reserve = source.reserve.min(s.budget_bytes);
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
        Ok((source, permit))
    }

    /// The decode job on a lane: the frame and the slices of the round-1 writes.
    async fn decode(
        &self,
        work: &mut Work<'_, D>,
        source: &Source,
        permit: B::Permit,
        writes: Vec<Write<D::Input>>,
        colour: Option<Table>,
    ) -> Result<(Arc<D::Frame>, Vec<D::Input>), Stop> {
        let job = DecodeJob {
            image: source.image.clone(),
            origin: source.origin,
            size: work.size,
            writes,
            colour,
            hold: Box::new(permit),
            request: Arc::clone(&work.request),
        };
        work.clock.enter(DECODE);
        let started = Instant::now();
        let handle = self
            .device
            .decode(job, &work.cancel)
            .map_err(|e| Stop::Fail(job_failed(DECODE, &e)))?;
        let decoded = handle
            .wait()
            .await
            .map_err(|e| Stop::Fail(job_failed(DECODE, &e)))?;
        let lane_wait = started.elapsed().as_secs_f64();
        match decoded? {
            Decoded::Refused { status, stages } => {
                settle_job_time(work.clock, lane_wait, stages);
                Err(Stop::Status(status))
            }
            Decoded::Frame {
                frame,
                inputs,
                stages,
            } => {
                settle_job_time(work.clock, lane_wait, stages);
                Ok((frame, inputs))
            }
        }
    }

    async fn run(
        &self,
        clock: &mut StageClock,
        photo: Photo,
        seen: &mut (u32, u32),
    ) -> Result<PhotoAnswer, Stop> {
        let s = &self.settings;
        if self.device.lost() {
            return Err(Stop::Fail(PipelineError::ContextLost(
                "the CUDA context of this engine is lost; it is restarting".into(),
            )));
        }
        let plan = &photo.plan;
        let runs = Runs::of(plan);
        let (source, permit) = self.admit(clock, &photo, &runs).await?;
        let size = source.size;
        *seen = size;
        let request = Arc::clone(&photo.request);
        let calls = PhotoCalls::new(
            Arc::clone(&self.device),
            Arc::clone(&self.metrics),
            s.window,
            s.slab_wait,
            tokio::time::Instant::from_std(request.clock.deadline),
        );
        let mut work = Work {
            clock,
            calls,
            answers: Answers {
                states: unsupported(plan),
                stopped: Vec::new(),
                out: Vec::new(),
            },
            device: Arc::clone(&self.device),
            cancel: request.cancel_token().clone(),
            request,
            size,
        };

        let colour = self.colour_of(&source).await?;
        let late = late::late(plan, &BUILT);
        let mut round1 = work.stage_round1(plan, &runs, &late).await?;
        let table = match &colour {
            Some(Colour::Convert(t)) => Some(Arc::clone(t)),
            Some(Colour::Srgb | Colour::Ignored) | None => None,
        };
        let writes = std::mem::take(&mut round1.writes);
        let (frame, inputs) = self
            .decode(&mut work, &source, permit, writes, table)
            .await?;
        if let Some(colour) = &colour {
            self.metrics
                .colour
                .get_or_create(&ByOutcome {
                    outcome: colour.label().into(),
                })
                .inc();
        }
        let calls = std::mem::take(&mut round1.calls);
        let sent = work.send_round1(calls, inputs);
        let after = work
            .after_round1(&photo, &runs, round1, sent, &late, frame)
            .await?;
        if let Some(r) = after.faces {
            work.finish_faces(&runs, r).await?;
        }
        work.finish(&runs, after.cut, after.round2, after.embed_frame)
            .await?;
        Ok(PhotoAnswer {
            status: "ok".into(),
            decode_route: source.decode_route,
            branches: work.answers.states,
            outputs: ordered(plan, size, work.answers.out),
        })
    }
}

impl<D: Device, B: FrameBudget> PhotoFlow<D, B> {
    /// What the colour step does with the photo: its table built once per profile on the finish
    /// pool (the job's handle is kept by a task of its own, since a dropped handle tells the pool
    /// the job is unwanted), waited for here without holding any pool thread.
    ///
    /// `None` on the host route, whose frame no colour step touches.
    async fn colour_of(&self, source: &Source) -> Result<Option<Colour>, Stop> {
        if source.origin == Origin::Host {
            return Ok(None);
        }
        let device = Arc::clone(&self.device);
        let upload: Upload = Arc::new(move |host: &[u8]| device.colour_table(host));
        let device = Arc::clone(&self.device);
        let spawn = move |job: Box<dyn FnOnce() + Send>| {
            let handle = device
                .host(job, &CancelToken::new())
                .map_err(|e| GpuFault::Failed(format!("colour table build: {e}")))?;
            tokio::spawn(async move { drop(handle.wait().await) });
            Ok(())
        };
        self.colour
            .lookup(source.icc.as_deref(), source.components, &spawn, &upload)
            .await
            .map(Some)
            .map_err(|e| match e {
                ColourError::Overload(o) => Stop::Fail(PipelineError::Overload(o)),
                ColourError::Fault(f) => f.into(),
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
