//! The steps of one photo between its admission and its answer, one function per step of the
//! module documentation (`photo/mod.rs`): the round-1 slices, the settle of the detector, round 2
//! in chunks, and the finish. They share one [`Work`]: the photo's clock, calls, answers and
//! cancellation.

use std::iter::Peekable;
use std::sync::Arc;
use std::time::Instant;

use gpu_frame::{CancelToken, JobHandle};

use super::calls::{Answer, CallId, PhotoCalls};
use super::clock::{
    BLUR, CROP_COPY, CROP_JPEG, DETECT, EMBED_CROPS, EMBED_FRAME, EMBED_PE, LETTERBOX, StageClock,
};
use super::cut::Cut;
use super::device::{Device, LaunchJob, Launched, Write};
use super::faces::FaceRound2;
use super::late::LateCall;
use super::{Answers, Runs, Stop, engine, job_failed, settle_job_time};
use crate::branches::crops::CropJpeg;
use crate::branches::{InputSpec, detect, embed, faces};
use crate::fault::GpuFault;
use crate::pipeline::{Photo, RequestState, Tensor};
use crate::wire::EnginePlan;
use crate::wire::branch_status::{BranchState, DEPENDENCY_FAILED, MODEL_ERROR};

/// A round-1 call to send once its input is written: owner, stage, model, input, output names.
pub(super) type Round1Call<'p> = (
    &'static str,
    &'static str,
    &'p str,
    InputSpec,
    &'static [&'static str],
);

/// What round 1 staged: the writes of the decode job, the calls they feed, the detector canvas
/// and the face detector's calls (their passes, in the order of their round-1 calls).
pub(super) struct Round1<'p, I> {
    pub writes: Vec<Write<I>>,
    pub calls: Vec<Round1Call<'p>>,
    pub canvas: Option<detect::Canvas>,
    pub faces: Vec<faces::DetCall>,
    /// The round-1 calls of late branches, sent after the conditions are checked.
    pub late: Vec<LateCall<'p>>,
}

/// What round 2 produced: the crop calls in flight, the detections picked for them, the blur
/// tensor and the crop JPEG jobs.
pub(super) struct Round2 {
    pub crop_calls: Vec<CallId>,
    pub picked: Vec<bool>,
    pub blur: Option<Tensor>,
    pub jpegs: Vec<JobHandle<Result<CropJpeg, GpuFault>>>,
}

/// What the steps after round 1 hand to the finish.
pub(super) struct AfterRound1 {
    /// Whether there were rectangles to cut (detected or stored).
    pub cut: bool,
    pub round2: Round2,
    pub faces: Option<FaceRound2>,
    pub embed_frame: Option<CallId>,
}

/// The id of `owner`'s call among those sent.
fn call_of(sent: &[(&str, CallId)], owner: &str) -> Option<CallId> {
    sent.iter().find(|(o, _)| *o == owner).map(|(_, id)| *id)
}

/// The ids of every call of `owner` among those sent, in order.
fn calls_of(sent: &[(&str, CallId)], owner: &str) -> Vec<CallId> {
    sent.iter()
        .filter(|(o, _)| *o == owner)
        .map(|(_, id)| *id)
        .collect()
}

/// One photo's shared state while its rounds run.
pub(super) struct Work<'a, D: Device> {
    pub clock: &'a mut StageClock,
    pub calls: PhotoCalls<D>,
    pub answers: Answers,
    pub device: Arc<D>,
    pub request: Arc<RequestState>,
    pub cancel: CancelToken,
    /// The oriented frame size.
    pub size: (u32, u32),
}

impl<D: Device> Work<'_, D> {
    /// Round 1: the slices of the inputs that need only the frame, and the writes that fill them.
    pub async fn stage_round1<'p>(
        &mut self,
        plan: &'p EnginePlan,
        runs: &Runs,
        late: &[&'static str],
    ) -> Result<Round1<'p, D::Input>, Stop> {
        let (w, h) = self.size;
        let mut round = Round1 {
            writes: Vec::new(),
            calls: Vec::new(),
            canvas: None,
            faces: Vec::new(),
            late: Vec::new(),
        };
        let mut entries = Vec::new();
        if let (true, Some(spec)) = (runs.detect, &plan.detector) {
            let c = detect::canvas(spec, w, h).map_err(engine)?;
            entries.push(LateCall {
                call: (
                    "detect",
                    DETECT,
                    spec.model.as_str(),
                    c.input.clone(),
                    &[detect::OUTPUT],
                ),
                fill: Some(c.fill),
                resamples: vec![c.resample],
                stage: LETTERBOX,
            });
            round.canvas = Some(c);
        }
        if let (true, Some(spec)) = (runs.embed_frame, &plan.embed) {
            let (input, r) = embed::frame_square(spec, w, h).map_err(engine)?;
            entries.push(LateCall {
                call: (
                    "embed_frame",
                    EMBED_PE,
                    spec.model.as_str(),
                    input,
                    &[embed::OUTPUT],
                ),
                fill: None,
                resamples: vec![r],
                stage: EMBED_FRAME,
            });
        }
        if let (true, Some(spec)) = (runs.faces, &plan.faces) {
            let (calls, detector) = self.face_entries(spec)?;
            entries.extend(calls);
            round.faces = detector;
        }
        for entry in entries {
            // a late branch's inputs wait for its condition (`late.rs`)
            if late.contains(&entry.call.0) {
                round.late.push(entry);
                continue;
            }
            let bytes = entry.call.3.bytes();
            let slice = self.calls.stage(self.clock, bytes, entry.stage).await?;
            round.writes.push(Write {
                input: slice,
                fill: entry.fill,
                resamples: entry.resamples,
                stage: entry.stage,
            });
            round.calls.push(entry.call);
        }
        Ok(round)
    }

    /// One launch job, waited for, its time counted from `stage` on: what it wrote.
    ///
    /// # Errors
    ///
    /// The job's failure, which fails the photo.
    pub async fn launch_waited(
        &mut self,
        stage: &'static str,
        job: LaunchJob<D::Frame, D::Input>,
    ) -> Result<Launched<D::Input>, Stop> {
        self.clock.enter(stage);
        let started = Instant::now();
        let launched = self
            .device
            .launch(job, &self.cancel)
            .map_err(|e| Stop::Fail(job_failed(stage, &e)))?
            .wait()
            .await
            .map_err(|e| Stop::Fail(job_failed(stage, &e)))??;
        settle_job_time(self.clock, started.elapsed().as_secs_f64(), launched.stages);
        Ok(Launched {
            stages: Vec::new(),
            ..launched
        })
    }

    /// Send every round-1 call with the slice its decode wrote.
    pub fn send_round1(
        &mut self,
        calls: Vec<Round1Call<'_>>,
        inputs: Vec<D::Input>,
    ) -> Vec<(&'static str, CallId)> {
        calls
            .into_iter()
            .zip(inputs)
            .map(|((owner, stage, model, input, outputs), slice)| {
                let id = self
                    .calls
                    .send(owner, stage, model, input, slice, outputs.to_vec());
                (owner, id)
            })
            .collect()
    }

    /// Everything after round 1 went out, in `_Rounds` order: the settle of the early branches
    /// (the detector's rectangles, or the stored boxes in their place; the faces unless late),
    /// the conditions and the late round 1, round 2 of the early branches, then the settle and
    /// round 2 of the late ones. The photo's own hold on `frame` ends here.
    ///
    /// # Errors
    ///
    /// A stop of any of those steps.
    pub async fn after_round1<'p>(
        &mut self,
        photo: &'p Photo,
        runs: &Runs,
        mut round1: Round1<'p, D::Input>,
        mut sent: Vec<(&'static str, CallId)>,
        late: &[&'static str],
        frame: Arc<D::Frame>,
    ) -> Result<AfterRound1, Stop> {
        let plan = &photo.plan;
        let cut = match &photo.boxes {
            Some(boxes) => Some(Cut::stored(boxes, self.size)?),
            None => self
                .settle_detect(call_of(&sent, "detect"), round1.canvas.as_ref())
                .await?
                .map(|d| Cut::detected(&d, self.size)),
        };
        if cut.is_none() && runs.embed_crops {
            self.stop_crops_without_boxes();
        }
        let faces_late = late.contains(&"faces");
        let mut faces = None;
        if !faces_late {
            let detector = std::mem::take(&mut round1.faces);
            faces = self
                .settle_faces(plan, detector, calls_of(&sent, "faces"))
                .await?;
        }
        let classes = cut.as_ref().and_then(|c| c.classes.as_deref());
        let running = self.check_conditions(plan, late, classes);
        let late_calls = round1
            .late
            .into_iter()
            .filter(|l| running.contains(&l.call.0))
            .collect();
        sent.extend(self.late_round1(&frame, late_calls).await?);
        let round2 = match &cut {
            Some(cut) => self.round2(plan, runs, cut, &frame).await?,
            None => Round2 {
                crop_calls: Vec::new(),
                picked: Vec::new(),
                blur: None,
                jpegs: Vec::new(),
            },
        };
        if faces_late {
            faces = self
                .settle_faces(plan, round1.faces, calls_of(&sent, "faces"))
                .await?;
        }
        let face_round = match &faces {
            Some(f) => Some(self.round2_faces(plan, runs, f, &frame).await?),
            None => None,
        };
        // jobs still running keep the frame to their wait
        drop(frame);
        Ok(AfterRound1 {
            cut: cut.is_some(),
            round2,
            faces: face_round,
            embed_frame: call_of(&sent, "embed_frame"),
        })
    }

    /// Settle: the detector's answer as frame rectangles; a rejected call fails the branch.
    pub async fn settle_detect(
        &mut self,
        id: Option<CallId>,
        canvas: Option<&detect::Canvas>,
    ) -> Result<Option<detect::Detections>, Stop> {
        let (Some(id), Some(c)) = (id, canvas) else {
            return Ok(None);
        };
        let (w, h) = self.size;
        match self.calls.result(self.clock, id).await? {
            Answer::Outputs(out) => {
                let d = self
                    .clock
                    .time(DETECT, || detect::settle(&out, &c.letterbox, w, h))
                    .map_err(engine)?;
                self.answers.out.extend(detect::outputs(&d));
                Ok(Some(d))
            }
            Answer::Rejected(why) => {
                tracing::warn!("detect: model call failed: {why}");
                self.answers
                    .stop(&mut self.calls, "detect", BranchState::Error(MODEL_ERROR));
                Ok(None)
            }
        }
    }

    /// Round 2: crop squares and blur sums on the launch pool, crop JPEGs on the finish pool,
    /// all cut from `cut` (the detector's rectangles or the stored boxes).
    pub async fn round2(
        &mut self,
        plan: &EnginePlan,
        runs: &Runs,
        cut: &Cut,
        frame: &Arc<D::Frame>,
    ) -> Result<Round2, Stop> {
        let (w, h) = self.size;
        let mut out = Round2 {
            crop_calls: Vec::new(),
            picked: Vec::new(),
            blur: None,
            jpegs: Vec::new(),
        };
        let mut chunks = Vec::new();
        if let (true, Some(spec)) = (self.answers.running("embed_crops"), &plan.crop_embed) {
            let planned = embed::crop_plan(
                spec,
                plan.embed_classes.as_ref(),
                &cut.rects,
                cut.classes.as_deref(),
            )
            .map_err(engine)?;
            out.picked = planned.picked;
            chunks = planned.chunks;
        }
        let model = plan.crop_embed.as_ref().map_or("", |c| c.model.as_str());
        let mut chunks = chunks.into_iter().peekable();
        let mut pending_blur = cut.blur.as_ref().map(|(summed, _)| summed.clone());
        let mut jpegs_started = false;
        while chunks.peek().is_some() || pending_blur.is_some() {
            let (batch, staged) = self.stage_batch(&mut chunks).await?;
            let first = pending_blur.is_some();
            let job = LaunchJob {
                frame: Arc::clone(frame),
                writes: staged,
                warps: Vec::new(),
                blur: pending_blur.take().unwrap_or_default(),
                blur_stage: BLUR,
                request: Arc::clone(&self.request),
            };
            self.clock
                .enter(if batch.is_empty() { BLUR } else { EMBED_CROPS });
            let started = Instant::now();
            let launched = self
                .device
                .launch(job, &self.cancel)
                .map_err(|e| Stop::Fail(job_failed(EMBED_CROPS, &e)))?
                .wait()
                .await
                .map_err(|e| Stop::Fail(job_failed(EMBED_CROPS, &e)))??;
            settle_job_time(self.clock, started.elapsed().as_secs_f64(), launched.stages);
            if let (true, Some((_, clamped))) = (first, &cut.blur) {
                out.blur = Some(detect::blur_sums(w, h, clamped, &launched.blur).map_err(engine)?);
            }
            for (chunk, slice) in batch.into_iter().zip(launched.inputs) {
                out.crop_calls.push(self.calls.send(
                    "embed_crops",
                    EMBED_PE,
                    model,
                    chunk.input,
                    slice,
                    vec![embed::OUTPUT],
                ));
            }
            if !jpegs_started {
                out.jpegs = self.start_jpegs(runs, &cut.rects, frame)?;
                jpegs_started = true;
            }
        }
        // stored boxes with no crop squares launch nothing: their JPEGs start here
        if !jpegs_started {
            out.jpegs = self.start_jpegs(runs, &cut.rects, frame)?;
        }
        Ok(out)
    }

    /// One crop JPEG per rectangle on the finish pool, when the plan asks for them.
    fn start_jpegs(
        &self,
        runs: &Runs,
        rects: &[[i64; 4]],
        frame: &Arc<D::Frame>,
    ) -> Result<Vec<JobHandle<Result<CropJpeg, GpuFault>>>, Stop> {
        if !runs.crops {
            return Ok(Vec::new());
        }
        rects
            .iter()
            .map(|rect| {
                self.device
                    .crop_jpeg(Arc::clone(frame), *rect, &self.cancel)
                    .map_err(|e| Stop::Fail(job_failed(CROP_COPY, &e)))
            })
            .collect()
    }

    /// The chunks that fit the photo's window now (at least one when any is left), and the writes
    /// of their slices.
    async fn stage_batch(
        &mut self,
        chunks: &mut Peekable<std::vec::IntoIter<embed::Chunk>>,
    ) -> Result<(Vec<embed::Chunk>, Vec<Write<D::Input>>), Stop> {
        let mut batch = Vec::new();
        let mut staged = Vec::new();
        while let Some(next) = chunks.peek() {
            let bytes = next.input.bytes();
            if !batch.is_empty() && !self.calls.within_window(bytes) {
                break;
            }
            let slice = self.calls.stage(self.clock, bytes, EMBED_CROPS).await?;
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
        Ok((batch, staged))
    }

    /// Finish: collect what is in flight, then mark the branches that ended well.
    pub async fn finish(
        &mut self,
        runs: &Runs,
        detected: bool,
        round2: Round2,
        embed_frame: Option<CallId>,
    ) -> Result<(), Stop> {
        if let Some(sums) = round2.blur {
            self.answers.out.push(("blur_sums", sums));
        }
        if let Some(id) = embed_frame {
            self.finish_embed_frame(id).await?;
        }
        if detected && self.answers.running("embed_crops") && runs.embed_crops {
            self.finish_embed_crops(round2.crop_calls, &round2.picked)
                .await?;
        }
        if detected && runs.crops {
            self.finish_crops(round2.jpegs).await?;
        }
        self.answers.mark_built(runs);
        Ok(())
    }

    /// Finish: the frame encoder's vector, or the branch failed.
    async fn finish_embed_frame(&mut self, id: CallId) -> Result<(), Stop> {
        match self.calls.result(self.clock, id).await? {
            Answer::Outputs(out) => {
                self.answers.out.push((
                    "whole_embedding",
                    embed::whole_embedding(&out).map_err(engine)?,
                ));
            }
            Answer::Rejected(why) => {
                tracing::warn!("embed_frame: model call failed: {why}");
                self.answers.stop(
                    &mut self.calls,
                    "embed_frame",
                    BranchState::Error(MODEL_ERROR),
                );
            }
        }
        Ok(())
    }

    /// Finish: the crop vectors of every chunk, or the branch failed at the first rejected call.
    async fn finish_embed_crops(
        &mut self,
        crop_calls: Vec<CallId>,
        picked: &[bool],
    ) -> Result<(), Stop> {
        let mut vectors = Vec::with_capacity(crop_calls.len());
        for id in crop_calls {
            match self.calls.result(self.clock, id).await? {
                Answer::Outputs(out) => vectors.push(out),
                Answer::Rejected(why) => {
                    tracing::warn!("embed_crops: model call failed: {why}");
                    self.answers.stop(
                        &mut self.calls,
                        "embed_crops",
                        BranchState::Error(MODEL_ERROR),
                    );
                    break;
                }
            }
        }
        if self.answers.running("embed_crops") {
            self.answers
                .out
                .extend(embed::crop_outputs(picked, &vectors).map_err(engine)?);
        }
        Ok(())
    }

    /// Finish: wait for the crop JPEGs and add their copy and encode seconds to the clock.
    async fn finish_crops(
        &mut self,
        jpegs: Vec<JobHandle<Result<CropJpeg, GpuFault>>>,
    ) -> Result<(), Stop> {
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
        self.clock.add(CROP_COPY, copy);
        if !data.is_empty() {
            self.clock.add(CROP_JPEG, encode);
        }
        self.answers.out.push(("crop_jpegs", Tensor::Bytes(data)));
        Ok(())
    }

    /// Whether a failed detect stops what needs its boxes.
    pub fn stop_crops_without_boxes(&mut self) {
        self.answers.stop(
            &mut self.calls,
            "embed_crops",
            BranchState::Error(DEPENDENCY_FAILED),
        );
    }
}
