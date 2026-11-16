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
use super::device::{Device, LaunchJob, Write};
use super::{Answers, Runs, Stop, engine, job_failed, settle_job_time};
use crate::branches::crops::CropJpeg;
use crate::branches::{InputSpec, detect, embed, faces};
use crate::fault::GpuFault;
use crate::pipeline::{RequestState, Tensor};
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
}

/// What round 2 produced: the crop calls in flight, the detections picked for them, the blur
/// tensor and the crop JPEG jobs.
pub(super) struct Round2 {
    pub crop_calls: Vec<CallId>,
    pub picked: Vec<bool>,
    pub blur: Option<Tensor>,
    pub jpegs: Vec<JobHandle<Result<CropJpeg, GpuFault>>>,
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
    ) -> Result<Round1<'p, D::Input>, Stop> {
        let (w, h) = self.size;
        let mut round = Round1 {
            writes: Vec::new(),
            calls: Vec::new(),
            canvas: None,
            faces: Vec::new(),
        };
        if let (true, Some(spec)) = (runs.detect, &plan.detector) {
            let c = detect::canvas(spec, w, h).map_err(engine)?;
            let slice = self
                .calls
                .stage(self.clock, c.input.bytes(), LETTERBOX)
                .await?;
            round.writes.push(Write {
                input: slice,
                fill: Some(c.fill),
                resamples: vec![c.resample],
                stage: LETTERBOX,
            });
            round.calls.push((
                "detect",
                DETECT,
                spec.model.as_str(),
                c.input.clone(),
                &[detect::OUTPUT],
            ));
            round.canvas = Some(c);
        }
        if let (true, Some(spec)) = (runs.embed_frame, &plan.embed) {
            let (input, r) = embed::frame_square(spec, w, h).map_err(engine)?;
            let slice = self
                .calls
                .stage(self.clock, input.bytes(), EMBED_FRAME)
                .await?;
            round.writes.push(Write {
                input: slice,
                fill: None,
                resamples: vec![r],
                stage: EMBED_FRAME,
            });
            round.calls.push((
                "embed_frame",
                EMBED_PE,
                spec.model.as_str(),
                input,
                &[embed::OUTPUT],
            ));
        }
        if let (true, Some(spec)) = (runs.faces, &plan.faces) {
            self.stage_faces(spec, &mut round).await?;
        }
        Ok(round)
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

    /// Round 2: crop squares and blur sums on the launch pool, crop JPEGs on the finish pool.
    pub async fn round2(
        &mut self,
        plan: &EnginePlan,
        runs: &Runs,
        d: &detect::Detections,
        frame: &Arc<D::Frame>,
    ) -> Result<Round2, Stop> {
        let (w, h) = self.size;
        let (summed, clamped) = detect::blur_rects(d, w, h);
        let mut out = Round2 {
            crop_calls: Vec::new(),
            picked: Vec::new(),
            blur: None,
            jpegs: Vec::new(),
        };
        let mut chunks = Vec::new();
        if let (true, Some(spec)) = (self.answers.running("embed_crops"), &plan.crop_embed) {
            let cut = embed::crop_plan(
                spec,
                plan.embed_classes.as_ref(),
                &d.rects,
                Some(&d.classes),
            )
            .map_err(engine)?;
            out.picked = cut.picked;
            chunks = cut.chunks;
        }
        let model = plan.crop_embed.as_ref().map_or("", |c| c.model.as_str());
        let mut chunks = chunks.into_iter().peekable();
        let mut pending_blur = Some(summed);
        while chunks.peek().is_some() || pending_blur.is_some() {
            let (batch, staged) = self.stage_batch(&mut chunks).await?;
            let job = LaunchJob {
                frame: Arc::clone(frame),
                writes: staged,
                warps: Vec::new(),
                blur: pending_blur.take().unwrap_or_default(),
                blur_stage: BLUR,
                request: Arc::clone(&self.request),
            };
            let first = out.blur.is_none();
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
            if first {
                out.blur = Some(detect::blur_sums(w, h, &clamped, &launched.blur).map_err(engine)?);
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
            if first && runs.crops {
                for rect in &d.rects {
                    out.jpegs.push(
                        self.device
                            .crop_jpeg(Arc::clone(frame), *rect, &self.cancel)
                            .map_err(|e| Stop::Fail(job_failed(CROP_COPY, &e)))?,
                    );
                }
            }
        }
        Ok(out)
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
