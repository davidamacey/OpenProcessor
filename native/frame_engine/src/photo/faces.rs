//! The face branches' steps in one photo's flow (`frame_engine/branches/faces.py` as
//! `_Rounds` runs it): the detector passes staged in round 1, the settle (anchor decode and NMS
//! on the finish pool), round 2 (the face warps into recogniser batches and the face blur sums on
//! the launch pool) and the finish (the vectors).
//!
//! A rejected detector call fails `faces` (`model_error`) and `face_embed` with it
//! (`dependency_failed`); a rejected recogniser call fails `face_embed` only. A photo with no face
//! makes no recogniser call and no blur launch.

use std::iter::Peekable;
use std::sync::Arc;
use std::time::Instant;

use super::calls::{Answer, CallId};
use super::clock::{FACE_ALIGN, FACE_BLUR, FACE_DETECT, FACE_EMBED, FACE_LETTERBOX};
use super::device::{Device, LaunchJob, Warp, WarpKind, Write};
use super::rounds::{Round1, Work};
use super::{Runs, Stop, engine, job_failed, settle_job_time};
use crate::branches::faces::embed::{self as face_vectors, Chunk};
use crate::branches::faces::scrfd::{Faces, OUTPUT_NAMES};
use crate::branches::faces::{self, DetCall};
use crate::pipeline::Tensor;
use crate::wire::EnginePlan;
use crate::wire::branch_status::{BranchState, MODEL_ERROR};
use crate::wire::plan::FaceSpec;

/// What the face round 2 produced: the recogniser calls in flight, which faces they embed, and
/// the face blur tensor.
pub(super) struct FaceRound2 {
    calls: Vec<CallId>,
    embedded: Vec<bool>,
    blur: Tensor,
}

impl<D: Device> Work<'_, D> {
    /// Round 1 (`faces`): one slice per detector call, zero-filled, its passes resampled into it
    /// by the decode job.
    pub async fn stage_faces<'p>(
        &mut self,
        spec: &'p FaceSpec,
        round: &mut Round1<'p, D::Input>,
    ) -> Result<(), Stop> {
        let (w, h) = self.size;
        let calls = faces::detector_calls(spec, w, h).map_err(engine)?;
        for c in &calls {
            let bytes = c.input.bytes();
            let slice = self.calls.stage(self.clock, bytes, FACE_LETTERBOX).await?;
            round.writes.push(Write {
                input: slice,
                fill: Some(0),
                resamples: c.resamples.clone(),
                stage: FACE_LETTERBOX,
            });
            round.calls.push((
                "faces",
                FACE_DETECT,
                spec.detector.as_str(),
                c.input.clone(),
                &OUTPUT_NAMES,
            ));
        }
        round.faces = calls;
        Ok(())
    }

    /// Settle (`faces`): every detector answer, then the decode and NMS on the finish pool; the
    /// face tensors go to the answer. `None` when the branch did not run or a call was rejected.
    pub async fn settle_faces(
        &mut self,
        plan: &EnginePlan,
        calls: Vec<DetCall>,
        ids: Vec<CallId>,
    ) -> Result<Option<Faces>, Stop> {
        let Some(spec) = plan.faces.clone().filter(|_| !ids.is_empty()) else {
            return Ok(None);
        };
        let mut answers = Vec::with_capacity(ids.len());
        for id in ids {
            match self.calls.result(self.clock, id).await? {
                Answer::Outputs(out) => answers.push(out),
                Answer::Rejected(why) => {
                    tracing::warn!("faces: model call failed: {why}");
                    let state = BranchState::Error(MODEL_ERROR);
                    self.answers.stop(&mut self.calls, "faces", state);
                    return Ok(None);
                }
            }
        }
        let (w, h) = self.size;
        self.clock.enter(FACE_DETECT);
        let started = Instant::now();
        let settled = self
            .device
            .host(
                move || faces::settle(&spec, w, h, &calls, &answers),
                &self.cancel,
            )
            .map_err(|e| Stop::Fail(job_failed(FACE_DETECT, &e)))?
            .wait()
            .await
            .map_err(|e| Stop::Fail(job_failed(FACE_DETECT, &e)))??;
        self.clock.add(FACE_DETECT, started.elapsed().as_secs_f64());
        let found = settled.map_err(engine)?;
        self.answers.out.extend(faces::outputs(&found));
        Ok(Some(found))
    }

    /// Round 2 (`face_embed`, and the face blur sums of `faces`): the warps in chunks of the
    /// plan's batch, staged within the photo's window, written with the blur sums by launch jobs;
    /// the recogniser calls go out.
    pub async fn round2_faces(
        &mut self,
        plan: &EnginePlan,
        runs: &Runs,
        found: &Faces,
        frame: &Arc<D::Frame>,
    ) -> Result<FaceRound2, Stop> {
        let (w, h) = self.size;
        let clamped = faces::blur_rects(found, w, h);
        let rects: Vec<[i64; 4]> = clamped.iter().flatten().copied().collect();
        let (mut calls, mut embedded, mut sums) = (Vec::new(), Vec::new(), Vec::new());
        let mut chunks = Vec::new();
        let recogniser = plan.faces.as_ref().map_or("", |f| f.recogniser.as_str());
        if let (true, true, Some(spec)) = (
            runs.face_embed,
            self.answers.running("face_embed"),
            &plan.faces,
        ) {
            let cut = face_vectors::plan(found, spec.batch).map_err(engine)?;
            embedded = cut.embedded;
            chunks = cut.chunks;
        }
        let mut chunks = chunks.into_iter().peekable();
        let mut pending_blur = (!rects.is_empty()).then_some(rects);
        while chunks.peek().is_some() || pending_blur.is_some() {
            let (batch, warps) = self.stage_warps(&mut chunks).await?;
            let blur = pending_blur.take();
            let summed = blur.is_some();
            let (warped, got) = self.launch_faces(frame, warps, blur).await?;
            if summed {
                sums = got;
            }
            for (chunk, slice) in batch.into_iter().zip(warped) {
                let outputs = vec![faces::OUTPUT];
                let id = self.calls.send(
                    "face_embed",
                    FACE_EMBED,
                    recogniser,
                    chunk.input,
                    slice,
                    outputs,
                );
                calls.push(id);
            }
        }
        Ok(FaceRound2 {
            calls,
            embedded,
            blur: faces::blur_sums(&clamped, &sums).map_err(engine)?,
        })
    }

    /// One launch job of face warps and (the first time) the face blur sums, waited for: the
    /// warped slices and the sums.
    async fn launch_faces(
        &mut self,
        frame: &Arc<D::Frame>,
        warps: Vec<Warp<D::Input>>,
        blur: Option<Vec<[i64; 4]>>,
    ) -> Result<(Vec<D::Input>, Vec<[i64; 2]>), Stop> {
        let stage = if warps.is_empty() {
            FACE_BLUR
        } else {
            FACE_ALIGN
        };
        let job = LaunchJob {
            frame: Arc::clone(frame),
            writes: Vec::new(),
            warps,
            blur: blur.unwrap_or_default(),
            blur_stage: FACE_BLUR,
            request: Arc::clone(&self.request),
        };
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
        Ok((launched.warped, launched.blur))
    }

    /// The warp chunks that fit the photo's window now (at least one when any is left), and the
    /// warps of their slices.
    async fn stage_warps(
        &mut self,
        chunks: &mut Peekable<std::vec::IntoIter<Chunk>>,
    ) -> Result<(Vec<Chunk>, Vec<Warp<D::Input>>), Stop> {
        let mut batch = Vec::new();
        let mut warps = Vec::new();
        while let Some(next) = chunks.peek() {
            let bytes = next.input.bytes();
            if !batch.is_empty() && !self.calls.within_window(bytes) {
                break;
            }
            let slice = self.calls.stage(self.clock, bytes, FACE_ALIGN).await?;
            if let Some(chunk) = chunks.next() {
                warps.push(Warp {
                    input: slice,
                    kind: WarpKind::Faces(chunk.warps.clone()),
                    stage: FACE_ALIGN,
                });
                batch.push(chunk);
            }
        }
        Ok((batch, warps))
    }

    /// Finish (`faces`, `face_embed`): the face blur sums, then the vectors of every recogniser
    /// call, or `face_embed` failed at the first rejected call.
    pub async fn finish_faces(&mut self, runs: &Runs, round: FaceRound2) -> Result<(), Stop> {
        self.answers.out.push(("face_blur_sums", round.blur));
        if !(runs.face_embed && self.answers.running("face_embed")) {
            return Ok(());
        }
        let mut vectors = Vec::with_capacity(round.calls.len());
        for id in round.calls {
            match self.calls.result(self.clock, id).await? {
                Answer::Outputs(out) => vectors.push(out),
                Answer::Rejected(why) => {
                    tracing::warn!("face_embed: model call failed: {why}");
                    let state = BranchState::Error(MODEL_ERROR);
                    self.answers.stop(&mut self.calls, "face_embed", state);
                    return Ok(());
                }
            }
        }
        let tensors = face_vectors::outputs(&round.embedded, &vectors).map_err(engine)?;
        self.answers.out.extend(tensors);
        Ok(())
    }
}
