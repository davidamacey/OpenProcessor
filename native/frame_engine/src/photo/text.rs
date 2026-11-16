//! The text branch's steps in one photo's flow (`frame_engine/branches/ocr.py` as `_Rounds` runs
//! it): the detector input staged in round 1, the settle (DB boxes on the finish pool), round 2
//! (the line warps of each recogniser batch on the launch pool, staged within the photo's window,
//! and the recogniser calls) and the finish (CTC, every line back in reading order).
//!
//! A rejected detector or recogniser call fails `ocr` (`model_error`) and it answers no text
//! tensor, as the Python branch whose hook raised; a photo with no text makes no recogniser call.

use std::iter::Peekable;
use std::sync::Arc;
use std::time::Instant;

use super::calls::{Answer, CallId};
use super::clock::{OCR_BOXES, OCR_CROPS, OCR_DECODE, OCR_DETECT, OCR_PREP, OCR_REC};
use super::device::{Device, LaunchJob, Warp, WarpKind};
use super::late::LateCall;
use super::rounds::Work;
use super::{Stop, engine, job_failed};
use crate::branches::ocr::ctc::CtcDecoder;
use crate::branches::ocr::{self, LineBatch, Settled};
use crate::wire::EnginePlan;
use crate::wire::branch_status::{BranchState, MODEL_ERROR};
use crate::wire::plan::OcrSpec;

/// What the text round 2 produced: the kept lines and, per recogniser call, the lines it reads.
pub(super) struct TextRound2 {
    settled: Settled,
    calls: Vec<(Vec<usize>, CallId)>,
}

impl<D: Device> Work<'_, D> {
    /// Round 1 (`ocr`): the detector's input, the frame resized into BGR planes (by the decode
    /// job, or by the late round when the plan makes `ocr` late).
    pub fn ocr_entry<'p>(&self, spec: &'p OcrSpec) -> Result<LateCall<'p>, Stop> {
        let (w, h) = self.size;
        let (input, resample) = ocr::det_canvas(spec, w, h).map_err(engine)?;
        Ok(LateCall {
            call: (
                "ocr",
                OCR_DETECT,
                spec.detector.as_str(),
                input,
                &ocr::DET_OUTPUTS,
            ),
            fill: None,
            resamples: vec![resample],
            stage: OCR_PREP,
        })
    }

    /// Settle (`ocr`): the probability map's boxes on the finish pool. `None` when the branch did
    /// not run or its call was rejected.
    pub async fn settle_ocr(
        &mut self,
        plan: &EnginePlan,
        id: Option<CallId>,
    ) -> Result<Option<Settled>, Stop> {
        let (Some(id), Some(spec)) = (id, plan.ocr.clone()) else {
            return Ok(None);
        };
        let answer = match self.calls.result(self.clock, id).await? {
            Answer::Outputs(out) => out,
            Answer::Rejected(why) => {
                tracing::warn!("ocr: model call failed: {why}");
                let state = BranchState::Error(MODEL_ERROR);
                self.answers.stop(&mut self.calls, "ocr", state);
                return Ok(None);
            }
        };
        let (w, h) = self.size;
        self.clock.enter(OCR_BOXES);
        let started = Instant::now();
        let settled = self
            .device
            .host(move || ocr::settle(&answer, &spec, w, h), &self.cancel)
            .map_err(|e| Stop::Fail(job_failed(OCR_BOXES, &e)))?
            .wait()
            .await
            .map_err(|e| Stop::Fail(job_failed(OCR_BOXES, &e)))??;
        self.clock.add(OCR_BOXES, started.elapsed().as_secs_f64());
        settled.map(Some).map_err(engine)
    }

    /// Round 2 (`ocr`): the batches by width, their warps staged within the photo's window and
    /// written by launch jobs; the recogniser calls go out.
    pub async fn round2_ocr(
        &mut self,
        plan: &EnginePlan,
        settled: Settled,
        frame: &Arc<D::Frame>,
    ) -> Result<TextRound2, Stop> {
        let Some(spec) = &plan.ocr else {
            return Err(engine("ocr round 2 without its settings"));
        };
        let batches = self
            .clock
            .time(OCR_CROPS, || ocr::line_batches(&settled, spec))
            .map_err(engine)?;
        let mut calls = Vec::with_capacity(batches.len());
        let mut batches = batches.into_iter().peekable();
        while batches.peek().is_some() {
            let (batch, warps) = self.stage_lines(&mut batches).await?;
            let warped = self.launch_lines(frame, warps).await?;
            for (b, slice) in batch.into_iter().zip(warped) {
                let outputs = ocr::REC_OUTPUTS.to_vec();
                let model = spec.recogniser.as_str();
                let id = self
                    .calls
                    .send("ocr", OCR_REC, model, b.input, slice, outputs);
                calls.push((b.indices, id));
            }
        }
        Ok(TextRound2 { settled, calls })
    }

    /// The batches that fit the photo's window now (at least one when any is left), and the
    /// warps of their slices.
    async fn stage_lines(
        &mut self,
        batches: &mut Peekable<std::vec::IntoIter<LineBatch>>,
    ) -> Result<(Vec<LineBatch>, Vec<Warp<D::Input>>), Stop> {
        let (mut batch, mut warps) = (Vec::new(), Vec::new());
        while let Some(next) = batches.peek() {
            let bytes = next.input.bytes();
            if !batch.is_empty() && !self.calls.within_window(bytes) {
                break;
            }
            let input = self.calls.stage(self.clock, bytes, OCR_CROPS).await?;
            if let Some(b) = batches.next() {
                let kind = WarpKind::Text(b.warp.clone());
                warps.push(Warp {
                    input,
                    kind,
                    stage: OCR_CROPS,
                });
                batch.push(b);
            }
        }
        Ok((batch, warps))
    }

    /// One launch job of line warps, waited for: the warped slices in order.
    async fn launch_lines(
        &mut self,
        frame: &Arc<D::Frame>,
        warps: Vec<Warp<D::Input>>,
    ) -> Result<Vec<D::Input>, Stop> {
        let job = LaunchJob {
            frame: Arc::clone(frame),
            writes: Vec::new(),
            warps,
            blur: Vec::new(),
            blur_stage: OCR_CROPS,
            request: Arc::clone(&self.request),
        };
        let launched = self.launch_waited(OCR_CROPS, job).await?;
        Ok(launched.warped)
    }

    /// Finish (`ocr`): every recogniser answer, then the six text tensors; at the first rejected
    /// call the branch fails and answers none.
    pub async fn finish_ocr(
        &mut self,
        round: TextRound2,
        decoder: &CtcDecoder,
    ) -> Result<(), Stop> {
        let mut answers = Vec::with_capacity(round.calls.len());
        for (indices, id) in round.calls {
            match self.calls.result(self.clock, id).await? {
                Answer::Outputs(out) => answers.push((indices, out)),
                Answer::Rejected(why) => {
                    tracing::warn!("ocr: model call failed: {why}");
                    let state = BranchState::Error(MODEL_ERROR);
                    self.answers.stop(&mut self.calls, "ocr", state);
                    return Ok(());
                }
            }
        }
        let tensors = self
            .clock
            .time(OCR_DECODE, || {
                ocr::outputs(&round.settled, &answers, decoder)
            })
            .map_err(engine)?;
        self.answers.out.extend(tensors);
        Ok(())
    }
}
