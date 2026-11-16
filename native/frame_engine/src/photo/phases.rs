//! The phases of one photo after its round-1 calls went out, in the order `_Rounds.run` of
//! `frame_engine/engine.py` gives them: the settle of the early branches, the conditions and the
//! late round 1, round 2 of the early branches (the renditions start after it), the settle and
//! round 2 of the late ones, then the finish of every branch.
//!
//! The detector's branches are always early; `faces` (with `face_embed`) and `ocr` are early or
//! late as the plan's `if_detected` makes them (`late.rs`). A branch is one or the other, so each
//! of their phases runs once, in its place.

use std::sync::Arc;

use super::calls::CallId;
use super::cut::Cut;
use super::device::Device;
use super::faces::FaceRound2;
use super::renditions::RenditionsInFlight;
use super::rounds::{Round1, Round2, Work};
use super::text::TextRound2;
use super::{Runs, Stop};
use crate::branches::faces::DetCall;
use crate::branches::faces::scrfd::Faces;
use crate::branches::ocr::Settled;
use crate::branches::ocr::ctc::CtcDecoder;
use crate::pipeline::Photo;
use crate::wire::EnginePlan;

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

/// What the settle of the conditional branches found: the faces and the text boxes.
#[derive(Default)]
struct Settles {
    faces: Option<Faces>,
    text: Option<Settled>,
}

/// What round 2 of the conditional branches left in flight.
#[derive(Default)]
struct Rounds2 {
    faces: Option<FaceRound2>,
    text: Option<TextRound2>,
}

/// What the phases after round 1 left in flight, collected by [`Later::finish`].
pub(super) struct Later {
    /// Whether there were rectangles to cut (detected or stored).
    cut: bool,
    round2: Round2,
    faces: Option<FaceRound2>,
    text: Option<TextRound2>,
    renditions: Option<RenditionsInFlight>,
    embed_frame: Option<CallId>,
}

impl<D: Device> Work<'_, D> {
    /// Every phase after round 1 went out, up to the finish. The photo's own hold on `frame`
    /// ends here; jobs still running keep it to their wait.
    ///
    /// # Errors
    ///
    /// A stop of any of those phases.
    pub async fn after_round1<'p>(
        &mut self,
        photo: &'p Photo,
        runs: &Runs,
        mut round1: Round1<'p, D::Input>,
        mut sent: Vec<(&'static str, CallId)>,
        late: &[&'static str],
        frame: Arc<D::Frame>,
    ) -> Result<Later, Stop> {
        let plan = &photo.plan;
        let early = |b: &str| !late.contains(&b);
        let cut = self.settle_cut(photo, runs, &round1, &sent).await?;
        let settled = self
            .settle_conditional(plan, &mut round1.faces, &sent, early)
            .await?;
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
        let first = self.round2_conditional(plan, runs, settled, &frame).await?;
        let renditions = self.start_renditions(plan, &frame)?;
        let settled = self
            .settle_conditional(plan, &mut round1.faces, &sent, |b| !early(b))
            .await?;
        let then = self.round2_conditional(plan, runs, settled, &frame).await?;
        drop(frame);
        Ok(Later {
            cut: cut.is_some(),
            round2,
            faces: first.faces.or(then.faces),
            text: first.text.or(then.text),
            renditions,
            embed_frame: call_of(&sent, "embed_frame"),
        })
    }

    /// The rectangles round 2 cuts: the request's stored boxes, or the detector's settle. With
    /// none, the crop vectors stop (`dependency_failed`).
    async fn settle_cut(
        &mut self,
        photo: &Photo,
        runs: &Runs,
        round1: &Round1<'_, D::Input>,
        sent: &[(&'static str, CallId)],
    ) -> Result<Option<Cut>, Stop> {
        let cut = match &photo.boxes {
            Some(boxes) => Some(Cut::stored(boxes, self.size)?),
            None => self
                .settle_detect(call_of(sent, "detect"), round1.canvas.as_ref())
                .await?
                .map(|d| Cut::detected(&d, self.size)),
        };
        if cut.is_none() && runs.embed_crops {
            self.stop_crops_without_boxes();
        }
        Ok(cut)
    }

    /// The settle of the conditional branches `pick` selects, in branch order: the faces, then
    /// the text boxes. A branch that did not run or stopped settles to nothing.
    async fn settle_conditional(
        &mut self,
        plan: &EnginePlan,
        detector: &mut Vec<DetCall>,
        sent: &[(&'static str, CallId)],
        pick: impl Fn(&str) -> bool,
    ) -> Result<Settles, Stop> {
        let mut out = Settles::default();
        if pick("faces") {
            let calls = std::mem::take(detector);
            out.faces = self
                .settle_faces(plan, calls, calls_of(sent, "faces"))
                .await?;
        }
        if pick("ocr") {
            out.text = self.settle_ocr(plan, call_of(sent, "ocr")).await?;
        }
        Ok(out)
    }

    /// Round 2 of what `settled` holds, in branch order: the face warps, then the text lines.
    async fn round2_conditional(
        &mut self,
        plan: &EnginePlan,
        runs: &Runs,
        settled: Settles,
        frame: &Arc<D::Frame>,
    ) -> Result<Rounds2, Stop> {
        let mut out = Rounds2::default();
        if let Some(f) = &settled.faces {
            out.faces = Some(self.round2_faces(plan, runs, f, frame).await?);
        }
        if let Some(t) = settled.text {
            out.text = Some(self.round2_ocr(plan, t, frame).await?);
        }
        Ok(out)
    }
}

impl Later {
    /// Finish in the Python branch order: faces, text, renditions, then the detector's branches.
    ///
    /// # Errors
    ///
    /// A stop of any branch's finish.
    pub async fn finish<D: Device>(
        self,
        work: &mut Work<'_, D>,
        runs: &Runs,
        dictionary: &CtcDecoder,
    ) -> Result<(), Stop> {
        if let Some(r) = self.faces {
            work.finish_faces(runs, r).await?;
        }
        if let Some(t) = self.text {
            work.finish_ocr(t, dictionary).await?;
        }
        if let Some(job) = self.renditions {
            work.finish_renditions(job).await?;
        }
        work.finish(runs, self.cut, self.round2, self.embed_frame)
            .await
    }
}
