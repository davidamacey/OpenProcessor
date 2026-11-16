//! Branches that run only when the detector found something (`if_detected`, `_Rounds` of
//! `frame_engine/engine.py`, plan 17 section 2.2).
//!
//! A branch the plan names in `if_detected`, and every branch that needs it, skips round 1. After
//! the detector's settle its condition is checked against the detector's classes: when one of the
//! named classes was found, the branch's round-1 inputs are written from the frame, still held by
//! the photo, by one job on the launch pool, and its calls go out before the round-2 calls of the
//! others; its settle and round 2 follow theirs. A condition that does not hold ends the branch
//! and what needs it `not_run`; a detector that failed or did not run ends them
//! `dependency_failed`. A late round's slices are staged within the photo's window like any
//! other, so the photo's memory stays bounded by the same rules; a late call the model server
//! rejects fails its branch at its settle (`model_error`), and a fault of the late job fails the
//! photo, as round 2's would.
//!
//! Only the conditional branches of the protocol (`faces`, `ocr`; `CONDITIONAL` of
//! `src/domain/processing_plan.py`) can be late; this module schedules any built branch that
//! the plan makes late, and a branch module adds its round-1 calls as [`LateCall`]s.

use std::sync::Arc;
use std::time::Instant;

use super::calls::CallId;
use super::clock::LETTERBOX;
use super::device::{Device, LaunchJob, Write};
use super::rounds::{Round1Call, Work};
use super::{Stop, job_failed, settle_job_time};
use crate::kernels::Resample;
use crate::wire::EnginePlan;
use crate::wire::branch_status::{BranchState, DEPENDENCY_FAILED};

/// The `needs` of the branches: a branch and the branch whose round-1 answer it reads
/// (`Branch.needs`; `embed_crops` reads the detector's rectangles).
pub const NEEDS: [(&str, &str); 2] = [("embed_crops", "detect"), ("face_embed", "faces")];

/// The branches that need `name`, directly or through another, in [`NEEDS`] order.
pub fn dependents(name: &str) -> Vec<&'static str> {
    let mut found: Vec<&'static str> = Vec::new();
    let mut frontier = vec![name.to_owned()];
    while let Some(n) = frontier.pop() {
        for (branch, needs) in NEEDS {
            if needs == n && !found.contains(&branch) {
                found.push(branch);
                frontier.push(branch.to_owned());
            }
        }
    }
    found
}

/// The built branches the plan makes late: each one `if_detected` names, then what needs it,
/// in `built` order.
pub fn late(plan: &EnginePlan, built: &[&'static str]) -> Vec<&'static str> {
    let conditional: Vec<&str> = built
        .iter()
        .copied()
        .filter(|b| plan.runs(b) && plan.if_detected.contains_key(*b))
        .collect();
    built
        .iter()
        .copied()
        .filter(|b| {
            plan.runs(b)
                && (conditional.contains(b)
                    || conditional.iter().any(|c| dependents(c).contains(b)))
        })
        .collect()
}

/// What the check of a late branch's own condition says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// One of the named classes was found: the branch runs.
    Run,
    /// None was: `not_run`, and what needs it too.
    NotRun,
    /// The detector failed or did not run: `dependency_failed`, and what needs it too.
    DependencyFailed,
}

/// The verdict of `name`'s condition against the detector's classes (`None`: no answer), or
/// `None` for a late branch with no condition of its own (it follows the branch it needs).
pub fn verdict(plan: &EnginePlan, name: &str, classes: Option<&[i32]>) -> Option<Verdict> {
    let ids = plan.if_detected.get(name)?;
    Some(match classes {
        None => Verdict::DependencyFailed,
        Some(found)
            if found
                .iter()
                .any(|c| ids.iter().any(|id| i64::from(*id) == i64::from(*c))) =>
        {
            Verdict::Run
        }
        Some(_) => Verdict::NotRun,
    })
}

/// One late round-1 call: how its slice is written from the frame, and the call that reads it.
pub struct LateCall<'p> {
    /// Owner, stage, model, input and output name of the call.
    pub call: Round1Call<'p>,
    /// The value the whole slice is filled with first, if any.
    pub fill: Option<u8>,
    /// The resamples that write the slice.
    pub resamples: Vec<Resample>,
    /// The stage the write is timed as.
    pub stage: &'static str,
}

impl<D: Device> Work<'_, D> {
    /// Check each late branch's condition; end those that do not run, with what needs them.
    /// Answers the late branches that still run, in order.
    pub fn check_conditions(
        &mut self,
        plan: &EnginePlan,
        late: &[&'static str],
        classes: Option<&[i32]>,
    ) -> Vec<&'static str> {
        for name in late {
            if !self.answers.running(name) {
                continue;
            }
            let state = match verdict(plan, name, classes) {
                None | Some(Verdict::Run) => continue,
                Some(Verdict::NotRun) => BranchState::NotRun,
                Some(Verdict::DependencyFailed) => BranchState::Error(DEPENDENCY_FAILED),
            };
            // what needs it first, so they end with its state, not dependency_failed
            for branch in dependents(name).into_iter().chain(std::iter::once(*name)) {
                self.answers.stop(&mut self.calls, branch, state.clone());
            }
        }
        late.iter()
            .copied()
            .filter(|b| self.answers.running(b))
            .collect()
    }

    /// The late round 1: the slices staged within the photo's window, written from the frame by
    /// launch jobs (one per group that fits the window together), each group's calls sent when
    /// its job ended. Answers each call's owner and id.
    ///
    /// # Errors
    ///
    /// A slice that cannot be staged, or a launch job's failure (which fails the photo).
    pub async fn late_round1(
        &mut self,
        frame: &Arc<D::Frame>,
        late: Vec<LateCall<'_>>,
    ) -> Result<Vec<(&'static str, CallId)>, Stop> {
        let mut sent = Vec::with_capacity(late.len());
        let mut late = late.into_iter().peekable();
        while late.peek().is_some() {
            let mut writes = Vec::new();
            let mut calls = Vec::new();
            while let Some(next) = late.peek() {
                let bytes = next.call.3.bytes();
                if !calls.is_empty() && !self.calls.within_window(bytes) {
                    break;
                }
                let slice = self.calls.stage(self.clock, bytes, next.stage).await?;
                if let Some(l) = late.next() {
                    writes.push(Write {
                        input: slice,
                        fill: l.fill,
                        resamples: l.resamples,
                        stage: l.stage,
                    });
                    calls.push(l.call);
                }
            }
            let inputs = self.launch_writes(frame, writes).await?;
            sent.extend(self.send_round1(calls, inputs));
        }
        Ok(sent)
    }

    /// One launch job writing `writes` from the frame; the written slices, in order.
    async fn launch_writes(
        &mut self,
        frame: &Arc<D::Frame>,
        writes: Vec<Write<D::Input>>,
    ) -> Result<Vec<D::Input>, Stop> {
        let stage = writes.first().map_or(LETTERBOX, |w| w.stage);
        let job = LaunchJob {
            frame: Arc::clone(frame),
            writes,
            blur: Vec::new(),
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
        Ok(launched.inputs)
    }
}
