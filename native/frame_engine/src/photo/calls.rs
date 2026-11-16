//! The model calls of one photo (`PhotoCalls` of `frame_engine/branches/base.py`): its slices of
//! the region, its calls in flight, its window.
//!
//! The window bounds the region bytes one photo holds while it has calls in flight
//! (`OP_FRAME_ENGINE_REGION_MIB / OP_FRAME_ENGINE_WORKERS`): staging a slice that would pass it
//! first collects the photo's oldest call (its answer is kept for [`PhotoCalls::result`]). A slice
//! is always given when no call is in flight, so the bound is `max(window, one call)` and a photo
//! never waits on itself, only on the region (a bounded wait, then an overload).
//!
//! Each call runs as its own task owning its slice; the slice goes back to the region when the
//! call ended, answered, failed or cancelled (dropping the task cancels the call's HTTP/2 stream:
//! `ModelCalls::infer`). A call that failed is counted by model, code and reason as the Python
//! transport counts it; one that may mean the server is gone asks for a health probe.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use gpu_frame::{CallError, OutputTensor, SliceError};
use tokio::task::JoinHandle;
use tokio::time::Instant;

use super::Stop;
use super::clock::StageClock;
use super::device::{Device, ModelCall};
use crate::admission::Overload;
use crate::branches::InputSpec;
use crate::metrics::{ByModelCall, Metrics};
use crate::pipeline::PipelineError;

/// A call's answer as the branches see it.
#[derive(Debug, Clone, PartialEq)]
pub enum Answer {
    Outputs(Vec<OutputTensor>),
    /// The server answered the call with an error: its branch fails (`model_error`), the photo
    /// goes on.
    Rejected(String),
}

/// A call's number in its photo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallId(usize);

/// A running call; dropping it aborts the task, which drops the call and its slice.
struct Running<I>(JoinHandle<(Result<Vec<OutputTensor>, CallError>, I)>);

impl<I> Drop for Running<I> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

enum Slot<I> {
    Open {
        running: Running<I>,
        bytes: usize,
        owner: &'static str,
        stage: &'static str,
        model: String,
    },
    Done(Answer),
    Cancelled,
}

pub struct PhotoCalls<D: Device> {
    device: Arc<D>,
    metrics: Arc<Metrics>,
    window: usize,
    slab_wait: Duration,
    deadline: Instant,
    /// Bytes of the slices this photo holds: staged, or read by a call not yet collected.
    held: usize,
    open: VecDeque<usize>,
    slots: Vec<Slot<D::Input>>,
}

/// The Python transport's code names (`StatusCode.UNAVAILABLE`) for tonic's (`Unavailable`).
fn status_code(tonic: &str) -> String {
    let mut out = String::from("StatusCode.");
    for (i, c) in tonic.chars().enumerate() {
        if c.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_uppercase());
    }
    out
}

impl<D: Device> PhotoCalls<D> {
    pub fn new(
        device: Arc<D>,
        metrics: Arc<Metrics>,
        window: usize,
        slab_wait: Duration,
        deadline: Instant,
    ) -> Self {
        Self {
            device,
            metrics,
            window,
            slab_wait,
            deadline,
            held: 0,
            open: VecDeque::new(),
            slots: Vec::new(),
        }
    }

    /// Whether `bytes` more may be staged now: always when no call is in flight, else within
    /// the window.
    fn fits(&self, bytes: usize) -> bool {
        self.open.is_empty() || self.within_window(bytes)
    }

    /// Whether `bytes` more stay within the window beside what the photo holds. A slice staged
    /// but not yet sent counts as a call in flight would: chunks are batched into one launch only
    /// while this holds, so each chunk is staged where the Python engine, which sends a chunk
    /// before it stages the next, would stage it.
    pub fn within_window(&self, bytes: usize) -> bool {
        self.held + bytes <= self.window
    }

    /// A slice for one call's input; the waits for room count in `stage`.
    pub async fn stage(
        &mut self,
        clock: &mut StageClock,
        bytes: usize,
        stage: &'static str,
    ) -> Result<D::Input, Stop> {
        clock.enter(stage);
        let start = std::time::Instant::now();
        while !self.fits(bytes) {
            let Some(&oldest) = self.open.front() else {
                break;
            };
            self.settle(oldest).await?;
        }
        let taken = self.device.slice(bytes, self.slab_wait).await;
        clock.add(stage, start.elapsed().as_secs_f64());
        let slice = taken.map_err(|e| match e {
            SliceError::TooLarge { .. } => Stop::Status("over_budget"),
            overload @ SliceError::Overload { .. } => {
                Stop::Fail(PipelineError::Overload(Overload {
                    cause: "slab".into(),
                    detail: overload.to_string(),
                }))
            }
        })?;
        self.held += bytes;
        Ok(slice)
    }

    /// Send one call reading `slice` (written, its stream waited for) without waiting for it.
    pub fn send(
        &mut self,
        owner: &'static str,
        stage: &'static str,
        model: &str,
        input: InputSpec,
        slice: D::Input,
        outputs: Vec<&'static str>,
    ) -> CallId {
        let bytes = input.bytes();
        let call = ModelCall {
            model: model.to_owned(),
            input,
            slice,
            outputs,
            deadline: self.deadline,
        };
        let running = Running(tokio::spawn(Arc::clone(&self.device).infer(call)));
        self.slots.push(Slot::Open {
            running,
            bytes,
            owner,
            stage,
            model: model.to_owned(),
        });
        self.open.push_back(self.slots.len() - 1);
        CallId(self.slots.len() - 1)
    }

    /// The call's answer, waited for in its stage. A server that is gone, a deadline or a region
    /// it cannot find fail the photo (`UNAVAILABLE`, retried by the API).
    pub async fn result(&mut self, clock: &mut StageClock, id: CallId) -> Result<Answer, Stop> {
        if let Some(Slot::Open { stage, .. }) = self.slots.get(id.0) {
            let stage = *stage;
            let start = std::time::Instant::now();
            clock.enter(stage);
            let settled = self.settle(id.0).await;
            clock.add(stage, start.elapsed().as_secs_f64());
            settled?;
        }
        match self.slots.get(id.0) {
            Some(Slot::Done(answer)) => Ok(answer.clone()),
            _ => Err(Stop::Fail(PipelineError::Engine(
                "the result of a call never sent or cancelled".into(),
            ))),
        }
    }

    async fn settle(&mut self, k: usize) -> Result<(), Stop> {
        self.open.retain(|&o| o != k);
        let slot = std::mem::replace(&mut self.slots[k], Slot::Cancelled);
        let Slot::Open {
            mut running,
            bytes,
            model,
            ..
        } = slot
        else {
            self.slots[k] = slot;
            return Ok(());
        };
        let joined = (&mut running.0).await;
        self.held = self.held.saturating_sub(bytes);
        let outcome = match joined {
            Ok((outcome, slice)) => {
                // the server is done with the slice: answered or failed
                drop(slice);
                outcome
            }
            Err(e) => {
                return Err(Stop::Fail(PipelineError::Engine(format!(
                    "{model}: the call's task ended: {e}"
                ))));
            }
        };
        match outcome {
            Ok(outputs) => {
                self.slots[k] = Slot::Done(Answer::Outputs(outputs));
                Ok(())
            }
            Err(e) => {
                let (code, reason) = match &e {
                    CallError::Model { code, .. } => (status_code(code), "rejected"),
                    CallError::Deadline => ("StatusCode.DEADLINE_EXCEEDED".into(), "deadline"),
                    CallError::Unavailable(_) => ("StatusCode.UNAVAILABLE".into(), "unavailable"),
                    CallError::RegionUnknown(_) => ("StatusCode.NOT_FOUND".into(), "unavailable"),
                };
                self.metrics
                    .triton_errors
                    .get_or_create(&ByModelCall {
                        model: model.clone(),
                        code,
                        reason: reason.into(),
                    })
                    .inc();
                self.device.call_failed(&e);
                if let CallError::Model { .. } = e {
                    self.slots[k] = Slot::Done(Answer::Rejected(format!("{model}: {e}")));
                    Ok(())
                } else {
                    Err(Stop::Fail(PipelineError::ModelServerUnavailable(format!(
                        "{model}: {e}"
                    ))))
                }
            }
        }
    }

    /// Give up `owner`'s calls in flight (its branch failed) and their slices.
    pub fn cancel(&mut self, owner: &'static str) {
        let mine: Vec<usize> = self
            .open
            .iter()
            .copied()
            .filter(|&k| matches!(&self.slots[k], Slot::Open { owner: o, .. } if *o == owner))
            .collect();
        for k in mine {
            self.open.retain(|&o| o != k);
            if let Slot::Open { bytes, .. } = std::mem::replace(&mut self.slots[k], Slot::Cancelled)
            {
                self.held = self.held.saturating_sub(bytes);
            }
        }
    }

    /// Bytes this photo holds of the region.
    pub fn held(&self) -> usize {
        self.held
    }

    /// Calls in flight.
    pub fn in_flight(&self) -> usize {
        self.open.len()
    }
}

#[cfg(test)]
mod tests {
    use super::status_code;

    #[test]
    fn codes_are_named_as_the_python_transport_labels_them() {
        assert_eq!(status_code("Internal"), "StatusCode.INTERNAL");
        assert_eq!(
            status_code("InvalidArgument"),
            "StatusCode.INVALID_ARGUMENT"
        );
        assert_eq!(
            status_code("ResourceExhausted"),
            "StatusCode.RESOURCE_EXHAUSTED"
        );
    }
}
