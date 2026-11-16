//! The renditions' steps in one photo's flow (`frame_engine/branches/derivatives.py`): the job
//! goes to the launch pool after the round-2 model calls went out, so its resamples and encodes
//! overlap them; the finish collects the JPEGs in plan order. A failed encode fails the photo
//! under the fault rule (`gpu_memory` for memory), not a model call.

use std::sync::Arc;

use gpu_frame::JobHandle;

use super::clock::{DERIVATIVE_JPEG, DERIVATIVE_RENDER};
use super::device::Device;
use super::rounds::Work;
use super::{Stop, engine, job_failed};
use crate::branches::derivatives::{self, Rendered, RenditionJob};
use crate::fault::GpuFault;
use crate::wire::EnginePlan;

/// The rendition job in flight.
pub(super) type RenditionsInFlight = JobHandle<Result<Rendered, GpuFault>>;

impl<D: Device> Work<'_, D> {
    /// Start the plan's renditions of `frame` on the launch pool; `None` when it asks for none.
    pub fn start_renditions(
        &mut self,
        plan: &EnginePlan,
        frame: &Arc<D::Frame>,
    ) -> Result<Option<RenditionsInFlight>, Stop> {
        if plan.derivatives.is_empty() {
            return Ok(None);
        }
        let (w, h) = self.size;
        let renditions = derivatives::plan(w, h, &plan.derivatives).map_err(engine)?;
        let job = RenditionJob {
            frame: Arc::clone(frame),
            renditions,
            request: Arc::clone(&self.request),
        };
        self.device
            .renditions(job, &self.cancel)
            .map(Some)
            .map_err(|e| Stop::Fail(job_failed(DERIVATIVE_RENDER, &e)))
    }

    /// Finish (`derivatives`): the JPEGs, and the job's seconds per stage.
    pub async fn finish_renditions(&mut self, job: RenditionsInFlight) -> Result<(), Stop> {
        self.clock.enter(DERIVATIVE_JPEG);
        let rendered = job
            .wait()
            .await
            .map_err(|e| Stop::Fail(job_failed(DERIVATIVE_JPEG, &e)))??;
        self.clock.merge(rendered.stages);
        self.answers.out.push(derivatives::output(rendered.jpegs));
        Ok(())
    }
}
