//! The seam between one photo's flow and the card: what the flow hands to the pools (a decode
//! with its round-1 writes, a launch of round-2 writes and blur sums, a crop JPEG), the region
//! slices model inputs live in, and the model call. The GPU pipeline (`engine.rs`) fills it with
//! `gpu_frame`; the photo tests fill it with host threads and a host slab, so the flow (stages,
//! bounds, cancellation, failures) runs without a card.
//!
//! Ownership is the plan's (section 2.3): a slice goes into the job that writes it and comes back
//! after that job's stream was waited for; the frame is held by the photo and borrowed by jobs
//! through an `Arc`, each job waiting for its own stream before it lets go; the frame budget's
//! permit travels with the frame and is released with it.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use gpu_frame::{CallError, CancelToken, JobError, JobHandle, OutputTensor, SliceError};
use tokio::time::Instant;

use crate::branches::InputSpec;
use crate::branches::crops::CropJpeg;
use crate::fault::GpuFault;
use crate::kernels::Resample;
use crate::pipeline::RequestState;

/// Seconds per stage measured inside a job, added to the photo's clock.
pub type Stages = Vec<(&'static str, f64)>;

/// What a job writes into one region slice: all of it filled first (the canvas, the squares'
/// margins), then each resample, timed as `stage`.
#[derive(Debug)]
pub struct Write<I> {
    pub input: I,
    pub fill: Option<u8>,
    pub resamples: Vec<Resample>,
    pub stage: &'static str,
}

/// A decode on a lane: nvJPEG, the orientation, the size check, then the round-1 writes and one
/// wait for the lane's stream.
pub struct DecodeJob<I> {
    /// The image proper (up to its end-of-image marker), a slice of the request message.
    pub image: Bytes,
    pub orientation: u32,
    /// The oriented size the header promised; a decode of another size is refused.
    pub size: (u32, u32),
    pub writes: Vec<Write<I>>,
    /// Held as long as the frame (the frame budget's permit).
    pub hold: Box<dyn Send + Sync>,
    pub request: Arc<RequestState>,
}

/// What a decode gave.
pub enum Decoded<F, I> {
    /// The oriented frame and the written slices, in the order of the writes.
    Frame {
        frame: Arc<F>,
        inputs: Vec<I>,
        stages: Stages,
    },
    /// No frame: `decoder` or `gpu_memory`; the slices were dropped after the stream wait.
    Refused {
        status: &'static str,
        stages: Stages,
    },
}

/// Round-2 kernels on the launch pool: the writes, then the blur sums of `blur`, one wait.
pub struct LaunchJob<F, I> {
    pub frame: Arc<F>,
    pub writes: Vec<Write<I>>,
    pub blur: Vec<[i64; 4]>,
    pub request: Arc<RequestState>,
}

/// What a launch gave: the written slices in order and one sum pair per blur rectangle.
pub struct Launched<I> {
    pub inputs: Vec<I>,
    pub blur: Vec<[i64; 2]>,
    pub stages: Stages,
}

/// One model call reading one slice; the slice comes back with the answer.
pub struct ModelCall<I> {
    pub model: String,
    pub input: InputSpec,
    pub slice: I,
    pub outputs: Vec<&'static str>,
    /// The request's deadline; the call's own bound is the smaller of it and the per-call timeout.
    pub deadline: Instant,
}

pub type JobResult<T> = Result<JobHandle<Result<T, GpuFault>>, JobError>;

/// The card as one photo's flow uses it.
pub trait Device: Send + Sync + 'static {
    type Frame: Send + Sync + 'static;
    type Input: Send + Sync + 'static;

    /// Whether the context is lost (every photo is then refused).
    fn lost(&self) -> bool;

    /// A slice of the registered region, waiting at most `wait` for room.
    fn slice(
        &self,
        bytes: usize,
        wait: Duration,
    ) -> impl Future<Output = Result<Self::Input, SliceError>> + Send;

    fn decode(
        &self,
        job: DecodeJob<Self::Input>,
        cancel: &CancelToken,
    ) -> JobResult<Decoded<Self::Frame, Self::Input>>;

    fn launch(
        &self,
        job: LaunchJob<Self::Frame, Self::Input>,
        cancel: &CancelToken,
    ) -> JobResult<Launched<Self::Input>>;

    /// The JPEG of `rect` (`x1, y1, x2, y2`) of the frame, on the finish pool.
    fn crop_jpeg(
        &self,
        frame: Arc<Self::Frame>,
        rect: [i64; 4],
        cancel: &CancelToken,
    ) -> JobResult<CropJpeg>;

    /// Send one call; dropping the future cancels it and gives the slice back.
    fn infer(
        self: Arc<Self>,
        call: ModelCall<Self::Input>,
    ) -> impl Future<Output = (Result<Vec<OutputTensor>, CallError>, Self::Input)> + Send + 'static;

    /// A call failed: a failure that may mean the server is gone asks for a health probe now.
    fn call_failed(&self, error: &CallError);
}
