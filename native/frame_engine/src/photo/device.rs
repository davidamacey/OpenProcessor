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
    /// The region slice this write fills.
    pub input: I,
    /// Byte value that first fills the whole slice, or `None` to leave it as it is.
    pub fill: Option<u8>,
    /// The resamples written into the slice, in order.
    pub resamples: Vec<Resample>,
    /// Stage the write's seconds are counted under.
    pub stage: &'static str,
}

/// A decode on a lane: nvJPEG, the orientation, the size check, then the round-1 writes and one
/// wait for the lane's stream.
pub struct DecodeJob<I> {
    /// The image proper (up to its end-of-image marker), a slice of the request message.
    pub image: Bytes,
    /// EXIF orientation 1 to 8 to apply to the decoded frame.
    pub orientation: u32,
    /// The oriented size the header promised; a decode of another size is refused.
    pub size: (u32, u32),
    /// The round-1 writes done after the decode, one per input slice.
    pub writes: Vec<Write<I>>,
    /// Held as long as the frame (the frame budget's permit).
    pub hold: Box<dyn Send + Sync>,
    /// The request the job belongs to, for its stage and cancel state.
    pub request: Arc<RequestState>,
}

/// What a decode gave.
pub enum Decoded<F, I> {
    /// The oriented frame and the written slices, in the order of the writes.
    Frame {
        /// The oriented frame on the card, shared by the later rounds.
        frame: Arc<F>,
        /// The slices that were written, in the order of the writes.
        inputs: Vec<I>,
        /// Seconds per stage measured inside the job.
        stages: Stages,
    },
    /// No frame: `decoder` or `gpu_memory`; the slices were dropped after the stream wait.
    Refused {
        /// The photo status that refuses the image, `decoder` or `gpu_memory`.
        status: &'static str,
        /// Seconds per stage measured before the refusal.
        stages: Stages,
    },
}

/// Round-2 kernels on the launch pool: the writes, then the blur sums of `blur`, one wait.
pub struct LaunchJob<F, I> {
    /// The frame the kernels read.
    pub frame: Arc<F>,
    /// The round-2 writes, one per input slice.
    pub writes: Vec<Write<I>>,
    /// Rectangles `x1, y1, x2, y2` in frame pixels whose Laplacian sums are taken; may be empty.
    pub blur: Vec<[i64; 4]>,
    /// The request the job belongs to, for its stage and cancel state.
    pub request: Arc<RequestState>,
}

/// What a launch gave: the written slices in order and one sum pair per blur rectangle.
pub struct Launched<I> {
    /// The slices that were written, in the order of the writes.
    pub inputs: Vec<I>,
    /// `(sum, sum of squares)` of the Laplacian for each blur rectangle, in the order sent.
    pub blur: Vec<[i64; 2]>,
    /// Seconds per stage measured inside the job.
    pub stages: Stages,
}

/// One model call reading one slice; the slice comes back with the answer.
pub struct ModelCall<I> {
    /// Name of the model on the server.
    pub model: String,
    /// The input tensor's name and shape.
    pub input: InputSpec,
    /// The region slice holding the input; the call owns it until the answer returns it.
    pub slice: I,
    /// Names of the output tensors to read back.
    pub outputs: Vec<&'static str>,
    /// The request's deadline; the call's own bound is the smaller of it and the per-call timeout.
    pub deadline: Instant,
}

/// What a pool job returns: a handle to wait on, or why it was not accepted; inside, the job's result or the GPU fault.
pub type JobResult<T> = Result<JobHandle<Result<T, GpuFault>>, JobError>;

/// The card as one photo's flow uses it.
pub trait Device: Send + Sync + 'static {
    /// The decoded frame on the card, shared by the rounds of one photo.
    type Frame: Send + Sync + 'static;
    /// A region slice that one model call reads.
    type Input: Send + Sync + 'static;

    /// Whether the context is lost (every photo is then refused).
    fn lost(&self) -> bool;

    /// A slice of the registered region, waiting at most `wait` for room.
    fn slice(
        &self,
        bytes: usize,
        wait: Duration,
    ) -> impl Future<Output = Result<Self::Input, SliceError>> + Send;

    /// Decodes on a lane and writes the round-1 inputs.
    ///
    /// # Errors
    ///
    /// The pool is shut down, or the job is cancelled, panics or, inside the handle, the GPU work fails.
    fn decode(
        &self,
        job: DecodeJob<Self::Input>,
        cancel: &CancelToken,
    ) -> JobResult<Decoded<Self::Frame, Self::Input>>;

    /// Runs the round-2 kernels on the launch pool.
    ///
    /// # Errors
    ///
    /// The pool is shut down, or the job is cancelled, panics or, inside the handle, the GPU work fails.
    fn launch(
        &self,
        job: LaunchJob<Self::Frame, Self::Input>,
        cancel: &CancelToken,
    ) -> JobResult<Launched<Self::Input>>;

    /// The JPEG of `rect` (`x1, y1, x2, y2`) of the frame, on the finish pool.
    ///
    /// # Errors
    ///
    /// The pool is shut down, or the job is cancelled, panics or, inside the handle, the copy or encode fails.
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
