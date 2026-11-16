//! The seam between the front and what makes the answers: a [`Pipeline`] takes one admitted
//! photo and returns its tensors. `stub` answers without CUDA (the contract test's subject); the
//! real pipeline (decode lanes, kernels, model calls through `gpu_frame`) is the other one.
//! Everything the front owns, and nothing else, is on this side of the trait: the statuses of
//! protocol version 1, the response tensors and the check order.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Mutex;
use std::time::Instant;

use bytes::Bytes;
use gpu_frame::CancelToken;
use tokio::sync::Notify;

use crate::admission::{Overload, RequestClock};
use crate::wire::EnginePlan;
use crate::wire::branch_status::BranchState;

/// One output tensor with its datatype; `shape` is the response's (`-1` never appears).
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    /// BOOL tensor.
    Bool {
        /// Dimensions as the response states them.
        shape: Vec<i64>,
        /// Elements in row-major order, as many as the shape's product.
        data: Vec<bool>,
    },
    /// INT32 tensor.
    Int32 {
        /// Dimensions as the response states them.
        shape: Vec<i64>,
        /// Elements in row-major order, as many as the shape's product.
        data: Vec<i32>,
    },
    /// INT64 tensor.
    Int64 {
        /// Dimensions as the response states them.
        shape: Vec<i64>,
        /// Elements in row-major order, as many as the shape's product.
        data: Vec<i64>,
    },
    /// FP32 tensor.
    Fp32 {
        /// Dimensions as the response states them.
        shape: Vec<i64>,
        /// Elements in row-major order, as many as the shape's product.
        data: Vec<f32>,
    },
    /// FP64 tensor.
    Fp64 {
        /// Dimensions as the response states them.
        shape: Vec<i64>,
        /// Elements in row-major order, as many as the shape's product.
        data: Vec<f64>,
    },
    /// BYTES, one element per entry (`crop_jpegs`, `ocr_texts`); the shape is `[len]`.
    Bytes(Vec<Bytes>),
}

impl Tensor {
    /// The KServe datatype name.
    pub fn datatype(&self) -> &'static str {
        match self {
            Self::Bool { .. } => "BOOL",
            Self::Int32 { .. } => "INT32",
            Self::Int64 { .. } => "INT64",
            Self::Fp32 { .. } => "FP32",
            Self::Fp64 { .. } => "FP64",
            Self::Bytes(_) => "BYTES",
        }
    }
}

/// The photo's own failure, mapped to a gRPC status by the front.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// Budget, region or colour-table wait ran out: `RESOURCE_EXHAUSTED`, retried by the API.
    Overload(Overload),
    /// The model server did not answer: `UNAVAILABLE`.
    ModelServerUnavailable(String),
    /// The CUDA context is lost: `UNAVAILABLE`, and the process exits for a restart.
    ContextLost(String),
    /// Anything else: `INTERNAL`, details starting with the engine's error prefix.
    Engine(String),
}

/// What one admitted photo is answered with.
#[derive(Debug)]
pub struct PhotoAnswer {
    /// `ok` or one of the contract's photo statuses.
    pub status: String,
    /// `gpu`, `cpu_<reason>` or `none`.
    pub decode_route: String,
    /// One entry per reported branch.
    pub branches: BTreeMap<String, BranchState>,
    /// The tensors of the plan, in the order they are answered.
    pub outputs: Vec<(String, Tensor)>,
}

/// The state of one request that every stage reads: its deadline, the stage it is in, and
/// whether its caller is gone (the stream ended or the deadline passed). Each hand-off checks
/// the cancel token (`gpu_frame`'s pools check it before a job starts) and releases what the
/// photo holds instead of starting work.
#[derive(Debug)]
pub struct RequestState {
    /// The request's deadline and arrival time.
    pub clock: RequestClock,
    stage: Mutex<&'static str>,
    cancelled: CancelToken,
}

impl RequestState {
    /// A state for a request with `clock`, in stage `queue`, not cancelled.
    pub fn new(clock: RequestClock) -> Self {
        Self {
            clock,
            stage: Mutex::new("queue"),
            cancelled: CancelToken::new(),
        }
    }

    /// The stage the request last entered (`op_frame_engine_request_deadlines` is labelled by it).
    pub fn enter(&self, stage: &'static str) {
        if let Ok(mut current) = self.stage.lock() {
            *current = stage;
        }
    }

    /// The stage the request last entered; `queue` before any.
    pub fn stage(&self) -> &'static str {
        self.stage.lock().map_or("queue", |s| *s)
    }

    /// Marks the caller gone, so queued pool jobs of the request end without running.
    pub fn cancel(&self) {
        self.cancelled.cancel();
    }

    /// Whether the caller is gone.
    pub fn cancelled(&self) -> bool {
        self.cancelled.is_cancelled()
    }

    /// The token every pool job of this request is submitted with.
    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancelled
    }

    /// When the request reached the front.
    pub fn arrived(&self) -> Instant {
        self.clock.arrived
    }
}

/// One admitted photo: the encoded file (a slice of the request message), its validated plan and
/// the stored boxes (`Some` when the request carried `boxes`, even with no rows).
#[derive(Debug)]
pub struct Photo {
    /// The encoded image file, a slice of the request message.
    pub image: Bytes,
    /// The plan the request asked for, already validated.
    pub plan: EnginePlan,
    /// Stored boxes as `x1, y1, x2, y2`; `Some` when the request carried `boxes`, even with no rows.
    pub boxes: Option<Vec<[f64; 4]>>,
    /// The request's deadline, stage and cancel state.
    pub request: std::sync::Arc<RequestState>,
}

/// What the front calls to answer a photo and to learn whether it can take work.
pub trait Pipeline: Send + Sync + 'static {
    /// Whether the pipeline can take work: the model server answers, the region is registered and
    /// the context is not lost. One atomic read; never waits (the API asks before every photo).
    fn ready(&self) -> bool;

    /// Whether the CUDA context is lost: the health tick then stops the process with a failure
    /// for the restart policy.
    fn lost(&self) -> bool;

    /// Notified when a model call failed, so the health tick probes at once instead of waiting
    /// out its period (readiness follows the probe, never one failed call).
    fn wake(&self) -> &Notify;

    /// One health probe (model server, region re-registration), run every
    /// `OP_FRAME_ENGINE_HEALTH_SECONDS` and at once after a failed call; it refreshes what
    /// [`Pipeline::ready`] reads.
    fn check(&self) -> impl Future<Output = ()> + Send;

    /// The answer for one admitted photo. The future is dropped when the request ends early.
    fn process(
        &self,
        photo: Photo,
    ) -> impl Future<Output = Result<PhotoAnswer, PipelineError>> + Send;

    /// After the drain, at a clean stop: give back what outlives the process's photos (the GPU
    /// pipeline unregisters its region from the model server, then frees it).
    fn close(self) -> impl Future<Output = ()> + Send
    where
        Self: Sized,
    {
        async {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_starts_in_the_queue_and_reports_the_stage_it_last_entered() {
        let state = RequestState::new(RequestClock::start(Instant::now(), 60.0));
        assert_eq!(state.stage(), "queue");
        state.enter("embed_pe");
        assert_eq!(state.stage(), "embed_pe");
        assert!(!state.cancelled());
        state.cancel();
        assert!(state.cancelled());
        assert_eq!(state.arrived(), state.clock.arrived);
    }

    #[test]
    fn each_tensor_names_its_kserve_datatype() {
        let tensors = [
            (
                Tensor::Bool {
                    shape: vec![1],
                    data: vec![true],
                },
                "BOOL",
            ),
            (
                Tensor::Int32 {
                    shape: vec![1],
                    data: vec![1],
                },
                "INT32",
            ),
            (
                Tensor::Int64 {
                    shape: vec![1],
                    data: vec![1],
                },
                "INT64",
            ),
            (
                Tensor::Fp32 {
                    shape: vec![1],
                    data: vec![1.0],
                },
                "FP32",
            ),
            (
                Tensor::Fp64 {
                    shape: vec![1],
                    data: vec![1.0],
                },
                "FP64",
            ),
            (Tensor::Bytes(vec![]), "BYTES"),
        ];
        for (tensor, datatype) in tensors {
            assert_eq!(tensor.datatype(), datatype);
        }
    }
}
