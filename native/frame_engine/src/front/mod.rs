//! The KServe v2 gRPC front of protocol version 1 (`frame_engine/front.py`), on tonic.
//!
//! It answers the five calls the API's Triton client makes: `ModelInfer` (unary, one image) and
//! `ServerLive`, `ServerReady`, `ModelReady` and `ModelMetadata` for the model `frame_engine`;
//! every other call is `UNIMPLEMENTED` (`build.rs` leaves the rest out of the router).
//!
//! The statuses and the order of the checks are the Python front's, which the API's retry policy
//! (`src/clients/frame_engine.py`) is written against: an unknown model is `NOT_FOUND`; a
//! malformed request or plan `INVALID_ARGUMENT` (the plan fault's name first); an engine that is
//! not ready or is draining `UNAVAILABLE`; more photos than `OP_FRAME_ENGINE_MAX_IN_FLIGHT`
//! `RESOURCE_EXHAUSTED`. A failure inside the photo is `RESOURCE_EXHAUSTED` (an overload),
//! `UNAVAILABLE` (the model server, or a lost CUDA context) or `INTERNAL` whose details start
//! with the engine's error prefix (an `INTERNAL` without it was made by the transport, and the API
//! sends that one again). A photo still running at the request deadline is `DEADLINE_EXCEEDED`,
//! logged and counted with the stage it was in.
//!
//! A message above `OP_FRAME_ENGINE_MAX_MESSAGE_MIB` is `OUT_OF_RANGE` in tonic
//! (`tonic::codec::decode`); a response layer turns that one status into `RESOURCE_EXHAUSTED`, as
//! gRPC C-core answers (the API retries it). Channel settings are plan 17 section 2.4: HTTP/2
//! windows of 1 MiB per stream and 2 MiB per connection for a 10 MB body, and no server
//! keepalive and no ping policy: h2 answers every ping (so the API's idle pings every 30 s never
//! draw a GOAWAY).

mod response;
mod serve;

use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use self::response::{build_response, failure_status};
pub use self::serve::{ServeConfig, serve};
use crate::admission::{InFlightCap, RequestClock};
use crate::metrics::Metrics;
use crate::pipeline::{Photo, Pipeline, RequestState};
use crate::proto::grpc_inference_service_server::GrpcInferenceService;
use crate::proto::model_metadata_response::TensorMetadata;
use crate::proto::{
    ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse,
    ModelReadyRequest, ModelReadyResponse, ServerLiveRequest, ServerLiveResponse,
    ServerReadyRequest, ServerReadyResponse,
};
use crate::pyfmt::repr_str;
use crate::wire::contract::contract;
use crate::wire::request::parse_request;

/// What the front reads from the settings.
#[derive(Debug, Clone, Copy)]
pub struct FrontConfig {
    /// Photos admitted at once, `OP_FRAME_ENGINE_MAX_IN_FLIGHT`; more are answered `RESOURCE_EXHAUSTED`.
    pub max_in_flight: u32,
    /// Seconds from arrival that a request may take, `OP_FRAME_ENGINE_REQUEST_DEADLINE_SECONDS`.
    pub request_deadline_seconds: f64,
}

struct State<P> {
    pipeline: Arc<P>,
    metrics: Arc<Metrics>,
    cap: Arc<InFlightCap>,
    draining: std::sync::atomic::AtomicBool,
    request_deadline_seconds: f64,
}

impl<P: Pipeline> State<P> {
    fn ready(&self) -> bool {
        !self.draining.load(std::sync::atomic::Ordering::Acquire) && self.pipeline.ready()
    }
}

/// The service; hand it to [`serve`].
pub struct Front<P> {
    state: Arc<State<P>>,
}

/// A handle on a running front: drain it, ask whether it is idle.
pub struct FrontHandle<P> {
    state: Arc<State<P>>,
}

impl<P> Clone for FrontHandle<P> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<P: Pipeline> Front<P> {
    /// The service over `pipeline`, and the handle used to drain it and to watch its in-flight count.
    pub fn new(
        pipeline: Arc<P>,
        metrics: Arc<Metrics>,
        config: FrontConfig,
    ) -> (Self, FrontHandle<P>) {
        let state = Arc::new(State {
            pipeline,
            metrics,
            cap: InFlightCap::new(config.max_in_flight),
            draining: std::sync::atomic::AtomicBool::new(false),
            request_deadline_seconds: config.request_deadline_seconds,
        });
        (
            Self {
                state: Arc::clone(&state),
            },
            FrontHandle { state },
        )
    }
}

impl<P: Pipeline> FrontHandle<P> {
    /// Readiness turns false and new photos are refused `UNAVAILABLE`; photos in flight go on.
    pub fn begin_drain(&self) {
        self.state
            .draining
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Completes when no photo is in flight; used during shutdown after `begin_drain`.
    pub async fn wait_idle(&self) {
        self.state.cap.wait_idle().await;
    }

    /// Photos admitted and not yet answered.
    pub fn in_flight(&self) -> u32 {
        self.state.cap.current()
    }
}

fn unknown_model(name: &str) -> Status {
    Status::not_found(format!("unknown model {}", repr_str(name)))
}

fn tensor_metadata(name: &str, datatype: &str, shape: &[i64]) -> TensorMetadata {
    TensorMetadata {
        name: name.to_owned(),
        datatype: datatype.to_owned(),
        shape: shape.to_vec(),
    }
}

/// Cancels the request when the handler is dropped before the photo was answered (the caller's
/// stream ended) or the deadline passed; every stage then releases what the photo holds.
struct CancelUnlessDone {
    request: Arc<RequestState>,
    done: bool,
}

impl Drop for CancelUnlessDone {
    fn drop(&mut self) {
        if !self.done {
            self.request.cancel();
        }
    }
}

#[tonic::async_trait]
impl<P: Pipeline> GrpcInferenceService for Front<P> {
    async fn server_live(
        &self,
        _request: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _request: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse {
            ready: self.state.ready(),
        }))
    }

    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        let request = request.into_inner();
        let c = contract();
        let ready = request.name == c.model.name
            && c.model_ready.versions_accepted.contains(&request.version)
            && self.state.ready();
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn model_metadata(
        &self,
        request: Request<ModelMetadataRequest>,
    ) -> Result<Response<ModelMetadataResponse>, Status> {
        let request = request.into_inner();
        let c = contract();
        if request.name != c.model.name {
            return Err(unknown_model(&request.name));
        }
        Ok(Response::new(ModelMetadataResponse {
            name: c.model.name.clone(),
            versions: vec![c.model.version.clone()],
            platform: c.model.platform.clone(),
            inputs: c
                .inputs
                .iter()
                .map(|t| tensor_metadata(&t.name, &t.datatype, &t.shape))
                .collect(),
            outputs: c
                .outputs
                .iter()
                .map(|t| tensor_metadata(&t.name, &t.datatype, &t.shape))
                .collect(),
        }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let arrived = Instant::now();
        let request = request.into_inner();
        let state = &self.state;
        if request.model_name != contract().model.name {
            return Err(unknown_model(&request.model_name));
        }
        let parsed =
            parse_request(&request).map_err(|e| Status::invalid_argument(e.to_string()))?;
        if !state.ready() {
            return Err(Status::unavailable("frame engine not ready"));
        }
        let permit = state.cap.try_admit().map_err(|in_flight| {
            state.metrics.count_overload("in_flight");
            Status::resource_exhausted(format!(
                "overload: {in_flight} requests in flight (cap {})",
                state.cap.cap()
            ))
        })?;
        state.metrics.in_flight.set(i64::from(state.cap.current()));
        let clock = RequestClock::start(arrived, state.request_deadline_seconds);
        let shared = Arc::new(RequestState::new(clock));
        let mut guard = CancelUnlessDone {
            request: Arc::clone(&shared),
            done: false,
        };
        let photo = Photo {
            image: parsed.image,
            plan: parsed.plan,
            boxes: parsed.boxes,
            request: Arc::clone(&shared),
        };
        let outcome = tokio::time::timeout_at(
            tokio::time::Instant::from_std(clock.deadline),
            state.pipeline.process(photo),
        )
        .await;
        // the slot is free before the status is sent, so a client that sees the answer and asks
        // again is not refused for the request it just finished
        drop(permit);
        state.metrics.in_flight.set(i64::from(state.cap.current()));
        match outcome {
            Err(_elapsed) => {
                let stage = shared.stage();
                state.metrics.count_deadline(stage);
                tracing::error!(
                    "request answered DEADLINE_EXCEEDED after {:.1} s in stage {stage}",
                    state.request_deadline_seconds
                );
                Err(Status::deadline_exceeded(format!(
                    "{}{}",
                    contract().model.engine_error_prefix,
                    clock.exceeded_message(stage)
                )))
            }
            Ok(Err(failure)) => Err(failure_status(failure, &state.metrics)),
            Ok(Ok(answer)) => {
                guard.done = true;
                build_response(&request, answer).map(Response::new)
            }
        }
    }
}

#[cfg(test)]
mod tests;
