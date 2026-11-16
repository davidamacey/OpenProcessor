//! The engine as a client: a pool of channels, a cap on photos in flight, a deadline on every
//! call, the retry policy of [`crate::retry`], the readiness verdict.
//!
//! One unary KServe v2 `ModelInfer` per photo to the model `frame_engine`: inputs `image` and
//! `plan` (BYTES `[1]`), the optional `boxes` (FP64 `[n, 4]`), all in `raw_input_contents`, and no
//! `outputs` filter, so the answer is the whole answer (`src/domain/frame_engine_wire.py`,
//! `contracts/json/frame_engine_v1.json`).
//!
//! Back pressure: a photo takes a permit of a bounded semaphore for the length of one call and
//! waits for one, up to its deadline, when the cap is reached
//! (<https://docs.rs/tokio/1.54.1/tokio/sync/struct.Semaphore.html>); nothing queues without a bound.
//! Load balancing: the channels are separate connections used round robin, as the Python pool
//! uses its channels; `tonic`'s `Channel::balance_list` keys its endpoints by URI, so several
//! connections to one address are not what it is for.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use bytes::{BufMut, Bytes, BytesMut};
use tokio::sync::Semaphore;
use tokio::time::{Instant, sleep, timeout};
use tonic::{Request, Status};

use crate::answer::Answer;
use crate::channel::{Client, engine_channel};
use crate::codec::pack_bytes;
use crate::error::{ConfigError, EngineError};
use crate::model_server::ModelServer;
use crate::observer::{CallObserver, ProbeOutcome};
use crate::plan::{DetectorSpec, EnginePlan};
use crate::proto::model_infer_request::InferInputTensor;
use crate::proto::{ModelInferRequest, ModelInferResponse, ModelReadyRequest};
use crate::retry::{self, Class, Verdict, classify};
use crate::settings::EngineSettings;

/// The model name and protocol of the engine.
pub const MODEL: &str = "frame_engine";
/// Readiness probes made before the engine is judged unreachable.
pub const PROBE_ATTEMPTS: u32 = 3;
/// How long one probe may take.
pub const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// How long a positive readiness verdict is trusted (`READY_TTL_S`).
pub const READY_TTL: Duration = Duration::from_secs(5);

/// What a readiness probe learned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Readiness {
    /// How the probe ended.
    pub outcome: ProbeOutcome,
    /// How many calls it took.
    pub attempts: u32,
    /// The last transport failure's status name; empty when every attempt was answered.
    pub code: String,
    /// The last transport failure in words; empty when every attempt was answered.
    pub error: String,
}

impl Readiness {
    /// Whether the engine said it can take photos.
    #[must_use]
    pub fn ready(&self) -> bool {
        matches!(self.outcome, ProbeOutcome::Ready | ProbeOutcome::Recovered)
    }
}

/// The client of one engine.
pub struct EngineClient {
    channels: Vec<Client>,
    next: AtomicUsize,
    permits: Semaphore,
    call_timeout: Duration,
    deadline: Duration,
    ready: Mutex<HashMap<DetectorSpec, Instant>>,
    observer: Option<Arc<dyn CallObserver>>,
}

fn bytes_input(name: &str) -> InferInputTensor {
    InferInputTensor {
        name: name.to_owned(),
        datatype: "BYTES".to_owned(),
        shape: vec![1],
        parameters: HashMap::new(),
        contents: None,
    }
}

fn boxes_input(boxes: &[[f64; 4]]) -> (InferInputTensor, Bytes) {
    let mut raw = BytesMut::with_capacity(boxes.len() * 32);
    for value in boxes.iter().flatten() {
        raw.put_f64_le(*value);
    }
    let tensor = InferInputTensor {
        name: "boxes".to_owned(),
        datatype: "FP64".to_owned(),
        shape: vec![i64::try_from(boxes.len()).unwrap_or(i64::MAX), 4],
        parameters: HashMap::new(),
        contents: None,
    };
    (tensor, raw.freeze())
}

/// The request for one photo: `image` and `plan`, and `boxes` when the caller has the detections
/// (zero rows are allowed).
fn infer_request(image: &[u8], plan: &str, boxes: Option<&[[f64; 4]]>) -> ModelInferRequest {
    let mut inputs = vec![bytes_input("image"), bytes_input("plan")];
    let mut raw = vec![pack_bytes(image), pack_bytes(plan.as_bytes())];
    if let Some(boxes) = boxes {
        let (tensor, bytes) = boxes_input(boxes);
        inputs.push(tensor);
        raw.push(bytes);
    }
    ModelInferRequest {
        model_name: MODEL.to_owned(),
        inputs,
        raw_input_contents: raw,
        ..Default::default()
    }
}

impl EngineClient {
    /// A client whose channels dial on first use (and again after a lost connection).
    ///
    /// # Errors
    ///
    /// [`ConfigError`] when the address is not a URI or a pool size is zero.
    ///
    /// # Panics
    ///
    /// Panics when called outside a tokio runtime, which the channels' workers need.
    pub fn new(
        settings: &EngineSettings,
        observer: Option<Arc<dyn CallObserver>>,
    ) -> Result<Self, ConfigError> {
        settings.check()?;
        let channels = (0..settings.channels)
            .map(|_| engine_channel(&settings.url))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            channels,
            next: AtomicUsize::new(0),
            permits: Semaphore::new(settings.max_in_flight),
            call_timeout: settings.call_timeout,
            deadline: settings.deadline,
            ready: Mutex::new(HashMap::new()),
            observer,
        })
    }

    fn channel(&self, index: usize) -> Client {
        self.channels[index % self.channels.len()].clone()
    }

    fn round_robin(&self) -> Client {
        self.channel(self.next.fetch_add(1, Ordering::Relaxed))
    }

    /// One call, under the in-flight cap and the call timeout.
    async fn call(
        &self,
        request: &ModelInferRequest,
        give_up_at: Instant,
    ) -> Result<ModelInferResponse, Status> {
        let wait = give_up_at.saturating_duration_since(Instant::now());
        let _permit = match timeout(wait, self.permits.acquire()).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => return Err(Status::unavailable("the engine client is closed")),
            Err(_) => {
                return Err(Status::resource_exhausted(
                    "the in-flight cap stayed full until the photo's deadline",
                ));
            }
        };
        let mut client = self.round_robin();
        match timeout(
            self.call_timeout,
            client.model_infer(Request::new(request.clone())),
        )
        .await
        {
            Ok(response) => response.map(tonic::Response::into_inner),
            Err(_) => Err(Status::deadline_exceeded(format!(
                "no answer within {:?}",
                self.call_timeout
            ))),
        }
    }

    /// The engine's answer for the encoded photo `image` under `plan`; `boxes` (stored
    /// `bbox_norm` rows) stand in for the detector.
    ///
    /// An overloaded engine is asked again with a jittered backoff until the photo's deadline,
    /// an unavailable engine or a call lost in transit [`retry::RETRIES`] times, and nothing else
    /// at all (see [`crate::retry`]). A call that ends in an error drops the cached readiness
    /// verdicts, so the next photo asks the engine again.
    ///
    /// # Errors
    ///
    /// [`EngineError`]: the engine gave no answer for the photo, or its answer is not one this
    /// client reads.
    pub async fn process(
        &self,
        image: &[u8],
        plan: &EnginePlan,
        boxes: Option<&[[f64; 4]]>,
    ) -> Result<Answer, EngineError> {
        let text = plan.to_json().map_err(EngineError::Plan)?;
        let request = infer_request(image, &text, boxes);
        let started = Instant::now();
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let status = match self.call(&request, started + self.deadline).await {
                Ok(response) => {
                    return Answer::parse(&response, plan).map_err(EngineError::Answer);
                }
                Err(status) => status,
            };
            let code = crate::error::code_name(status.code());
            let class = classify(&status);
            let verdict = retry::decide(
                class,
                attempt,
                started.elapsed(),
                self.deadline,
                fastrand::f64(),
            );
            match verdict {
                Verdict::After(wait) => {
                    if let Some(observer) = &self.observer {
                        observer.retried(code);
                    }
                    sleep(wait).await;
                }
                Verdict::Stop => {
                    if let Some(observer) = &self.observer {
                        observer.exhausted(code);
                    }
                    self.forget_ready();
                    return Err(failure(class, &status, attempt));
                }
            }
        }
    }

    fn forget_ready(&self) {
        self.ready
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    /// Asks `model` (the engine by default: [`MODEL`]) whether it is ready, up to
    /// [`PROBE_ATTEMPTS`] times on successive channels, each limited to [`PROBE_TIMEOUT`]. A probe
    /// the transport refuses is not the engine saying "not ready": it is asked again, 0.1 s and
    /// 0.2 s apart; an engine that refuses connections is judged in about 0.3 s, one that accepts
    /// and never answers in under 6.3 s.
    pub async fn probe_ready(&self, model: &str) -> Readiness {
        let mut last = (String::new(), String::new());
        for attempt in 1..=PROBE_ATTEMPTS {
            let mut client = self.channel(usize::try_from(attempt - 1).unwrap_or(0));
            let request = ModelReadyRequest {
                name: model.to_owned(),
                version: String::new(),
            };
            let answer = timeout(PROBE_TIMEOUT, client.model_ready(Request::new(request))).await;
            match answer {
                Ok(Ok(response)) => {
                    let outcome = match (response.into_inner().ready, attempt) {
                        (false, _) => ProbeOutcome::NotReady,
                        (true, 1) => ProbeOutcome::Ready,
                        (true, _) => ProbeOutcome::Recovered,
                    };
                    return Readiness {
                        outcome,
                        attempts: attempt,
                        code: String::new(),
                        error: String::new(),
                    };
                }
                Ok(Err(status)) => {
                    last = (
                        crate::error::code_name(status.code()).to_owned(),
                        status.to_string(),
                    );
                }
                Err(_) => {
                    last = (
                        "DEADLINE_EXCEEDED".to_owned(),
                        format!("no answer within {PROBE_TIMEOUT:?}"),
                    );
                }
            }
            if attempt < PROBE_ATTEMPTS {
                sleep(Duration::from_secs_f64(
                    (0.1 * f64::from(1u32 << (attempt - 1))).min(0.5),
                ))
                .await;
            }
        }
        Readiness {
            outcome: ProbeOutcome::Undelivered,
            attempts: PROBE_ATTEMPTS,
            code: last.0,
            error: last.1,
        }
    }

    /// Whether the engine can take photos of this detector shape now: it says ready, and the
    /// detector and encoders in `models` are loaded in the model server (explicit model control:
    /// a model that is not is loaded here). A positive verdict is kept for [`READY_TTL`]; a failed
    /// probe is not kept, and a failed call drops the verdicts ([`EngineClient::process`]).
    pub async fn ready_for(
        &self,
        detector: &DetectorSpec,
        models: &[&str],
        server: &ModelServer,
    ) -> bool {
        let fresh = |at: &Instant| at.elapsed() < READY_TTL;
        if self.verdicts().get(detector).is_some_and(fresh) {
            return true;
        }
        let readiness = self.probe_ready(MODEL).await;
        if let Some(observer) = &self.observer {
            observer.probed(readiness.outcome);
        }
        log_probe(&readiness);
        if !readiness.ready() {
            self.verdicts().remove(detector);
            return false;
        }
        for model in models {
            if server.is_model_ready(model).await {
                continue;
            }
            if let Err(error) = server.load_model(model).await {
                tracing::warn!(%error, "ingest_engine_unavailable");
                self.verdicts().remove(detector);
                return false;
            }
        }
        self.verdicts().insert(detector.clone(), Instant::now());
        true
    }

    fn verdicts(&self) -> std::sync::MutexGuard<'_, HashMap<DetectorSpec, Instant>> {
        self.ready.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

fn log_probe(readiness: &Readiness) {
    match readiness.outcome {
        ProbeOutcome::NotReady => tracing::warn!("ingest_engine_not_ready"),
        ProbeOutcome::Undelivered => tracing::warn!(
            attempts = readiness.attempts,
            code = %readiness.code,
            error = %readiness.error,
            "ingest_engine_probe_undelivered"
        ),
        ProbeOutcome::Recovered => {
            tracing::info!(
                attempts = readiness.attempts,
                "ingest_engine_probe_recovered"
            );
        }
        ProbeOutcome::Ready => {}
    }
}

/// The error of a failed call that will not be sent again.
fn failure(class: Class, status: &Status, attempts: u32) -> EngineError {
    let message = status.message().to_owned();
    match (class, status.code()) {
        (Class::Overload, _) => EngineError::Overloaded { attempts, message },
        (Class::Transient, _) => EngineError::Unavailable { attempts, message },
        (Class::Final, code @ (tonic::Code::InvalidArgument | tonic::Code::NotFound)) => {
            EngineError::Rejected { code, message }
        }
        (Class::Final, code) => EngineError::Failed {
            code,
            attempts,
            message,
        },
    }
}
