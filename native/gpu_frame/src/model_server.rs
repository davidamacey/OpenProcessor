//! The model-call filling for a KServe v2 model server (Triton) over its CUDA shared-memory
//! extension: tonic channels, the region's registration and its rules, and the infer call
//! (plan 17 sections 1.3, 2.2, 2.4 and 7; the Python engine's `transport.py`).
//!
//! The inputs travel one of two ways, chosen once at start ([`InputMode`],
//! `OP_FRAME_ENGINE_TRANSPORT`): as slices of the registered region (`cuda`), or copied to the
//! host and sent in the request's `raw_input_contents` (`body`, the Python `BodyTransport`, for a
//! server that cannot import the region). In body mode the region is still allocated and every
//! input is still a slice of it (the slab bounds the bytes in flight), but it is not exported
//! (`Region::local`) and nothing is registered, purged or unregistered: the health tick asks
//! readiness only. Each host copy belongs to its slice's call, so the host bytes in flight are
//! bounded by the region. The outputs come back in `raw_output_contents` in both modes.
//!
//! Region rules, all kept:
//! * at start every region the server holds under this engine's prefix is unregistered first: a
//!   killed predecessor leaves its region registered and its memory pinned in the server;
//! * the region is registered with its 64-byte IPC handle as `raw_handle`;
//! * the health tick checks the server holds the region and registers it again when it does not
//!   (a restarted server forgets every registration); a call that uses the region and is refused
//!   with `INVALID_ARGUMENT` or `NOT_FOUND` asks the server's status, re-registers when the region
//!   is missing (single flight, by registration generation) and is sent once more from the same
//!   slices by the same deadline. The extension specifies no error text and no lifetime across
//!   restarts, so the text "Unable to find shared memory region" is never relied on
//!   (<https://github.com/triton-inference-server/server/blob/r26.10/docs/protocol/extension_shared_memory.md>);
//! * shutdown unregisters, then frees; the memory is freed only after a successful unregister
//!   (the region type enforces it, `region.rs`);
//! * an unregister request never carries an empty name: the extension unregisters every region
//!   of every client for an empty name.
//!
//! Channel settings (plan 17 section 2.4, the Python engine's `CHANNEL_OPTIONS`): HTTP/2 keepalive
//! every 30 s with a 10 s timeout, also while idle (Triton accepts pings 20 s apart), 512 MiB
//! messages both ways, `OP_FRAME_ENGINE_TRITON_CHANNELS` connections each opened lazily, connect
//! timeout = the call timeout (<https://docs.rs/tonic/0.14.9/tonic/transport/channel/struct.Endpoint.html>).
//! A call's bound is set both as its `grpc-timeout` (`tonic::Request::set_timeout`) and as a tokio
//! timeout; dropping the call's future cancels its HTTP/2 stream.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use tokio::time::Instant;
use tonic::transport::{Channel, Endpoint};
use tonic::{Code, Status};

use crate::model_calls::{CallError, InferCall, ModelCalls, OutputTensor, call_timeout};
use crate::region::Region;

mod registration;
pub use registration::{Health, RegistrationCause, stale_regions};
use registration::{Identity, Registrar, ServerApi};

#[expect(
    clippy::all,
    clippy::pedantic,
    clippy::allow_attributes_without_reason,
    reason = "generated from Triton's protos by build.rs; its lints are the generator's, not this crate's"
)]
pub(crate) mod proto {
    tonic::include_proto!("inference");
}

use proto::grpc_inference_service_client::GrpcInferenceServiceClient as Client;
use proto::infer_parameter::ParameterChoice;
use proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};

/// The message limit both ways (the Python engine's 512 MiB).
pub const MAX_MESSAGE_BYTES: usize = 512 << 20;

/// How the model inputs reach the server, fixed at start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    /// Slices of the registered region (Triton's CUDA shared-memory extension).
    SharedMemory,
    /// Copied to the host and sent in the request body; the region is never registered.
    Body,
}

/// How the engine reaches its model server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelServerConfig {
    /// `host:port` of the gRPC endpoint, or a full `<http://host:port`> URI.
    pub url: String,
    /// HTTP/2 connections, used round robin.
    pub channels: usize,
    /// Each call's bound before the request's own deadline is applied; also the connect timeout.
    pub call_timeout: Duration,
    /// The prefix every region of this card's engine carries; purged at start. Never empty.
    pub region_prefix: String,
    /// This process's region name; starts with the prefix.
    pub region_name: String,
    /// The card as the model server numbers it.
    pub device_id: i64,
    /// How the inputs travel.
    pub inputs: InputMode,
}

/// A configuration refused before any connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// The server URL is not a valid endpoint; carries the parser's message.
    Url(String),
    /// The region name or prefix breaks the rule that the name extends a non-empty prefix.
    RegionName(String),
    /// Zero channels were asked for.
    Channels,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Url(m) => write!(f, "model server URL: {m}"),
            ConfigError::RegionName(m) => write!(f, "region name: {m}"),
            ConfigError::Channels => write!(f, "at least one model server channel is needed"),
        }
    }
}

impl std::error::Error for ConfigError {}

/// `host:port` as an `<http://`> URI; a URI with a scheme is kept.
pub fn endpoint_uri(url: &str) -> String {
    if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    }
}

fn check_names(prefix: &str, name: &str) -> Result<(), ConfigError> {
    if prefix.is_empty() {
        return Err(ConfigError::RegionName(
            "the prefix is empty (a purge would reach every region)".into(),
        ));
    }
    if !name.starts_with(prefix) || name.len() == prefix.len() {
        return Err(ConfigError::RegionName(format!(
            "{name:?} must extend the prefix {prefix:?}"
        )));
    }
    Ok(())
}

/// The tonic side of [`ServerApi`]: the shared-memory and readiness calls go to channel 0, as in
/// the Python engine.
struct TritonApi {
    clients: Vec<Client<Channel>>,
    next: AtomicUsize,
}

impl TritonApi {
    fn control(&self) -> Client<Channel> {
        self.clients[0].clone()
    }

    fn any(&self) -> Client<Channel> {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len();
        self.clients[i].clone()
    }
}

fn timed<T>(message: T, timeout: Duration) -> tonic::Request<T> {
    let mut r = tonic::Request::new(message);
    r.set_timeout(timeout);
    r
}

async fn bounded<T>(
    timeout: Duration,
    call: impl Future<Output = Result<T, Status>>,
) -> Result<T, Status> {
    tokio::time::timeout(timeout, call)
        .await
        .unwrap_or_else(|_| {
            Err(Status::deadline_exceeded(
                "no answer within the call's deadline",
            ))
        })
}

impl ServerApi for TritonApi {
    async fn server_ready(&self, timeout: Duration) -> Result<bool, Status> {
        let mut c = self.control();
        let r = bounded(
            timeout,
            c.server_ready(timed(proto::ServerReadyRequest {}, timeout)),
        )
        .await?;
        Ok(r.into_inner().ready)
    }

    async fn region_names(&self, timeout: Duration) -> Result<Vec<String>, Status> {
        let mut c = self.control();
        let req = proto::CudaSharedMemoryStatusRequest {
            name: String::new(),
        };
        let r = bounded(timeout, c.cuda_shared_memory_status(timed(req, timeout))).await?;
        Ok(r.into_inner().regions.into_keys().collect())
    }

    async fn register(
        &self,
        name: &str,
        raw_handle: &[u8],
        device_id: i64,
        bytes: u64,
        timeout: Duration,
    ) -> Result<(), Status> {
        let mut c = self.control();
        let req = proto::CudaSharedMemoryRegisterRequest {
            name: name.to_string(),
            raw_handle: raw_handle.to_vec(),
            device_id,
            byte_size: bytes,
        };
        bounded(timeout, c.cuda_shared_memory_register(timed(req, timeout)))
            .await
            .map(|_| ())
    }

    async fn unregister(&self, name: &str, timeout: Duration) -> Result<(), Status> {
        if name.is_empty() {
            return Err(Status::invalid_argument(
                "refused: an empty name unregisters every region",
            ));
        }
        let mut c = self.control();
        let req = proto::CudaSharedMemoryUnregisterRequest {
            name: name.to_string(),
        };
        bounded(
            timeout,
            c.cuda_shared_memory_unregister(timed(req, timeout)),
        )
        .await
        .map(|_| ())
    }
}

/// The model server filling of [`ModelCalls`], owning the registered region.
pub struct ModelServer {
    registrar: Registrar<TritonApi>,
    region: Region,
    call_timeout: Duration,
    inputs: InputMode,
}

impl fmt::Debug for ModelServer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelServer")
            .field("region", &self.registrar.name())
            .field("ready", &self.is_ready())
            .finish()
    }
}

impl ModelServer {
    /// Channels to the server (opened lazily, so the server may start after the engine) and the
    /// region they will register. Call inside a tokio runtime. Nothing is sent until
    /// [`ModelServer::check`].
    ///
    /// # Errors
    ///
    /// [`ConfigError::RegionName`] when the region name does not extend a non-empty prefix,
    /// [`ConfigError::Channels`] when zero channels are asked for, [`ConfigError::Url`] when the
    /// URL is not a valid endpoint.
    pub fn new(cfg: ModelServerConfig, region: Region) -> Result<ModelServer, ConfigError> {
        check_names(&cfg.region_prefix, &cfg.region_name)?;
        if cfg.channels == 0 {
            return Err(ConfigError::Channels);
        }
        let endpoint = Endpoint::from_shared(endpoint_uri(&cfg.url))
            .map_err(|e| ConfigError::Url(e.to_string()))?
            .tcp_nodelay(true)
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true)
            .connect_timeout(cfg.call_timeout);
        let clients = (0..cfg.channels)
            .map(|_| {
                Client::new(endpoint.connect_lazy())
                    .max_decoding_message_size(MAX_MESSAGE_BYTES)
                    .max_encoding_message_size(MAX_MESSAGE_BYTES)
            })
            .collect();
        let api = TritonApi {
            clients,
            next: AtomicUsize::new(0),
        };
        let id = Identity {
            prefix: cfg.region_prefix,
            name: cfg.region_name,
            handle: *region.raw_handle(),
            bytes: region.len() as u64,
            device_id: cfg.device_id,
        };
        let registrar = Registrar::new(
            api,
            id,
            cfg.inputs,
            cfg.call_timeout,
            region.registration_flag(),
        );
        Ok(ModelServer {
            registrar,
            region,
            call_timeout: cfg.call_timeout,
            inputs: cfg.inputs,
        })
    }

    /// The health tick: the server answers ready and holds the region. The first successful
    /// check purges this prefix's leftovers and registers; later ones register again when the
    /// server lost the region.
    pub async fn check(&self) -> Health {
        self.registrar.check().await
    }

    /// Ready as of the last check or registration (one atomic read).
    pub fn is_ready(&self) -> bool {
        self.registrar.is_ready()
    }

    /// The region the calls read from.
    pub fn region(&self) -> &Region {
        &self.region
    }

    /// The region's registered name.
    pub fn region_name(&self) -> &str {
        self.registrar.name()
    }

    /// How many registrations happened for `cause`.
    pub fn registrations(&self, cause: RegistrationCause) -> u64 {
        self.registrar.registrations(cause)
    }

    /// Unregister the region, then drop this filling. Returns whether the server released the
    /// region: its memory is then freed when the last slice of it is dropped; otherwise it is
    /// left to process exit.
    pub async fn close(self) -> bool {
        self.registrar.close().await
    }

    async fn send(&self, call: &InferCall<'_>) -> Result<proto::ModelInferResponse, Status> {
        let Some(timeout) = call_timeout(self.call_timeout, call.deadline, Instant::now()) else {
            return Err(Status::deadline_exceeded("the request deadline passed"));
        };
        let inputs = request_inputs(call, self.inputs)
            .map_err(|m| Status::invalid_argument(format!("body transport: {m}")))?;
        let request = infer_request(self.registrar.name(), call.model, inputs, call.outputs);
        let mut client = self.registrar.api().any();
        bounded(timeout, client.model_infer(timed(request, timeout)))
            .await
            .map(tonic::Response::into_inner)
    }

    async fn infer_once_more(
        &self,
        call: &InferCall<'_>,
        first: Status,
    ) -> Result<proto::ModelInferResponse, CallError> {
        let seen = self.registrar.generation();
        match self.registrar.api().region_names(self.call_timeout).await {
            Ok(names) if names.iter().any(|n| n == self.registrar.name()) => {
                return Err(classify(&first));
            }
            Ok(_) => {}
            Err(status) => return Err(classify(&status)),
        }
        self.registrar
            .reregister(seen, RegistrationCause::RegionUnknown)
            .await
            .map_err(|status| match classify(&status) {
                CallError::Model { message, .. } => CallError::RegionUnknown(message),
                other => other,
            })?;
        self.send(call).await.map_err(|s| classify(&s))
    }
}

impl ModelCalls for ModelServer {
    async fn infer<'a>(&'a self, call: InferCall<'a>) -> Result<Vec<OutputTensor>, CallError> {
        if let Some(i) = call.inputs.iter().find(|i| !self.region.owns(i.slice)) {
            return Err(CallError::Model {
                code: "InvalidArgument".into(),
                message: format!("input {} is not a slice of the registered region", i.name),
            });
        }
        let answer = match self.send(&call).await {
            Ok(r) => Ok(r),
            Err(s)
                if self.inputs == InputMode::SharedMemory
                    && !call.inputs.is_empty()
                    && matches!(s.code(), Code::InvalidArgument | Code::NotFound) =>
            {
                self.infer_once_more(&call, s).await
            }
            Err(s) => Err(classify(&s)),
        };
        answer.and_then(|r| outputs_of(r, call.outputs))
    }
}

/// A call's status as the engine answers it (the Python engine's `_GONE` set).
pub fn classify(status: &Status) -> CallError {
    match status.code() {
        Code::Unavailable | Code::Cancelled => CallError::Unavailable(status.message().to_string()),
        Code::DeadlineExceeded => CallError::Deadline,
        code => CallError::Model {
            code: format!("{code:?}"),
            message: status.message().to_string(),
        },
    }
}

fn param(choice: ParameterChoice) -> proto::InferParameter {
    proto::InferParameter {
        parameter_choice: Some(choice),
    }
}

/// Where one input's bytes are: a byte range of the registered region, or in the body.
enum Source {
    Region { offset: usize, len: usize },
    Body(Bytes),
}

/// One input as the request names it.
struct RequestInput<'a> {
    name: &'a str,
    datatype: &'a str,
    shape: &'a [i64],
    source: Source,
}

/// The `ModelInfer` request: a region input carries the extension's `shared_memory_region`,
/// `shared_memory_byte_size` and `shared_memory_offset` parameters; a body input's bytes go into
/// `raw_input_contents` in input order (KServe v2: when that field is used, every input is in it).
fn infer_request(
    region: &str,
    model: &str,
    inputs: Vec<RequestInput<'_>>,
    outputs: &[&str],
) -> proto::ModelInferRequest {
    let mut raw = Vec::new();
    let inputs = inputs
        .into_iter()
        .map(|i| {
            let parameters = match i.source {
                Source::Region { offset, len } => region_parameters(region, offset, len),
                Source::Body(bytes) => {
                    raw.push(bytes);
                    HashMap::new()
                }
            };
            InferInputTensor {
                name: i.name.to_string(),
                datatype: i.datatype.to_string(),
                shape: i.shape.to_vec(),
                parameters,
                contents: None,
            }
        })
        .collect();
    let outputs = outputs
        .iter()
        .map(|name| InferRequestedOutputTensor {
            name: (*name).to_string(),
            parameters: HashMap::new(),
        })
        .collect();
    proto::ModelInferRequest {
        model_name: model.to_string(),
        inputs,
        outputs,
        raw_input_contents: raw,
        ..Default::default()
    }
}

fn region_parameters(
    region: &str,
    offset: usize,
    len: usize,
) -> HashMap<String, proto::InferParameter> {
    HashMap::from([
        (
            "shared_memory_region".to_string(),
            param(ParameterChoice::StringParam(region.to_string())),
        ),
        (
            "shared_memory_byte_size".to_string(),
            param(ParameterChoice::Int64Param(len as i64)),
        ),
        (
            "shared_memory_offset".to_string(),
            param(ParameterChoice::Int64Param(offset as i64)),
        ),
    ])
}

/// The call's inputs as `mode` sends them; a body input must carry its slice's bytes.
fn request_inputs<'a>(
    call: &'a InferCall<'a>,
    mode: InputMode,
) -> Result<Vec<RequestInput<'a>>, String> {
    call.inputs
        .iter()
        .map(|i| {
            let source = match (mode, i.body) {
                (InputMode::SharedMemory, _) => Source::Region {
                    offset: i.slice.offset(),
                    len: i.slice.len(),
                },
                (InputMode::Body, Some(b)) if b.len() == i.slice.len() => Source::Body(b.clone()),
                (InputMode::Body, Some(b)) => {
                    return Err(format!(
                        "input {} carries {} bytes for a slice of {}",
                        i.name,
                        b.len(),
                        i.slice.len()
                    ));
                }
                (InputMode::Body, None) => {
                    return Err(format!("input {} has no host copy", i.name));
                }
            };
            Ok(RequestInput {
                name: i.name,
                datatype: i.datatype,
                shape: i.shape,
                source,
            })
        })
        .collect()
}

/// The asked outputs, in the asked order, with their raw bytes (no copy).
fn outputs_of(
    response: proto::ModelInferResponse,
    wanted: &[&str],
) -> Result<Vec<OutputTensor>, CallError> {
    let proto::ModelInferResponse {
        outputs,
        raw_output_contents,
        ..
    } = response;
    if raw_output_contents.len() != outputs.len() {
        return Err(CallError::Model {
            code: "Internal".into(),
            message: format!(
                "{} outputs with {} raw contents",
                outputs.len(),
                raw_output_contents.len()
            ),
        });
    }
    let mut by_name: HashMap<String, OutputTensor> = outputs
        .into_iter()
        .zip(raw_output_contents)
        .map(|(o, data)| {
            (
                o.name.clone(),
                OutputTensor {
                    name: o.name,
                    datatype: o.datatype,
                    shape: o.shape,
                    data,
                },
            )
        })
        .collect();
    wanted
        .iter()
        .map(|w| {
            by_name.remove(*w).ok_or_else(|| CallError::Model {
                code: "Internal".into(),
                message: format!("output {w} missing from the answer"),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;

    #[test]
    fn statuses_map_as_the_python_engine_maps_them() {
        assert_eq!(
            classify(&Status::unavailable("down")),
            CallError::Unavailable("down".into())
        );
        assert_eq!(
            classify(&Status::cancelled("reset")),
            CallError::Unavailable("reset".into())
        );
        assert_eq!(
            classify(&Status::deadline_exceeded("late")),
            CallError::Deadline
        );
        assert_eq!(
            classify(&Status::invalid_argument("bad shape")),
            CallError::Model {
                code: "InvalidArgument".into(),
                message: "bad shape".into()
            }
        );
    }

    #[test]
    fn urls_gain_a_scheme_and_names_must_extend_the_prefix() {
        assert_eq!(
            endpoint_uri("triton-server:8001"),
            "http://triton-server:8001"
        );
        assert_eq!(endpoint_uri("http://h:1"), "http://h:1");
        assert!(check_names("op-fe-", "op-fe-card-fe0-host-1").is_ok());
        assert!(
            check_names("", "anything").is_err(),
            "an empty prefix would purge every region"
        );
        assert!(check_names("op-fe-", "op-fe-").is_err());
        assert!(check_names("op-fe-a-", "op-fe-b-1").is_err());
    }

    #[test]
    fn inputs_are_named_as_slices_of_the_region() {
        let shape = [1i64, 3, 640, 640];
        let inputs = vec![RequestInput {
            name: "images",
            datatype: "UINT8",
            shape: &shape,
            source: Source::Region {
                offset: 4096,
                len: 1_228_800,
            },
        }];
        let r = infer_request("op-fe-x-fe0-h-1", "detector", inputs, &["output0"]);
        assert_eq!(r.model_name, "detector");
        assert_eq!(
            r.outputs
                .iter()
                .map(|o| o.name.as_str())
                .collect::<Vec<_>>(),
            ["output0"]
        );
        let i = &r.inputs[0];
        assert_eq!(
            (i.name.as_str(), i.datatype.as_str(), i.shape.as_slice()),
            ("images", "UINT8", &shape[..])
        );
        assert!(
            i.contents.is_none() && r.raw_input_contents.is_empty(),
            "nothing travels in the body"
        );
        let p = |k: &str| i.parameters.get(k).and_then(|v| v.parameter_choice.clone());
        assert_eq!(
            p("shared_memory_region"),
            Some(ParameterChoice::StringParam("op-fe-x-fe0-h-1".into()))
        );
        assert_eq!(
            p("shared_memory_byte_size"),
            Some(ParameterChoice::Int64Param(1_228_800))
        );
        assert_eq!(
            p("shared_memory_offset"),
            Some(ParameterChoice::Int64Param(4096))
        );
        assert_eq!(i.parameters.len(), 3);
    }

    #[test]
    fn body_inputs_travel_in_raw_input_contents_with_the_same_names_and_shapes() {
        let (shape_a, shape_b) = ([1i64, 3, 4, 4], [2i64, 3, 2, 2]);
        let (a, b) = (Bytes::from(vec![7u8; 48]), Bytes::from(vec![9u8; 24]));
        let inputs = vec![
            RequestInput {
                name: "images",
                datatype: "UINT8",
                shape: &shape_a,
                source: Source::Body(a.clone()),
            },
            RequestInput {
                name: "crops",
                datatype: "UINT8",
                shape: &shape_b,
                source: Source::Body(b.clone()),
            },
        ];
        let r = infer_request("op-fe-x-fe0-h-1", "m", inputs, &["out"]);
        assert_eq!(r.raw_input_contents, [a.clone(), b]);
        assert_eq!(
            r.raw_input_contents[0].as_ptr(),
            a.as_ptr(),
            "moved, not copied"
        );
        assert!(
            r.inputs
                .iter()
                .all(|i| i.parameters.is_empty() && i.contents.is_none())
        );
        assert_eq!(
            r.inputs
                .iter()
                .map(|i| (i.name.as_str(), i.shape.clone()))
                .collect::<Vec<_>>(),
            [("images", shape_a.to_vec()), ("crops", shape_b.to_vec())]
        );
        assert_eq!(
            r.outputs
                .iter()
                .map(|o| o.name.as_str())
                .collect::<Vec<_>>(),
            ["out"]
        );
    }

    #[test]
    fn outputs_come_back_in_the_asked_order_without_a_copy() {
        let tensor = |name: &str| proto::model_infer_response::InferOutputTensor {
            name: name.into(),
            datatype: "FP32".into(),
            shape: vec![1, 2],
            ..Default::default()
        };
        let a = Bytes::from_static(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let b = Bytes::from_static(&[9; 8]);
        let response = proto::ModelInferResponse {
            outputs: vec![tensor("a"), tensor("b")],
            raw_output_contents: vec![a.clone(), b.clone()],
            ..Default::default()
        };
        let got = outputs_of(response.clone(), &["b", "a"]).unwrap();
        assert_eq!(got[0].name, "b");
        assert_eq!(got[1].data, a);
        assert_eq!(
            got[1].data.as_ptr(),
            a.as_ptr(),
            "the bytes are the response's own"
        );
        assert!(matches!(
            outputs_of(response, &["c"]),
            Err(CallError::Model { .. })
        ));
    }
}
