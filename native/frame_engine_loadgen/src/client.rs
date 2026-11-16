//! Protocol version 1 as a client: one unary `ModelInfer` per photo to the model `frame_engine`,
//! the inputs `image` and `plan` (BYTES, one element each, in `raw_input_contents`, an element
//! being a 4-byte little-endian length and its bytes), no `outputs` filter (as the API sends,
//! so the answer is the whole answer), and the answer read for `status`, `decode_route` and
//! `branch_status` (`contracts/json/frame_engine_v1.json`; `src/clients/frame_engine.py`).
//!
//! The channels carry the API's settings (`ENGINE_CHANNEL_OPTIONS`): HTTP/2 keepalive every 30 s
//! with a 10 s timeout, also idle; 1 MiB stream and 2 MiB connection receive windows; no adaptive
//! window; 512 MiB messages both ways
//! (<https://docs.rs/tonic/0.14.9/tonic/transport/channel/struct.Endpoint.html>). No retry: a
//! refusal is a result, not something to hide behind a second try.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use bytes::Bytes;
use op_engine_client::codec::{pack_bytes, unpack_bytes};
use serde_json::Value;
use tonic::transport::{Channel, Endpoint};

use crate::exchange::{Exchange, Outcome};
use crate::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::proto::model_infer_request::InferInputTensor;
use crate::proto::{
    CudaSharedMemoryStatusRequest, ModelInferRequest, ModelInferResponse, ModelReadyRequest,
    ServerReadyRequest,
};

/// A gRPC client of one HTTP/2 connection to the engine or the model server; cloning it shares the connection.
pub type Client = GrpcInferenceServiceClient<Channel>;

/// The model name and version of protocol version 1.
pub const MODEL: &str = "frame_engine";
const MIB: usize = 1 << 20;
/// `ENGINE_CHANNEL_OPTIONS`' message limit, both ways.
const MAX_MESSAGE_BYTES: usize = 512 * MIB;

/// `host:port` as an `<http://`> URI; a URI with a scheme is kept.
pub fn endpoint_uri(url: &str) -> String {
    if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    }
}

/// Open `channels` connections to `url`, each its own HTTP/2 connection.
///
/// # Errors
///
/// Returns `Err` when `channels` is 0, the URL is not a valid URI, or a connection is not made within `timeout`.
pub async fn connect(url: &str, channels: usize, timeout: Duration) -> Result<Vec<Client>, String> {
    if channels == 0 {
        return Err("at least one channel is needed".into());
    }
    let endpoint = Endpoint::from_shared(endpoint_uri(url))
        .map_err(|e| format!("{url}: {e}"))?
        .tcp_nodelay(true)
        .http2_keep_alive_interval(Duration::from_secs(30))
        .keep_alive_timeout(Duration::from_secs(10))
        .keep_alive_while_idle(true)
        .initial_stream_window_size(u32::try_from(MIB).ok())
        .initial_connection_window_size(u32::try_from(2 * MIB).ok())
        .http2_adaptive_window(false)
        .connect_timeout(timeout);
    let mut clients = Vec::with_capacity(channels);
    for _ in 0..channels {
        let channel = endpoint
            .connect()
            .await
            .map_err(|e| format!("cannot connect to {url}: {e}"))?;
        clients.push(
            Client::new(channel)
                .max_decoding_message_size(MAX_MESSAGE_BYTES)
                .max_encoding_message_size(MAX_MESSAGE_BYTES),
        );
    }
    Ok(clients)
}

fn bytes_input(name: &str) -> InferInputTensor {
    InferInputTensor {
        name: name.to_string(),
        datatype: "BYTES".to_string(),
        shape: vec![1],
        parameters: Default::default(),
        contents: None,
    }
}

/// The request for one photo: `image` then `plan`, both already packed.
pub fn infer_request(id: usize, image: &Bytes, plan: &Bytes) -> ModelInferRequest {
    ModelInferRequest {
        model_name: MODEL.to_string(),
        id: id.to_string(),
        inputs: vec![bytes_input("image"), bytes_input("plan")],
        raw_input_contents: vec![image.clone(), plan.clone()],
        ..Default::default()
    }
}

fn first_element(response: &ModelInferResponse, name: &str) -> Option<String> {
    let at = response.outputs.iter().position(|o| o.name == name)?;
    let raw = response.raw_output_contents.get(at)?;
    let items = unpack_bytes(raw)?;
    String::from_utf8(items.first()?.to_vec()).ok()
}

/// How the engine's answer to photo `id` ended: `status` `ok` and no branch in error is a
/// success; everything else is a failure under a key a reader can group by. When several
/// branches are in error the key names the first by name (`serde_json`'s plain object map is
/// sorted), so the same answer always lands under the same key.
pub fn outcome_of(response: &ModelInferResponse, id: usize) -> Outcome {
    let failed = |key: String| Outcome::Failed { key };
    if response.id != id.to_string() {
        return failed(format!("protocol:id_echo:{}", response.id));
    }
    let Some(status) = first_element(response, "status") else {
        return failed("protocol:no_status".into());
    };
    if status != "ok" {
        return failed(format!("status:{status}"));
    }
    let route = first_element(response, "decode_route").unwrap_or_default();
    if let Some(branches) = first_element(response, "branch_status") {
        match serde_json::from_str::<Value>(&branches) {
            Ok(Value::Object(map)) => {
                for (branch, state) in &map {
                    if state.get("status").and_then(Value::as_str) == Some("error") {
                        let reason = state.get("reason").and_then(Value::as_str).unwrap_or("?");
                        return failed(format!("branch:{branch}:{reason}"));
                    }
                }
            }
            _ => return failed("protocol:branch_status".into()),
        }
    } else {
        return failed("protocol:no_branch_status".into());
    }
    let bytes = response
        .raw_output_contents
        .iter()
        .map(|c| c.len() as u64)
        .sum();
    Outcome::Ok { route, bytes }
}

/// `ResourceExhausted` as `RESOURCE_EXHAUSTED`: the names the plan's status table uses.
fn code_name(code: tonic::Code) -> String {
    let mut out = String::new();
    for (i, c) in format!("{code:?}").chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(c.to_ascii_uppercase());
    }
    out
}

/// The failure key of a gRPC status.
pub fn grpc_key(status: &tonic::Status) -> String {
    format!("grpc:{}", code_name(status.code()))
}

/// The engine as an [`Exchange`]: photo `index` of the list, over the channels round robin.
pub struct EngineClient {
    clients: Vec<Client>,
    next: AtomicUsize,
    /// Each photo packed once, so a request clones a reference, never the photo.
    images: Vec<Bytes>,
    plan: Bytes,
    timeout: Duration,
}

impl EngineClient {
    /// An engine client that spreads photos over `clients` round robin, packing every photo and the `plan` text once; each exchange is bounded by `timeout`.
    pub fn new(clients: Vec<Client>, photos: &[Bytes], plan: &str, timeout: Duration) -> Self {
        EngineClient {
            clients,
            next: AtomicUsize::new(0),
            images: photos.iter().map(|p| pack_bytes(p)).collect(),
            plan: pack_bytes(plan.as_bytes()),
            timeout,
        }
    }

    fn channel(&self) -> Client {
        self.clients[self.next.fetch_add(1, Ordering::Relaxed) % self.clients.len()].clone()
    }
}

impl Exchange for EngineClient {
    async fn exchange(&self, index: usize) -> Outcome {
        let mut request =
            tonic::Request::new(infer_request(index, &self.images[index], &self.plan));
        request.set_timeout(self.timeout);
        let mut client = self.channel();
        match tokio::time::timeout(self.timeout, client.model_infer(request)).await {
            Ok(Ok(response)) => outcome_of(&response.into_inner(), index),
            Ok(Err(status)) => Outcome::Failed {
                key: grpc_key(&status),
            },
            Err(_) => Outcome::Failed {
                key: "client:timeout".into(),
            },
        }
    }
}

/// Whether the server answers `ServerReady` and `ModelReady` for `model` (`frame_engine` for an
/// engine; a model's own name for the model server): the gate before any row.
///
/// # Errors
///
/// Returns `Err` when the `ServerReady` or `ModelReady` call fails (the server is unreachable or answers an error status); a server that answers not ready is `Ok(false)`.
pub async fn is_ready(client: &mut Client, model: &str, timeout: Duration) -> Result<bool, String> {
    let mut server = tonic::Request::new(ServerReadyRequest {});
    server.set_timeout(timeout);
    let server = client
        .server_ready(server)
        .await
        .map_err(|e| format!("ServerReady: {e}"))?
        .into_inner()
        .ready;
    let mut model = tonic::Request::new(ModelReadyRequest {
        name: model.to_string(),
        version: String::new(),
    });
    model.set_timeout(timeout);
    let model = client
        .model_ready(model)
        .await
        .map_err(|e| format!("ModelReady: {e}"))?
        .into_inner()
        .ready;
    Ok(server && model)
}

/// The names of the CUDA shared-memory regions the model server holds, from its status call
/// (the shared-memory extension: an empty name asks for every region).
///
/// # Errors
///
/// Returns `Err` when the `CudaSharedMemoryStatus` call fails.
pub async fn cuda_regions(client: &mut Client, timeout: Duration) -> Result<Vec<String>, String> {
    let mut request = tonic::Request::new(CudaSharedMemoryStatusRequest::default());
    request.set_timeout(timeout);
    let response = client
        .cuda_shared_memory_status(request)
        .await
        .map_err(|e| format!("CudaSharedMemoryStatus: {e}"))?
        .into_inner();
    let mut names: Vec<String> = response.regions.into_keys().collect();
    names.sort();
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::model_infer_response::InferOutputTensor;

    fn answer(branch_status: &str) -> ModelInferResponse {
        let outputs = [
            ("status", "ok"),
            ("decode_route", "device"),
            ("branch_status", branch_status),
        ];
        ModelInferResponse {
            id: "7".into(),
            outputs: outputs
                .iter()
                .map(|(name, _)| InferOutputTensor {
                    name: (*name).into(),
                    ..Default::default()
                })
                .collect(),
            raw_output_contents: outputs
                .iter()
                .map(|(_, text)| pack_bytes(text.as_bytes()))
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn an_answer_with_no_branch_in_error_is_a_success() {
        let response = answer(r#"{"detect": {"status": "ok"}}"#);
        assert!(matches!(
            outcome_of(&response, 7),
            Outcome::Ok { ref route, .. } if route == "device"
        ));
    }

    #[test]
    fn several_branches_in_error_are_named_by_the_first_by_name() {
        let response = answer(
            r#"{"vectors": {"status": "error", "reason": "late"},
                "faces": {"status": "error", "reason": "early"}}"#,
        );
        assert!(matches!(
            outcome_of(&response, 7),
            Outcome::Failed { ref key } if key == "branch:faces:early"
        ));
    }
}
