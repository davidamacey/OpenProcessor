//! The response of one photo: its tensors packed as protocol version 1 writes them, and the
//! gRPC status of each way a photo fails.

use tonic::Status;

use crate::metrics::Metrics;
use crate::pipeline::{PhotoAnswer, PipelineError, Tensor};
use crate::proto::model_infer_response::InferOutputTensor;
use crate::proto::{ModelInferRequest, ModelInferResponse};
use crate::wire::branch_status;
use crate::wire::bytes_tensor::pack;
use crate::wire::contract::contract;

/// The response tensors of `answer`, filtered to `wanted` when the request named any: the
/// three always present first, then the photo's own in the order the pipeline gave them.
pub(super) fn build_response(
    request: &ModelInferRequest,
    answer: PhotoAnswer,
) -> Result<ModelInferResponse, Status> {
    let c = contract();
    let mut response = ModelInferResponse {
        model_name: c.model.name.clone(),
        model_version: c.model.version.clone(),
        id: request.id.clone(),
        ..Default::default()
    };
    let wanted: Vec<&str> = request.outputs.iter().map(|o| o.name.as_str()).collect();
    let mut add = |name: &str, datatype: &str, shape: Vec<i64>, payload: bytes::Bytes| {
        if wanted.is_empty() || wanted.contains(&name) {
            response.outputs.push(InferOutputTensor {
                name: name.to_owned(),
                datatype: datatype.to_owned(),
                shape,
                ..Default::default()
            });
            response.raw_output_contents.push(payload);
        }
    };
    let engine_error =
        |text: String| Status::internal(format!("{}{text}", c.model.engine_error_prefix));
    let text = |value: &str| pack(&[value.as_bytes()]);
    let [status, route, branches] = &c.always_present_outputs[..] else {
        return Err(engine_error(
            "the contract names no always-present outputs".into(),
        ));
    };
    add(status, "BYTES", vec![1], text(&answer.status));
    add(route, "BYTES", vec![1], text(&answer.decode_route));
    add(
        branches,
        "BYTES",
        vec![1],
        text(&branch_status::encode(&answer.branches)),
    );
    for (name, tensor) in answer.outputs {
        let Some(declared) = c.tensor(&name) else {
            return Err(engine_error(format!(
                "output {name} is not in protocol version 1"
            )));
        };
        if declared.datatype != tensor.datatype() {
            return Err(engine_error(format!(
                "output {name} is {}, the contract says {}",
                tensor.datatype(),
                declared.datatype
            )));
        }
        let (shape, payload) = match tensor {
            Tensor::Bytes(items) => (
                vec![i64::try_from(items.len()).unwrap_or(i64::MAX)],
                pack(&items),
            ),
            Tensor::Bool { shape, data } => (
                shape,
                data.iter()
                    .map(|b| u8::from(*b))
                    .collect::<Vec<u8>>()
                    .into(),
            ),
            Tensor::Int32 { shape, data } => (
                shape,
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into(),
            ),
            Tensor::Int64 { shape, data } => (
                shape,
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into(),
            ),
            Tensor::Fp32 { shape, data } => (
                shape,
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into(),
            ),
            Tensor::Fp64 { shape, data } => (
                shape,
                data.iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
                    .into(),
            ),
        };
        add(&name, &declared.datatype, shape, payload);
    }
    Ok(response)
}

pub(super) fn failure_status(failure: PipelineError, metrics: &Metrics) -> Status {
    match failure {
        PipelineError::Overload(overload) => {
            metrics.count_overload(&overload.cause);
            Status::resource_exhausted(format!("overload: {overload}"))
        }
        PipelineError::ModelServerUnavailable(why) => {
            Status::unavailable(format!("model server unavailable: {why}"))
        }
        PipelineError::ContextLost(why) => {
            Status::unavailable(format!("frame engine restarting: {why}"))
        }
        PipelineError::Engine(why) => {
            tracing::error!("frame engine request failed: {why}");
            Status::internal(format!("{}{why}", contract().model.engine_error_prefix))
        }
    }
}
