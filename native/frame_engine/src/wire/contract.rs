//! `contracts/json/frame_engine_v1.json`, embedded at build time and read once: the tensor
//! table, vocabularies and messages the front and the parser use. Never hand-copied here.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::Deserialize;

const TEXT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/json/frame_engine_v1.json"
));

/// The contract as `frame_engine_v1.json` states it: the tensor table, vocabularies, plan rules, request messages and metric families both sides must agree on.
#[derive(Debug, Deserialize)]
pub struct Contract {
    /// Identity of the model the API calls.
    pub model: Model,
    /// The request inputs in order: `image`, `plan`, then the optional `boxes`.
    pub inputs: Vec<InputTensor>,
    /// Every output tensor the engine can answer, with its datatype and shape.
    pub outputs: Vec<Tensor>,
    /// Outputs every answer carries whatever the plan: `status`, `decode_route`, `branch_status`.
    pub always_present_outputs: Vec<String>,
    /// Name of the output holding the photo's `[width, height]`.
    pub size_output: String,
    /// Branch names a plan may list.
    pub branches: Vec<String>,
    /// For each branch, the output tensors it answers.
    pub branch_outputs: BTreeMap<String, Vec<String>>,
    /// Outputs added by `crops: inline`.
    pub crops_inline_outputs: Vec<String>,
    /// The rendition (`derivatives`) field name, output and limits.
    pub derivatives: Derivatives,
    /// Request faults with their message text; `{name}` stands for the tensor name.
    pub request_rules: Vec<RequestRule>,
    /// Which model versions `ModelReady` accepts.
    pub model_ready: ModelReady,
    /// The values of the `decode_route` output.
    pub decode_routes: Vec<String>,
    /// The values of the `status` output.
    pub photo_statuses: Vec<String>,
    /// Limits and vocabularies the plan parser enforces.
    pub plan: PlanRules,
    /// The stage names of the stage metrics, in order.
    pub stages: Vec<String>,
    /// The metric families the engine serves.
    pub metrics: Vec<MetricFamily>,
    /// The optional `source` input, by which the native engine reads a photo from a file.
    pub read_by_key: ReadByKey,
}

/// The `read_by_key` extension: an input the Python engine does not have.
#[derive(Debug, Deserialize)]
pub struct ReadByKey {
    /// The engines that have the input.
    pub supported_by: Vec<String>,
    /// The input beside `image`; exactly one of the two travels in a request.
    pub input: InputTensor,
    /// Refusals of a request that sends both or neither, with their message text.
    pub request_rules: Vec<RequestRule>,
    /// The faults a `source` document or the engine's settings can raise, in check order.
    pub source_faults: Vec<String>,
    /// The photo statuses a read can answer, and when.
    pub photo_statuses: Vec<ReadStatus>,
}

/// A photo status of a read by key.
#[derive(Debug, Deserialize)]
pub struct ReadStatus {
    /// The `status` output's value.
    pub status: String,
    /// When the engine answers it.
    pub when: String,
}

/// One series family of the metrics endpoint.
#[derive(Debug, Deserialize)]
pub struct MetricFamily {
    /// Prometheus series name.
    pub name: String,
    /// Series kind, `counter` or `gauge`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Label names of the series.
    pub labels: Vec<String>,
}

/// The model's identity as the API addresses it.
#[derive(Debug, Deserialize)]
pub struct Model {
    /// Model name in requests.
    pub name: String,
    /// Model version in requests and metadata.
    pub version: String,
    /// Platform string in the metadata answer.
    pub platform: String,
    /// Version of the plan protocol; a plan's `version` must equal it.
    pub protocol_version: i64,
    /// Prefix of the details of an `INTERNAL` answer.
    pub engine_error_prefix: String,
}

/// An output tensor in the metadata.
#[derive(Debug, Deserialize)]
pub struct Tensor {
    /// Tensor name.
    pub name: String,
    /// KServe datatype name.
    pub datatype: String,
    /// Dimensions; `-1` is a variable size.
    pub shape: Vec<i64>,
}

/// A request input in the metadata.
#[derive(Debug, Deserialize)]
pub struct InputTensor {
    /// Tensor name.
    pub name: String,
    /// KServe datatype name.
    pub datatype: String,
    /// Dimensions; `-1` is a variable size.
    pub shape: Vec<i64>,
    /// Whether a request without it is refused.
    pub required: bool,
}

/// How image renditions are asked for and returned.
#[derive(Debug, Deserialize)]
pub struct Derivatives {
    /// Name of the plan field that lists the renditions.
    pub field: String,
    /// Name of the output tensor holding the encoded renditions.
    pub output: String,
    /// Accepted rendition formats.
    pub formats: Vec<String>,
    /// Smallest and largest rendition side, pixels.
    pub side_range: [i64; 2],
    /// JPEG quality (1 to 100) when a rendition names none.
    pub default_quality: i64,
}

/// A way a request is refused, with its message.
#[derive(Debug, Deserialize)]
pub struct RequestRule {
    /// Rule id, for example `unknown_input`.
    pub id: String,
    /// Message text; `{name}` is replaced by the tensor name.
    pub message: String,
}

/// What `ModelReady` accepts.
#[derive(Debug, Deserialize)]
pub struct ModelReady {
    /// Version strings that mean this model; the empty string means the latest.
    pub versions_accepted: Vec<String>,
}

/// The limits and vocabularies of a plan.
#[derive(Debug, Deserialize)]
pub struct PlanRules {
    /// Letterbox fill, 0 to 255, when the crop encoder names none.
    pub crop_fill_default: i64,
    /// Accepted `crop_recipe` values.
    pub crop_recipes: Vec<String>,
    /// Accepted `crops` values.
    pub crops_modes: Vec<String>,
    /// Accepted `frame_recipe` values.
    pub frame_recipes: Vec<String>,
    /// Accepted OCR `det_limit` values.
    pub ocr_det_limits: Vec<String>,
    /// Largest OCR `batch`.
    pub ocr_max_batch: i64,
    /// Largest OCR `det_side`, pixels.
    pub ocr_max_det_side: i64,
    /// Branches an `if_detected` condition may name.
    pub conditional_branches: Vec<String>,
}

impl Contract {
    /// The tensor of an input or output by name.
    pub fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.outputs.iter().find(|t| t.name == name)
    }

    /// The text of a request rule with the tensor name filled in: a template that quotes the
    /// name (`'{name}'`) gets Python's `repr` of it, as the Python front writes it.
    pub fn request_message(&self, rule: &str, name: &str) -> String {
        let template = self
            .request_rules
            .iter()
            .chain(&self.read_by_key.request_rules)
            .find(|r| r.id == rule)
            .map_or(rule, |r| r.message.as_str());
        if template.contains("'{name}'") {
            template.replace("'{name}'", &crate::pyfmt::repr_str(name))
        } else {
            template.replace("{name}", name)
        }
    }
}

/// The embedded contract.
pub fn contract() -> &'static Contract {
    static CONTRACT: LazyLock<Contract> = LazyLock::new(|| {
        serde_json::from_str(TEXT).expect("contracts/json/frame_engine_v1.json is valid")
    });
    &CONTRACT
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_embedded_contract_names_the_model_the_api_calls() {
        let c = contract();
        assert_eq!(c.model.name, "frame_engine");
        assert_eq!(c.model.version, "1");
        assert_eq!(c.model.platform, "frame_engine");
        assert_eq!(c.model.protocol_version, 1);
        assert_eq!(c.model.engine_error_prefix, "frame engine: ");
        assert_eq!(c.model_ready.versions_accepted, ["", "1"]);
    }

    #[test]
    fn the_tensor_table_is_the_python_tables() {
        let c = contract();
        let inputs: Vec<_> = c
            .inputs
            .iter()
            .map(|t| (t.name.as_str(), t.datatype.as_str()))
            .collect();
        assert_eq!(
            inputs,
            [("image", "BYTES"), ("plan", "BYTES"), ("boxes", "FP64")]
        );
        assert!(c.inputs.iter().filter(|t| t.required).count() == 2);
        assert_eq!(c.outputs.len(), 27);
        let size = c.tensor("size").expect("size");
        assert_eq!(
            (size.datatype.as_str(), size.shape.as_slice()),
            ("INT32", [2].as_slice())
        );
        let letterbox = c.tensor("letterbox").expect("letterbox");
        assert_eq!(
            (letterbox.datatype.as_str(), letterbox.shape.as_slice()),
            ("FP64", [3].as_slice())
        );
        let crops = c.tensor("crop_embeddings").expect("crop_embeddings");
        assert_eq!(crops.shape, [-1, -1]);
        assert_eq!(
            c.always_present_outputs,
            ["status", "decode_route", "branch_status"]
        );
        assert_eq!(
            c.branches,
            [
                "detect",
                "embed_frame",
                "embed_crops",
                "faces",
                "face_embed",
                "ocr"
            ]
        );
    }

    #[test]
    fn a_request_rule_fills_in_the_name_as_the_python_front_does() {
        let c = contract();
        assert_eq!(
            c.request_message("unknown_input", "extra"),
            "unknown input 'extra'"
        );
        assert_eq!(
            c.request_message("unknown_input", "it's"),
            "unknown input \"it's\""
        );
        assert_eq!(
            c.request_message("bytes_input_shape", "image"),
            "image must be BYTES [1]"
        );
        assert_eq!(
            c.request_message("raw_contents_count", ""),
            "every input must travel in raw_input_contents"
        );
    }
}
