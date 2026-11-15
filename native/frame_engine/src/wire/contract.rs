//! `contracts/json/frame_engine_v1.json`, embedded at build time and read once: the tensor
//! table, vocabularies and messages the front and the parser use. Never hand-copied here.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde::Deserialize;

const TEXT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/json/frame_engine_v1.json"
));

#[derive(Debug, Deserialize)]
pub struct Contract {
    pub model: Model,
    pub inputs: Vec<InputTensor>,
    pub outputs: Vec<Tensor>,
    pub always_present_outputs: Vec<String>,
    pub size_output: String,
    pub branches: Vec<String>,
    pub branch_outputs: BTreeMap<String, Vec<String>>,
    pub crops_inline_outputs: Vec<String>,
    pub derivatives: Derivatives,
    pub request_rules: Vec<RequestRule>,
    pub model_ready: ModelReady,
    pub decode_routes: Vec<String>,
    pub photo_statuses: Vec<String>,
    pub plan: PlanRules,
}

#[derive(Debug, Deserialize)]
pub struct Model {
    pub name: String,
    pub version: String,
    pub platform: String,
    pub protocol_version: i64,
    pub engine_error_prefix: String,
}

#[derive(Debug, Deserialize)]
pub struct Tensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
}

#[derive(Debug, Deserialize)]
pub struct InputTensor {
    pub name: String,
    pub datatype: String,
    pub shape: Vec<i64>,
    pub required: bool,
}

#[derive(Debug, Deserialize)]
pub struct Derivatives {
    pub field: String,
    pub output: String,
    pub formats: Vec<String>,
    pub side_range: [i64; 2],
    pub default_quality: i64,
}

#[derive(Debug, Deserialize)]
pub struct RequestRule {
    pub id: String,
    pub message: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelReady {
    pub versions_accepted: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct PlanRules {
    pub crop_fill_default: i64,
    pub crop_recipes: Vec<String>,
    pub crops_modes: Vec<String>,
    pub frame_recipes: Vec<String>,
    pub ocr_det_limits: Vec<String>,
    pub ocr_max_batch: i64,
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
