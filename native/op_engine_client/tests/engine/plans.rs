//! The plan text against what the Python builder writes
//! (`contracts/json/frame_engine_plan_build_cases.json`, from
//! `scripts/codegen/frame_engine_plan_build_cases.py`).

use std::collections::BTreeSet;

use op_engine_client::plan::pyjson::float_repr;
use op_engine_client::{Branch, CropVectors, DetectorSpec, EncoderModels, EnginePlan, IngestPlan};
use serde_json::Value;

use crate::support::repo_json;

const CASES: &str = "contracts/json/frame_engine_plan_build_cases.json";

fn encoders(value: &Value) -> EncoderModels {
    EncoderModels {
        image_model: value["image_model"].as_str().unwrap().to_owned(),
        crop_model: value["crop_model"].as_str().unwrap().to_owned(),
        crop_size: u32::try_from(value["crop_size"].as_u64().unwrap()).unwrap(),
    }
}

#[test]
fn every_plan_the_python_parser_accepts_is_written_as_python_writes_it() {
    let cases = repo_json(CASES);
    let parsed = cases["parsed"].as_array().unwrap();
    assert!(parsed.len() > 40, "the case file covers the vocabulary");
    for case in parsed {
        let id = case["id"].as_str().unwrap();
        let plan: EnginePlan = serde_json::from_str(case["input"].as_str().unwrap())
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(
            plan.to_json().unwrap(),
            case["text"].as_str().unwrap(),
            "{id}"
        );
    }
}

#[test]
fn the_cases_reach_every_branch_and_every_optional_part() {
    let cases = repo_json(CASES);
    let plans: Vec<EnginePlan> = cases["parsed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| serde_json::from_str(c["input"].as_str().unwrap()).unwrap())
        .collect();
    for branch in Branch::ALL {
        assert!(plans.iter().any(|p| p.runs(branch)), "{branch:?}");
    }
    assert!(
        plans
            .iter()
            .any(|p| p.faces.is_some() && p.ocr.is_some() || p.ocr.is_some())
    );
    assert!(plans.iter().any(|p| !p.derivatives.is_empty()));
    assert!(plans.iter().any(|p| !p.if_detected.is_empty()));
    assert!(plans.iter().any(|p| p.embed_classes.is_some()));
}

#[test]
fn the_ingest_plan_is_what_the_python_server_builds() {
    let cases = repo_json(CASES);
    for case in cases["ingest"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let detector: DetectorSpec = serde_json::from_value(case["detector"].clone()).unwrap();
        let choice = &case["crop_vectors"];
        let crop_vectors = match choice["kind"].as_str().unwrap() {
            "deferred" => CropVectors::Deferred,
            "every" => CropVectors::Every,
            "classes" => CropVectors::Classes(
                choice["classes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                    .collect::<BTreeSet<u32>>(),
            ),
            other => panic!("{id}: {other}"),
        };
        let plan = EnginePlan::ingest(
            &IngestPlan {
                detector,
                crop_vectors,
            },
            &encoders(&case["encoders"]),
        );
        assert_eq!(
            plan.to_json().unwrap(),
            case["text"].as_str().unwrap(),
            "{id}"
        );
    }
}

#[test]
fn the_stored_boxes_plan_is_what_the_python_reprocess_sends() {
    let cases = repo_json(CASES);
    for case in cases["stored_boxes"].as_array().unwrap() {
        let plan = EnginePlan::stored_boxes_embed(&encoders(&case["encoders"]));
        assert_eq!(plan.to_json().unwrap(), case["text"].as_str().unwrap());
    }
}

#[test]
fn model_names_are_escaped_as_python_escapes_them() {
    let cases = repo_json(CASES);
    for case in cases["strings"].as_array().unwrap() {
        let model = case["model"].as_str().unwrap();
        let plan = EnginePlan {
            branches: BTreeSet::from([Branch::Detect]),
            detector: Some(DetectorSpec {
                model: model.to_owned(),
                input_size: 640,
                letterbox_fill: 114,
            }),
            ..EnginePlan::default()
        };
        assert_eq!(
            plan.to_json().unwrap(),
            case["text"].as_str().unwrap(),
            "{model:?}"
        );
    }
}

#[test]
fn floats_are_spelled_as_python_spells_them() {
    let cases = repo_json(CASES);
    for case in cases["floats"].as_array().unwrap() {
        let bits = u64::from_str_radix(case["bits"].as_str().unwrap(), 16).unwrap();
        assert_eq!(
            float_repr(f64::from_bits(bits)),
            case["repr"].as_str().unwrap()
        );
    }
}

#[test]
fn a_plan_of_another_protocol_version_is_not_read() {
    let text = r#"{"version": 2, "branches": ["detect"]}"#;
    let error = serde_json::from_str::<EnginePlan>(text)
        .unwrap_err()
        .to_string();
    assert!(error.contains("protocol version 2"), "{error}");
}

#[test]
fn encoder_settings_come_from_the_environment_with_the_python_defaults() {
    use op_config::Env;
    let defaults = EncoderModels::from_env(&Env::from_pairs::<_, String, String>([])).unwrap();
    assert_eq!(defaults.image_model, "pe_image_encoder");
    assert_eq!(defaults.crop_model, "pe_crop_encoder");
    assert_eq!(defaults.crop_size, 224);
    let set = Env::from_pairs([("PE_CROP_SIZE", "336"), ("PE_IMAGE_MODEL", "img")]);
    let set = EncoderModels::from_env(&set).unwrap();
    assert_eq!((set.image_model.as_str(), set.crop_size), ("img", 336));
    let bad = Env::from_pairs([("PE_CROP_SIZE", "300")]);
    let error = EncoderModels::from_env(&bad).unwrap_err().to_string();
    assert!(error.contains("PE_CROP_SIZE"), "{error}");
}
