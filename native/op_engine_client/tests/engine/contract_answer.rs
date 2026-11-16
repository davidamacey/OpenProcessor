//! The closed vocabularies of the answer against the contract file
//! (`contracts/json/frame_engine_v1.json`, generated from the Python wire module and front).

use std::collections::BTreeSet;

use op_engine_client::answer::expected_outputs;
use op_engine_client::{
    Answer, Branch, BranchError, CropsMode, DecodeRoute, EnginePlan, PhotoStatus, RenditionSpec,
};
use serde_json::Value;

use crate::answers::{cases, response_of, response_with};
use crate::support::repo_json;

const CONTRACT: &str = "contracts/json/frame_engine_v1.json";

fn strings(value: &Value) -> Vec<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn the_photo_statuses_are_the_contracts() {
    let contract = repo_json(CONTRACT);
    let ours: Vec<String> = PhotoStatus::ALL
        .iter()
        .map(|(_, n)| (*n).to_owned())
        .collect();
    let mut want = strings(&contract["photo_statuses"]);
    want.extend(
        contract["read_by_key"]["photo_statuses"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["status"].as_str().unwrap().to_owned()),
    );
    assert_eq!(
        ours, want,
        "the protocol's statuses, then the read-by-key ones"
    );
}

#[test]
fn the_decode_routes_are_the_contracts() {
    let contract = repo_json(CONTRACT);
    let ours: Vec<String> = DecodeRoute::ALL
        .iter()
        .map(|(_, n)| (*n).to_owned())
        .collect();
    assert_eq!(ours, strings(&contract["decode_routes"]));
}

#[test]
fn the_branches_and_their_outputs_are_the_contracts() {
    let contract = repo_json(CONTRACT);
    let names: Vec<String> = Branch::ALL.iter().map(|b| b.name().to_owned()).collect();
    assert_eq!(names, strings(&contract["branches"]));
    for branch in Branch::ALL {
        let plan = EnginePlan {
            branches: BTreeSet::from([branch]),
            ..EnginePlan::default()
        };
        let want = strings(&contract["branch_outputs"][branch.name()]);
        assert_eq!(expected_outputs(&plan), want, "{}", branch.name());
    }
    let crops = EnginePlan {
        crops: CropsMode::Inline,
        ..EnginePlan::default()
    };
    assert_eq!(
        expected_outputs(&crops),
        strings(&contract["crops_inline_outputs"])
    );
    let renditions = EnginePlan {
        derivatives: vec![
            serde_json::from_str::<RenditionSpec>(r#"{"name": "t", "side": 64}"#).unwrap(),
        ],
        ..EnginePlan::default()
    };
    assert_eq!(
        expected_outputs(&renditions),
        [contract["derivatives"]["output"].as_str().unwrap()]
    );
}

#[test]
fn every_output_of_the_contract_is_read_by_the_goldens_of_the_python_answers() {
    let contract = repo_json(CONTRACT);
    let wanted: BTreeSet<String> = contract["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o["name"].as_str().unwrap().to_owned())
        .collect();
    let mut seen: BTreeSet<String> = ["status", "decode_route", "branch_status"]
        .map(str::to_owned)
        .into();
    for case in cases()["cases"].as_array().unwrap() {
        let response = response_of(case);
        for output in &response.outputs {
            // the datatype and shape each one carries are the contract's
            let spec = contract["outputs"]
                .as_array()
                .unwrap()
                .iter()
                .find(|o| o["name"] == output.name.as_str())
                .unwrap();
            assert_eq!(
                spec["datatype"],
                output.datatype.as_str(),
                "{}",
                output.name
            );
            seen.insert(output.name.clone());
        }
    }
    assert_eq!(seen, wanted);
}

#[test]
fn the_branch_states_and_error_reasons_are_the_contracts() {
    let contract = repo_json(CONTRACT);
    let reasons: Vec<String> = BranchError::ALL
        .iter()
        .map(|(_, n)| (*n).to_owned())
        .collect();
    assert_eq!(reasons, strings(&contract["branch_error_reasons"]));
    let plan: EnginePlan = serde_json::from_str(
        r#"{"version": 1, "branches": ["detect"], "detector": {"model": "d", "input_size": 640, "letterbox_fill": 114}}"#,
    )
    .unwrap();
    for state in strings(&contract["branch_states"]) {
        let entry = if state == "error" {
            format!(
                r#"{{"detect": {{"status": "error", "reason": "{}"}}}}"#,
                reasons[0]
            )
        } else {
            format!(r#"{{"detect": {{"status": "{state}"}}}}"#)
        };
        let answer = Answer::parse(&response_with("corrupt", "none", &entry), &plan);
        assert!(answer.is_ok(), "{state}: {answer:?}");
    }
}
