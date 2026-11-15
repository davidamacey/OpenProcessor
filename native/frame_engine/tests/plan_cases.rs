//! The native plan parser against `contracts/json/frame_engine_plan_cases.json`: every plan text
//! the Python parser was run on, with its answer. Same accept or reject, same fault, the same
//! message (but for `bad_json`, whose text is the Python JSON decoder's own), and a parsed plan
//! equal field by field.

use frame_engine::wire::{contract, parse_plan};
use serde_json::Value;

const CASES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/json/frame_engine_plan_cases.json"
));

/// A JSON value with every number as (is it a float, its value): the Python and Rust writers
/// spell `1.0` and `1e-05` differently, but never disagree on what they mean.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Number(n) => {
            let text = n.to_string();
            let is_float = text.contains(['.', 'e', 'E']);
            let x: f64 = text.parse().expect("a number");
            Value::Array(vec![Value::Bool(is_float), Value::from(x)])
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), canonical(v))).collect())
        }
        other => other.clone(),
    }
}

fn cases() -> Vec<Value> {
    let document: Value = serde_json::from_str(CASES).expect("the cases file is JSON");
    document["cases"]
        .as_array()
        .expect("a list of cases")
        .clone()
}

#[test]
fn the_parser_answers_every_case_as_python_does() {
    let all = cases();
    assert!(all.len() >= 200, "{} cases", all.len());
    let mut failures = Vec::new();
    for case in &all {
        let id = case["id"].as_str().expect("id");
        let text = case["plan"].as_str().expect("plan");
        let has_boxes = case["has_boxes"].as_bool().expect("has_boxes");
        let answer = parse_plan(text.as_bytes(), has_boxes);
        match (case.get("fault"), answer) {
            (None, Ok(plan)) => {
                let got = serde_json::to_value(&plan).expect("plan serialises");
                if canonical(&got) != canonical(&case["parsed"]) {
                    failures.push(format!(
                        "{id}: parsed {got} but Python parsed {}",
                        case["parsed"]
                    ));
                }
            }
            (None, Err(e)) => failures.push(format!("{id}: refused ({e}) but Python accepted it")),
            (Some(want), Ok(_)) => failures.push(format!("{id}: accepted but Python said {want}")),
            (Some(want), Err(e)) => {
                if want.as_str() != Some(e.fault) {
                    failures.push(format!("{id}: fault {} but Python said {want}", e.fault));
                } else if let Some(message) = case.get("message") {
                    if message.as_str() != Some(e.to_string().as_str()) {
                        failures.push(format!("{id}: message {e:?} but Python said {message}"));
                    }
                } else if e.fault != "bad_json" {
                    failures.push(format!("{id}: the case has no message"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        all.len(),
        failures.join("\n")
    );
}

#[test]
fn every_fault_of_the_contract_is_met_by_a_case() {
    let all = cases();
    let seen: std::collections::BTreeSet<&str> =
        all.iter().filter_map(|c| c["fault"].as_str()).collect();
    let faults = {
        let v: Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../contracts/json/frame_engine_v1.json"
        )))
        .expect("the contract is JSON");
        v["plan"]["faults"]
            .as_array()
            .expect("faults")
            .iter()
            .map(|f| f.as_str().expect("a name").to_owned())
            .collect::<Vec<_>>()
    };
    for fault in &faults {
        assert!(seen.contains(fault.as_str()), "no case for {fault}");
    }
    assert_eq!(contract().model.protocol_version, 1);
}
