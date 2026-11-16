//! The engine's `source` parser against `contracts/json/frame_engine_source_cases.json`: every
//! document of `scripts/codegen/frame_engine_source_cases.py` with its answer. Same accept or
//! reject, the same file reference, the same fault and message (but for `bad_source_json`, whose
//! text is the JSON decoder's own).

use frame_engine::wire::contract;
use frame_engine::wire::source::parse_source;
use serde_json::Value;

const CASES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../contracts/json/frame_engine_source_cases.json"
));

fn cases() -> Vec<Value> {
    let document: Value = serde_json::from_str(CASES).expect("the cases file is JSON");
    document["cases"].as_array().expect("cases").clone()
}

#[test]
fn the_parser_answers_every_case_as_the_table_says() {
    let all = cases();
    assert!(all.len() >= 50, "{} cases", all.len());
    let mut failures = Vec::new();
    for case in &all {
        let id = case["id"].as_str().expect("id");
        let text = case["text"].as_str().expect("text");
        match (case.get("fault"), parse_source(text.as_bytes())) {
            (None, Ok(file)) => {
                let claim = file.claim();
                let got = (file.key(), claim.size, claim.imohash.to_hex());
                let want = (
                    case["key"].as_str().expect("key"),
                    case["size"].as_u64().expect("size"),
                    case["imohash"].as_str().expect("imohash").to_owned(),
                );
                if got != want {
                    failures.push(format!("{id}: read {got:?}, expected {want:?}"));
                }
            }
            (None, Err(e)) => {
                failures.push(format!("{id}: refused ({e}) but the table accepts it"))
            }
            (Some(fault), Ok(file)) => {
                failures.push(format!(
                    "{id}: read {} but the table says {fault}",
                    file.key()
                ));
            }
            (Some(fault), Err(e)) => {
                let details = e.to_string();
                let fault = fault.as_str().expect("fault");
                let message = case.get("message").and_then(Value::as_str);
                let ok = match message {
                    Some(message) => details == message,
                    None => details.starts_with(&format!("{fault}: ")),
                };
                if !ok {
                    failures.push(format!("{id}: {details:?}, expected {fault} {message:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn every_fault_the_contract_names_has_a_case_but_the_one_from_the_settings() {
    let named = &contract().read_by_key.source_faults;
    let seen: std::collections::BTreeSet<String> = cases()
        .iter()
        .filter_map(|c| c.get("fault").and_then(Value::as_str).map(str::to_owned))
        .collect();
    let want: std::collections::BTreeSet<String> = named
        .iter()
        .filter(|f| *f != "root_not_configured")
        .cloned()
        .collect();
    assert_eq!(seen, want);
}
