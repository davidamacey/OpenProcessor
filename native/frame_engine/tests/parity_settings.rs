//! Every `OP_FRAME_ENGINE_*` setting against the Python engine's: the environments of
//! `native/frame_engine/tests/golden/settings_cases.json`, each run through the Python
//! `EngineSettings.from_env` by `scripts/codegen/frame_engine_parity_goldens.py`, answered here
//! with the same settings or the same refusal text. The settings the engines answer differently
//! are listed in the file with their reasons, and each is pinned below.

use std::collections::HashMap;

use frame_engine::settings::{Settings, Transport};
use serde_json::{Map, Value, json};

const CASES: &str = include_str!("golden/settings_cases.json");

fn load() -> Value {
    serde_json::from_str(CASES).expect("the committed cases are JSON")
}

fn read(env: &HashMap<String, String>) -> Result<Settings, String> {
    Settings::from_lookup(&|name| env.get(name).cloned()).map_err(|e| e.to_string())
}

/// The settings under the Python field names, numbers as numbers.
fn fields(s: &Settings) -> Map<String, Value> {
    let Value::Object(map) = json!({
        "triton_url": s.triton_url,
        "transport": match s.transport { Transport::Cuda => "cuda", Transport::Body => "body" },
        "region_mib": s.region_mib,
        "instance": s.instance,
        "triton_device_id": s.triton_device_id,
        "triton_channels": s.triton_channels,
        "triton_timeout_seconds": s.triton_timeout_seconds,
        "request_deadline_seconds": s.request_deadline_seconds,
        "workers": s.workers,
        "lanes": s.lanes,
        "encode_threads": s.encode_threads,
        "max_in_flight": s.max_in_flight,
        "pool_mib": s.pool_mib,
        "budget_mib": s.budget_mib,
        "budget_wait_seconds": s.budget_wait_seconds,
        "slab_wait_seconds": s.slab_wait_seconds,
        "colour_tables": s.colour_tables,
        "min_megapixels": s.min_megapixels,
        "max_megapixels": s.max_megapixels,
        "jpeg_quality": s.jpeg_quality,
        "crop_max_side": s.crop_max_side,
        "stage_sync": s.stage_sync,
        "port": s.port,
        "metrics_port": s.metrics_port,
        "max_message_mib": s.max_message_mib,
        "drain_seconds": s.drain_seconds,
        "health_seconds": s.health_seconds,
    }) else {
        unreachable!("an object literal")
    };
    map
}

fn same(a: &Value, b: &Value) -> bool {
    match (a.as_f64(), b.as_f64()) {
        (Some(x), Some(y)) => x == y,
        _ => a == b,
    }
}

#[test]
fn every_environment_is_answered_as_the_python_engine_answers_it() {
    let doc = load();
    let differences = doc["differences"].as_object().expect("differences");
    let mut replayed = 0;
    for case in doc["cases"].as_array().expect("cases") {
        let id = case["id"].as_str().expect("id");
        let env: HashMap<String, String> = case["env"]
            .as_object()
            .expect("env")
            .iter()
            .map(|(k, v)| (k.clone(), v.as_str().expect("text").to_owned()))
            .collect();
        // an empty variable is an unset one in both engines
        if env
            .get("OP_FRAME_ENGINE_SWITCH_INTERVAL_MS")
            .is_some_and(|v| !v.trim().is_empty())
        {
            assert!(
                differences.contains_key("OP_FRAME_ENGINE_SWITCH_INTERVAL_MS"),
                "{id}"
            );
            let text = read(&env).expect_err("a start with the switch interval set is refused");
            assert!(text.contains("OP_FRAME_ENGINE_SWITCH_INTERVAL_MS"), "{id}");
            continue;
        }
        replayed += 1;
        match (read(&env), case.get("ok"), case.get("error")) {
            (Ok(got), Some(want), None) => {
                let got = fields(&got);
                for (key, want) in want.as_object().expect("settings") {
                    // the default lane count is the engine's own (cores, then memory)
                    let lanes_set = env
                        .get("OP_FRAME_ENGINE_LANES")
                        .is_some_and(|v| !v.trim().is_empty());
                    if key == "lanes" && !lanes_set {
                        continue;
                    }
                    assert!(
                        same(&got[key], want),
                        "{id}: {key} is {} not {want}",
                        got[key]
                    );
                }
            }
            (Err(got), None, Some(want)) => assert_eq!(&got, want.as_str().expect("text"), "{id}"),
            (got, ok, error) => panic!("{id}: native {got:?}, python ok {ok:?} error {error:?}"),
        }
    }
    assert!(replayed > 250, "only {replayed} cases replayed");
}

#[test]
fn every_range_python_reads_is_read_by_the_native_engine() {
    let doc = load();
    let ranges = doc["ranges"].as_array().expect("ranges");
    assert!(ranges.len() >= 20);
    let source = include_str!("../src/settings.rs");
    for range in ranges {
        let name = range["name"].as_str().expect("name");
        assert!(
            source.contains(&format!("\"{name}\"")),
            "{name} is not read"
        );
    }
}

#[test]
fn the_native_only_settings_are_listed_with_a_reason() {
    let doc = load();
    let refused = doc["native_only_refused_values"]
        .as_object()
        .expect("native_only_refused_values");
    for (name, reason) in doc["native_only"].as_object().expect("native_only") {
        assert!(!reason.as_str().expect("reason").is_empty(), "{name}");
        match refused.get(name) {
            Some(bad) => {
                let bad = bad.as_str().expect("a refused value is text").to_owned();
                let env = HashMap::from([(name.clone(), bad)]);
                let text = read(&env).expect_err("a bad value is refused");
                assert!(text.contains(name), "{text}");
            }
            None => {
                let env = HashMap::from([(name.clone(), "/data/photos".to_owned())]);
                assert!(read(&env).is_ok(), "{name} takes any text");
            }
        }
    }
    for name in refused.keys() {
        assert!(
            doc["native_only"].get(name).is_some(),
            "{name} is not native only"
        );
    }
}

/// The `OP_FRAME_ENGINE_*` names in `settings.rs` outside its test module.
fn names_read_by_the_native_engine() -> Vec<String> {
    let source = include_str!("../src/settings.rs");
    let product = source.split("#[cfg(test)]").next().expect("a first part");
    let mut names: Vec<String> = product
        .split("\"OP_FRAME_ENGINE_")
        .skip(1)
        .filter_map(|rest| {
            let end = rest.find('"')?;
            let tail = &rest[..end];
            tail.bytes()
                .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                .then(|| format!("OP_FRAME_ENGINE_{tail}"))
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

#[test]
fn every_setting_the_native_engine_reads_is_pythons_or_a_listed_native_one() {
    let doc = load();
    let mut known: Vec<&str> = doc["python_names"]
        .as_array()
        .expect("python_names")
        .iter()
        .map(|n| n.as_str().expect("name"))
        .collect();
    known.extend(
        doc["differences"]
            .as_object()
            .expect("differences")
            .keys()
            .map(String::as_str),
    );
    known.extend(
        doc["native_only"]
            .as_object()
            .expect("native_only")
            .keys()
            .map(String::as_str),
    );
    let read = names_read_by_the_native_engine();
    assert!(
        read.len() >= 24,
        "only {} names found in settings.rs",
        read.len()
    );
    let unlisted: Vec<&String> = read
        .iter()
        .filter(|n| !known.contains(&n.as_str()))
        .collect();
    assert!(
        unlisted.is_empty(),
        "read by the native engine, not by the Python engine, and not listed as native only \
         with a reason in scripts/codegen/frame_engine_parity_goldens.py: {unlisted:?}"
    );
}
