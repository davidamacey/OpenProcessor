//! The capture mapping against `contracts/json/capture_cases.json`, which
//! `scripts/codegen/capture_cases.py` wrote by running the Python `parse_capture` and
//! `apply_location_policy` over hand-written tag sets and over ExifTool outputs recorded from the
//! committed test images.

mod common;

use std::path::Path;

use common::capture_json;
use op_media::capture::{
    EXIFTOOL_TAGS, FILE_ERROR_KEY, LocationPolicy, PARSER_VERSION, apply_location_policy,
    parse_capture, unknown_capture,
};
use serde_json::{Map, Value};

fn cases() -> Value {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/json/capture_cases.json");
    let text = std::fs::read_to_string(path).expect("the cases file is committed");
    serde_json::from_str(&text).expect("the cases file is JSON")
}

fn tags_of(case: &Value) -> Map<String, Value> {
    case["tags"].as_object().expect("tags is an object").clone()
}

#[test]
fn the_constants_are_the_python_ones() {
    let c = cases();
    let tags: Vec<&str> = c["exiftool_tags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_str().unwrap())
        .collect();
    assert_eq!(tags, EXIFTOOL_TAGS);
    assert_eq!(c["file_error_key"], FILE_ERROR_KEY);
    assert_eq!(c["parser_version"], PARSER_VERSION);
}

#[test]
fn every_parse_case_equals_python() {
    let c = cases();
    let rows = c["parse"].as_array().unwrap();
    assert!(rows.len() > 400, "{} cases", rows.len());
    let mut wrong = Vec::new();
    for case in rows {
        let got = capture_json(&parse_capture(&tags_of(case)));
        if got != case["expected"] {
            wrong.push(format!(
                "{}\n  tags: {}\n  rust:   {got}\n  python: {}",
                case["name"], case["tags"], case["expected"]
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} cases differ:\n{}",
        wrong.len(),
        rows.len(),
        wrong.join("\n")
    );
}

#[test]
fn every_location_policy_case_equals_python() {
    let c = cases();
    let rows = c["policy"].as_array().unwrap();
    assert!(rows.len() > 30, "{} cases", rows.len());
    for case in rows {
        let meta = parse_capture(&tags_of(case));
        for (policy, expected) in case["expected"].as_object().unwrap() {
            let policy: LocationPolicy = policy.parse().unwrap();
            let got = capture_json(&apply_location_policy(meta.clone(), policy));
            assert_eq!(&got, expected, "{} under {policy:?}", case["name"]);
        }
    }
}

#[test]
fn the_policy_names_and_the_refused_ones_are_python_s() {
    let c = cases();
    for name in c["policies"].as_array().unwrap() {
        assert!(name.as_str().unwrap().parse::<LocationPolicy>().is_ok());
    }
    for name in c["refused_policies"].as_array().unwrap() {
        let err = name
            .as_str()
            .unwrap()
            .parse::<LocationPolicy>()
            .unwrap_err();
        assert!(
            err.to_string().contains("location_policy must be one of"),
            "{err}"
        );
    }
    assert_eq!(LocationPolicy::default(), LocationPolicy::Store);
}

#[test]
fn an_outage_is_unknown_and_carries_no_parser_stamp() {
    let meta = capture_json(&unknown_capture());
    assert_eq!(meta["exif_status"], "unknown");
    assert_eq!(meta["parser"], Value::Null);
    assert_eq!(meta["has_location"], false);
}

/// Python raises `OverflowError` for these (a UTC time outside years 1 to 9999), so the cases
/// cannot hold them; here the file keeps its other fields and has no UTC time.
#[test]
fn a_utc_time_outside_the_calendar_is_dropped_and_not_an_error() {
    for (local, offset) in [
        ("0001:01:01 00:30:00", "+01:00"),
        ("9999:12:31 23:30:00", "-01:00"),
    ] {
        let tags: Map<String, Value> = serde_json::from_value(serde_json::json!({
            "EXIF:DateTimeOriginal": local,
            "EXIF:OffsetTimeOriginal": offset,
        }))
        .unwrap();
        let meta = parse_capture(&tags);
        assert_eq!(meta.taken_utc, None, "{local} {offset}");
        assert!(meta.taken_local.is_some());
        assert!(meta.offset_minutes.is_some());
    }
}
