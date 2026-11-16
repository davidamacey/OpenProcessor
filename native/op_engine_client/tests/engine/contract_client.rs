//! The rules of the client against the contract file
//! (`contracts/json/frame_engine_v1.json`, generated from the Python wire module and front).

use std::time::Duration;

use op_engine_client::retry::{Class, ENGINE_ERROR_PREFIX, classify};
use op_engine_client::settings::KEEPALIVE_INTERVAL;
use op_engine_client::{CROP_FILL, DEFAULT_RENDITION_QUALITY, MODEL, ProtocolVersion};
use tonic::{Code, Status};

use crate::support::{repo_file, repo_json};

const CONTRACT: &str = "contracts/json/frame_engine_v1.json";

#[test]
fn the_protocol_constants_are_the_contracts() {
    let contract = repo_json(CONTRACT);
    assert_eq!(contract["model"]["name"], MODEL);
    assert_eq!(
        contract["model"]["protocol_version"],
        ProtocolVersion::NUMBER
    );
    assert_eq!(
        contract["model"]["engine_error_prefix"],
        ENGINE_ERROR_PREFIX
    );
    assert_eq!(contract["plan"]["crop_fill_default"], CROP_FILL);
    assert_eq!(
        contract["derivatives"]["default_quality"],
        DEFAULT_RENDITION_QUALITY
    );
    assert_eq!(contract["boxes_zero_rows_allowed"], true);
}

fn class_of(code: Code, message: &str) -> Class {
    classify(&Status::new(code, message))
}

#[test]
fn each_status_the_engine_answers_is_retried_or_not_as_the_python_client_does() {
    let contract = repo_json(CONTRACT);
    // the table of plan 17 section 1.1: overload retries until the deadline, "unavailable" three
    // times, the engine's own INTERNAL and DEADLINE_EXCEEDED never, a lost call three times
    for failure in contract["failure_statuses"].as_array().unwrap() {
        let status = failure["status"].as_str().unwrap();
        let class = match status {
            "RESOURCE_EXHAUSTED" => class_of(Code::ResourceExhausted, "busy"),
            "UNAVAILABLE" => class_of(Code::Unavailable, "not ready"),
            "DEADLINE_EXCEEDED" => class_of(Code::DeadlineExceeded, "late"),
            "INTERNAL" => class_of(Code::Internal, &format!("{ENGINE_ERROR_PREFIX}boom")),
            other => panic!("a status the client has no rule for: {other}"),
        };
        let want = match status {
            "RESOURCE_EXHAUSTED" => Class::Overload,
            "UNAVAILABLE" => Class::Transient,
            _ => Class::Final,
        };
        assert_eq!(class, want, "{}", failure["failure"]);
    }
    assert_eq!(class_of(Code::Internal, "Stream removed"), Class::Transient);
    assert_eq!(class_of(Code::Internal, ""), Class::Transient);
    for check in contract["check_order"].as_array().unwrap() {
        let want = match check["status"].as_str().unwrap() {
            "NOT_FOUND" | "INVALID_ARGUMENT" => Class::Final,
            "UNAVAILABLE" => Class::Transient,
            "RESOURCE_EXHAUSTED" => Class::Overload,
            other => panic!("{other}"),
        };
        let code = match check["status"].as_str().unwrap() {
            "NOT_FOUND" => Code::NotFound,
            "INVALID_ARGUMENT" => Code::InvalidArgument,
            "UNAVAILABLE" => Code::Unavailable,
            _ => Code::ResourceExhausted,
        };
        assert_eq!(class_of(code, "x"), want, "{}", check["check"]);
    }
}

#[test]
fn the_keepalive_pings_stay_inside_what_the_engine_and_the_model_server_accept() {
    let contract = repo_json(CONTRACT);
    let policy = &contract["ping_policy"];
    let every = u64::try_from(KEEPALIVE_INTERVAL.as_millis()).unwrap();
    assert_eq!(policy["client_keepalive_time_ms"], every);
    assert_eq!(policy["client_pings_without_calls"], true);
    assert_eq!(policy["server_permits_pings_without_calls"], true);
    assert!(policy["server_min_ping_interval_ms"].as_u64().unwrap() <= every);
    // Triton: pings without calls permitted, and none refused for being under 20 s apart
    let compose = repo_file("docker-compose.yml");
    assert!(compose.contains("--grpc-keepalive-permit-without-calls=true"));
    let limit = compose
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("- --grpc-http2-min-recv-ping-interval-without-data=")
        })
        .unwrap();
    let limit = Duration::from_millis(limit.trim().parse().unwrap());
    assert!(limit <= KEEPALIVE_INTERVAL, "{limit:?}");
}
