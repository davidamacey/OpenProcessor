//! The route held to the Python route by the cases `scripts/codegen/upload_validation_cases.py`
//! generates: the same multipart request must get the same status and JSON body, the disk guard
//! the same verdict, the batch merge the same response. The expectations are never typed here.

use axum::body::Body;

use crate::common::{self, answer, multipart_request, rig};
use op_config::Env;
use op_ingest::disk_space::{DiskLevel, InsufficientStorage};
use op_ingest::settings::UploadSettings;
use op_ingest::upload::{IngestImageResponse, Refusal, SinkBatch, merge};
use serde_json::{Value, json};

fn cases() -> Value {
    serde_json::from_str(include_str!(
        "../../../../contracts/json/upload_validation_cases.json"
    ))
    .unwrap()
}

/// The body of a case: `size` bytes of `x` for a file, the text for a field.
fn encode(boundary: &str, parts: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for part in parts {
        out.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        let mut head = format!(
            "Content-Disposition: form-data; name=\"{}\"",
            part["name"].as_str().unwrap()
        );
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            out.extend_from_slice(format!("{head}\r\n\r\n{text}\r\n").as_bytes());
        } else if let Some(n) = part.get("text_repeat").and_then(Value::as_u64) {
            out.extend_from_slice(format!("{head}\r\n\r\n").as_bytes());
            out.extend(std::iter::repeat_n(b'x', usize::try_from(n).unwrap()));
            out.extend_from_slice(b"\r\n");
        } else {
            if let Some(filename) = part["filename"].as_str() {
                head.push_str(&format!("; filename=\"{filename}\""));
            }
            out.extend_from_slice(format!("{head}\r\nContent-Type: image/jpeg\r\n\r\n").as_bytes());
            out.extend(std::iter::repeat_n(
                b'x',
                usize::try_from(part["size"].as_u64().unwrap()).unwrap(),
            ));
            out.extend_from_slice(b"\r\n");
        }
    }
    out.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    out
}

fn env_pairs(small: &Value, case: &Value) -> Vec<(String, String)> {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for source in [small, case] {
        for (k, v) in source.as_object().into_iter().flatten() {
            let text = v.as_str().map_or_else(|| v.to_string(), str::to_string);
            pairs.retain(|(key, _)| key != k);
            pairs.push((k.clone(), text));
        }
    }
    pairs
}

/// Python's text after `image_paths is not JSON: ` is its own JSON module's message.
fn comparable(mut body: Value) -> Value {
    if let Some(Value::String(detail)) = body.get_mut("detail")
        && detail.starts_with("image_paths is not JSON: ")
    {
        *detail = "image_paths is not JSON: ".to_string();
    }
    body
}

fn scrub(mut body: Value) -> Value {
    if let Some(map) = body.as_object_mut() {
        map.remove("request_id");
    }
    comparable(body)
}

#[tokio::test]
async fn every_request_case_gets_the_status_and_body_python_gave() {
    let doc = cases();
    let boundary = doc["boundary"].as_str().unwrap();
    let mut ran = 0;
    for case in doc["requests"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let pairs = env_pairs(&doc["small_limits"], &case["env"]);
        let refs: Vec<(&str, &str)> = pairs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        let rig = rig(&refs);
        let (content_type, body, length) = if let Some(parts) = case.get("parts") {
            let body = encode(boundary, parts.as_array().unwrap());
            (
                format!("multipart/form-data; boundary={boundary}"),
                body,
                None,
            )
        } else {
            (
                case["content_type"].as_str().unwrap().to_string(),
                case["body"].as_str().unwrap().as_bytes().to_vec(),
                case.get("content_length").and_then(Value::as_u64),
            )
        };
        let request = multipart_request(&content_type, length, Body::from(body));
        let got = answer(&rig.app, request).await;
        let want = &case["answer"];
        assert_eq!(
            u64::from(got.status.as_u16()),
            want["status"].as_u64().unwrap(),
            "{name}: status, body {}",
            got.body
        );
        assert_eq!(
            scrub(got.body),
            comparable(want["body"].clone()),
            "{name}: body"
        );
        let accepted = rig
            .sink
            .accept_calls
            .load(std::sync::atomic::Ordering::SeqCst);
        assert!(
            accepted == 0 || rig.sink.aborts.lock().unwrap().len() == 1,
            "{name}: {accepted} photos reached the sink and the request was not aborted"
        );
        assert!(
            rig.sink.ends.lock().unwrap().is_empty(),
            "{name}: the sink committed"
        );
        assert!(
            common::landing_files(&rig.landing_root).is_empty(),
            "{name}: landing files left behind"
        );
        ran += 1;
    }
    assert!(ran > 60, "only {ran} cases ran");
}

#[test]
fn the_disk_guard_answers_python_s_verdict_and_body() {
    let doc = cases();
    for case in doc["disk"].as_array().unwrap() {
        let pairs = env_pairs(&json!({}), &case["env"]);
        let settings = UploadSettings::from_env(&Env::from_pairs(pairs.iter().cloned())).unwrap();
        let ratio = case["free_ratio"].as_f64().unwrap();
        let level = match settings.disk.classify(ratio) {
            DiskLevel::Ok => "ok",
            DiskLevel::Low => "low",
            DiskLevel::Critical => "critical",
        };
        assert_eq!(json!(level), case["level"], "{case}");
        let want = &case["answer"];
        if level == "critical" {
            let refusal = Refusal::InsufficientStorage(InsufficientStorage {
                volume: "data",
                free_ratio: ratio,
            });
            assert_eq!(
                u64::from(refusal.status().as_u16()),
                want["status"].as_u64().unwrap()
            );
            assert_eq!(refusal.body("-"), want["body"], "{case}");
        } else {
            assert_eq!(want["status"], 200, "{case}");
        }
    }
}

#[test]
fn thresholds_python_refuses_are_refused_naming_the_variable() {
    let doc = cases();
    for case in doc["disk_threshold_errors"].as_array().unwrap() {
        let pairs = env_pairs(&json!({}), &case["env"]);
        let got = UploadSettings::from_env(&Env::from_pairs(pairs.iter().cloned()));
        assert_eq!(got.is_err(), case["refused"].as_bool().unwrap(), "{case}");
        let message = got.unwrap_err().to_string();
        assert!(message.contains("OP_DISK_"), "{message}");
    }
}

#[test]
fn the_merge_of_refused_photos_and_the_pipeline_s_result_is_python_s() {
    let doc = cases();
    for case in doc["batch_merge"].as_array().unwrap() {
        let early: Vec<IngestImageResponse> =
            serde_json::from_value(case["early"].clone()).unwrap();
        let batch = case.get("batch").filter(|b| !b.is_null()).map(|b| {
            let mut b = b.clone();
            for result in b["results"].as_array_mut().unwrap() {
                // the pipeline calls the count `n_region_queued`; the route serves `n_regions`
                let queued = result["n_region_queued"].take();
                result["n_regions"] = queued;
            }
            serde_json::from_value::<SinkBatch>(b).unwrap()
        });
        let got = serde_json::to_value(merge(early, batch)).unwrap();
        assert_eq!(got, case["response"], "{}", case["name"]);
    }
}
