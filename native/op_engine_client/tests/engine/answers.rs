//! Answers the Python engine wrote, read by the Rust parser and compared with what the Python
//! client read out of the same bytes (`native/op_engine_client/tests/goldens/answers.json`, from
//! `scripts/codegen/frame_engine_answer_goldens.py`).

use base64::Engine as _;
use bytes::Bytes;
use op_engine_client::codec::pack_bytes;
use op_engine_client::proto::{ModelInferResponse, model_infer_response::InferOutputTensor};
use op_engine_client::{Answer, AnswerError, BranchName, BranchState, EnginePlan, PhotoStatus};
use prost::Message;
use serde_json::Value;

use crate::support::hex;

pub fn cases() -> Value {
    let text = include_str!("../goldens/answers.json");
    serde_json::from_str(text).unwrap()
}

pub fn response_of(case: &Value) -> ModelInferResponse {
    let raw = base64::engine::general_purpose::STANDARD
        .decode(case["response_b64"].as_str().unwrap())
        .unwrap();
    ModelInferResponse::decode(raw.as_slice()).unwrap()
}

pub fn case_named(id: &str) -> Value {
    cases()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["id"] == id)
        .unwrap_or_else(|| panic!("no case {id}"))
        .clone()
}

pub fn plan_of(case: &Value) -> EnginePlan {
    serde_json::from_str(case["plan"].as_str().unwrap()).unwrap()
}

fn le_f32(values: &[f32]) -> String {
    hex(&values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>())
}

fn le_i64(values: &[i64]) -> String {
    hex(&values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>())
}

fn le_i32(values: &[i32]) -> String {
    hex(&values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<u8>>())
}

fn flat<const N: usize>(rows: &[[f32; N]]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

fn items(list: &[Bytes]) -> Value {
    Value::Array(list.iter().map(|b| Value::String(hex(b))).collect())
}

/// The golden's hex of output `name`, if the answer has it.
fn bytes_of<'a>(expected: &'a Value, name: &str) -> Option<&'a str> {
    expected["outputs"][name]["hex"].as_str()
}

fn check_shape(expected: &Value, name: &str, shape: &[usize]) {
    let want: Vec<usize> = expected["outputs"][name]["shape"]
        .as_array()
        .unwrap_or_else(|| panic!("{name}: no shape"))
        .iter()
        .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    assert_eq!(want, shape, "{name}");
}

pub fn check_detections(answer: &Answer, expected: &Value) {
    let Some(d) = &answer.detections else {
        assert!(bytes_of(expected, "det_boxes").is_none());
        return;
    };
    assert_eq!(
        le_f32(&flat(&d.boxes)),
        bytes_of(expected, "det_boxes").unwrap()
    );
    check_shape(expected, "det_boxes", &[d.boxes.len(), 4]);
    assert_eq!(le_f32(&d.scores), bytes_of(expected, "det_scores").unwrap());
    assert_eq!(
        le_i32(&d.classes),
        bytes_of(expected, "det_classes").unwrap()
    );
    let letterbox: Vec<u8> = [d.letterbox.scale, d.letterbox.pad_w, d.letterbox.pad_h]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(hex(&letterbox), bytes_of(expected, "letterbox").unwrap());
    let sums: Vec<i64> = d.blur_sums.iter().flatten().copied().collect();
    assert_eq!(le_i64(&sums), bytes_of(expected, "blur_sums").unwrap());
}

pub fn check_vectors(answer: &Answer, expected: &Value) {
    if let Some(whole) = &answer.whole_embedding {
        assert_eq!(
            le_f32(whole),
            bytes_of(expected, "whole_embedding").unwrap()
        );
    }
    if let Some(crops) = &answer.crop_embeddings {
        assert_eq!(
            le_f32(&crops.vectors.data),
            bytes_of(expected, "crop_embeddings").unwrap()
        );
        check_shape(
            expected,
            "crop_embeddings",
            &[crops.vectors.rows, crops.vectors.dim],
        );
        let flags: Vec<u8> = crops.embedded.iter().map(|b| u8::from(*b)).collect();
        assert_eq!(hex(&flags), bytes_of(expected, "crop_embedded").unwrap());
    }
    if let Some(jpegs) = &answer.crop_jpegs {
        assert_eq!(items(jpegs), expected["outputs"]["crop_jpegs"]["items_hex"]);
    }
    if let Some(jpegs) = &answer.derivative_jpegs {
        assert_eq!(
            items(jpegs),
            expected["outputs"]["derivative_jpegs"]["items_hex"]
        );
    }
}

fn check_faces(answer: &Answer, expected: &Value) {
    if let Some(f) = &answer.faces {
        assert_eq!(
            le_f32(&flat(&f.boxes)),
            bytes_of(expected, "face_boxes").unwrap()
        );
        assert_eq!(
            le_f32(&f.scores),
            bytes_of(expected, "face_scores").unwrap()
        );
        assert_eq!(
            le_f32(&flat(&f.landmarks)),
            bytes_of(expected, "face_landmarks").unwrap()
        );
        let sums: Vec<i64> = f.blur_sums.iter().flatten().copied().collect();
        assert_eq!(le_i64(&sums), bytes_of(expected, "face_blur_sums").unwrap());
        assert_eq!(
            le_i32(&[f.overflow]),
            bytes_of(expected, "face_overflow").unwrap()
        );
    }
    if let Some(e) = &answer.face_embeddings {
        assert_eq!(
            le_f32(&e.vectors.data),
            bytes_of(expected, "face_embeddings").unwrap()
        );
        check_shape(
            expected,
            "face_embeddings",
            &[e.vectors.rows, e.vectors.dim],
        );
        let flags: Vec<u8> = e.embedded.iter().map(|b| u8::from(*b)).collect();
        assert_eq!(hex(&flags), bytes_of(expected, "face_embedded").unwrap());
    }
}

fn check_text(answer: &Answer, expected: &Value) {
    let Some(t) = &answer.text else { return };
    assert_eq!(
        le_f32(&flat(&t.quads)),
        bytes_of(expected, "ocr_quads").unwrap()
    );
    let texts: Vec<String> = t.texts.iter().map(|s| hex(s.as_bytes())).collect();
    let want: Vec<String> = expected["outputs"]["ocr_texts"]["items_hex"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(texts, want);
    assert_eq!(le_f32(&t.scores), bytes_of(expected, "ocr_scores").unwrap());
    assert_eq!(
        le_f32(&t.char_min),
        bytes_of(expected, "ocr_char_min").unwrap()
    );
    assert_eq!(
        le_f32(&t.det_scores),
        bytes_of(expected, "ocr_det_scores").unwrap()
    );
    assert_eq!(
        le_i32(&[t.overflow]),
        bytes_of(expected, "ocr_overflow").unwrap()
    );
}

pub fn check_faces_and_text(answer: &Answer, expected: &Value) {
    check_faces(answer, expected);
    check_text(answer, expected);
}

fn state_name(state: BranchState) -> (&'static str, &'static str) {
    match state {
        BranchState::Ok => ("ok", ""),
        BranchState::NotRun => ("not_run", ""),
        BranchState::Error(reason) => ("error", reason.as_str()),
    }
}

#[test]
fn every_python_answer_reads_the_same_in_rust() {
    let all = cases();
    let cases = all["cases"].as_array().unwrap();
    assert!(cases.len() >= 15);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let expected = &case["expected"];
        let answer = Answer::parse(&response_of(case), &plan_of(case))
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(answer.status.as_str(), expected["status"], "{id}");
        assert_eq!(
            answer.decode_route.as_str(),
            expected["decode_route"],
            "{id}"
        );
        assert_eq!(answer.ok(), expected["ok"].as_bool().unwrap(), "{id}");
        for (name, state) in &answer.branches {
            let [py_state, py_reason] = [
                expected["branches"][name.name()][0].as_str().unwrap(),
                expected["branches"][name.name()][1].as_str().unwrap(),
            ];
            assert_eq!(
                state_name(*state),
                (py_state, py_reason),
                "{id} {}",
                name.name()
            );
        }
        assert_eq!(
            answer.branches.len(),
            expected["branches"].as_object().unwrap().len(),
            "{id}"
        );
        if let Some(size) = answer.size {
            assert_eq!(
                le_i32(&[
                    i32::try_from(size.width).unwrap(),
                    i32::try_from(size.height).unwrap()
                ]),
                bytes_of(expected, "size").unwrap(),
                "{id}"
            );
        }
        check_detections(&answer, expected);
        check_vectors(&answer, expected);
        check_faces_and_text(&answer, expected);
    }
}

#[test]
fn a_photo_the_engine_did_not_decode_has_a_status_and_no_tensors() {
    for (id, status) in [
        ("not_decoded_corrupt", PhotoStatus::Corrupt),
        ("not_decoded_gpu_memory", PhotoStatus::GpuMemory),
    ] {
        let case = case_named(id);
        let answer = Answer::parse(&response_of(&case), &plan_of(&case)).unwrap();
        assert_eq!(answer.status, status);
        assert!(!answer.ok());
        assert!(answer.detections.is_none() && answer.size.is_none());
        assert_eq!(status.is_transient(), status == PhotoStatus::GpuMemory);
    }
}

#[test]
fn a_branch_that_did_not_finish_is_named() {
    let case = case_named("crop_encoder_failed");
    let answer = Answer::parse(&response_of(&case), &plan_of(&case)).unwrap();
    let (name, state) = answer.first_unfinished_branch().unwrap();
    assert_eq!(name.name(), "embed_crops");
    assert!(matches!(
        state,
        BranchState::Error(op_engine_client::BranchError::ModelError)
    ));
    assert!(answer.crop_embeddings.is_none() && answer.detections.is_some());
}

pub fn response_with(status: &str, route: &str, branches: &str) -> ModelInferResponse {
    let tensor = |name: &str| InferOutputTensor {
        name: name.to_owned(),
        datatype: "BYTES".to_owned(),
        shape: vec![1],
        ..Default::default()
    };
    ModelInferResponse {
        outputs: vec![
            tensor("status"),
            tensor("decode_route"),
            tensor("branch_status"),
        ],
        raw_output_contents: vec![
            pack_bytes(status.as_bytes()),
            pack_bytes(route.as_bytes()),
            pack_bytes(branches.as_bytes()),
        ],
        ..Default::default()
    }
}

fn detect_plan() -> EnginePlan {
    serde_json::from_str(
        r#"{"version": 1, "branches": ["detect"], "detector": {"model": "d", "input_size": 640, "letterbox_fill": 114}}"#,
    )
    .unwrap()
}

#[test]
fn a_status_a_newer_engine_adds_is_an_error_not_a_default() {
    let response = response_with(
        "quarantined",
        "none",
        r#"{"detect": {"status": "error", "reason": "image_failed"}}"#,
    );
    assert_eq!(
        Answer::parse(&response, &detect_plan()).unwrap_err(),
        AnswerError::UnknownStatus("quarantined".into())
    );
}

#[test]
fn a_route_a_branch_or_a_reason_that_is_not_in_the_contract_is_an_error() {
    let plan = detect_plan();
    let fine = r#"{"detect": {"status": "error", "reason": "image_failed"}}"#;
    let route = response_with("corrupt", "gpu_v2", fine);
    assert!(matches!(
        Answer::parse(&route, &plan),
        Err(AnswerError::UnknownDecodeRoute(_))
    ));
    let branch = response_with("corrupt", "none", r#"{"detect2": {"status": "ok"}}"#);
    assert!(matches!(
        Answer::parse(&branch, &plan),
        Err(AnswerError::UnknownBranch(_))
    ));
    let reason = response_with(
        "corrupt",
        "none",
        r#"{"detect": {"status": "error", "reason": "oom"}}"#,
    );
    assert!(matches!(
        Answer::parse(&reason, &plan),
        Err(AnswerError::UnknownBranchState(_))
    ));
    let state = response_with("corrupt", "none", r#"{"detect": {"status": "paused"}}"#);
    assert!(matches!(
        Answer::parse(&state, &plan),
        Err(AnswerError::UnknownBranchState(_))
    ));
    let object = response_with("corrupt", "none", "[1]");
    assert!(matches!(
        Answer::parse(&object, &plan),
        Err(AnswerError::BranchStatus(_))
    ));
}

#[test]
fn a_decoded_answer_must_report_exactly_the_plans_branches() {
    let plan = detect_plan();
    let none = response_with("ok", "gpu", "{}");
    assert!(matches!(
        Answer::parse(&none, &plan),
        Err(AnswerError::BranchStatus(_))
    ));
    let extra = response_with(
        "ok",
        "gpu",
        r#"{"detect": {"status": "ok"}, "ocr": {"status": "ok"}}"#,
    );
    assert!(matches!(
        Answer::parse(&extra, &plan),
        Err(AnswerError::BranchStatus(_))
    ));
}

#[test]
fn an_answer_whose_branches_all_ran_must_carry_their_outputs() {
    let plan = detect_plan();
    let response = response_with("ok", "gpu", r#"{"detect": {"status": "ok"}}"#);
    // no size, no tensors
    assert_eq!(
        Answer::parse(&response, &plan).unwrap_err(),
        AnswerError::Missing("size")
    );
}

#[test]
fn an_output_of_the_wrong_datatype_or_size_is_refused() {
    let case = case_named("frame_only");
    let plan = plan_of(&case);
    let mut wrong_type = response_of(&case);
    let at = wrong_type
        .outputs
        .iter()
        .position(|o| o.name == "det_scores")
        .unwrap();
    wrong_type.outputs[at].datatype = "FP64".into();
    assert!(matches!(
        Answer::parse(&wrong_type, &plan),
        Err(AnswerError::Datatype { .. })
    ));
    let mut cut = response_of(&case);
    let at = cut
        .outputs
        .iter()
        .position(|o| o.name == "det_boxes")
        .unwrap();
    let raw = cut.raw_output_contents[at].clone();
    cut.raw_output_contents[at] = raw.slice(..raw.len() - 1);
    assert!(matches!(
        Answer::parse(&cut, &plan),
        Err(AnswerError::Shape { .. })
    ));
    let mut missing_raw = response_of(&case);
    missing_raw.raw_output_contents.pop();
    assert!(matches!(
        Answer::parse(&missing_raw, &plan),
        Err(AnswerError::RawContents(_))
    ));
}

#[test]
fn a_group_with_some_of_its_outputs_is_refused() {
    let case = case_named("frame_only");
    let plan = plan_of(&case);
    let mut partial = response_of(&case);
    let at = partial
        .outputs
        .iter()
        .position(|o| o.name == "letterbox")
        .unwrap();
    partial.outputs.remove(at);
    partial.raw_output_contents.remove(at);
    assert_eq!(
        Answer::parse(&partial, &plan).unwrap_err(),
        AnswerError::Missing("letterbox")
    );
}

#[test]
fn the_photos_branch_names_sort_in_vocabulary_order_with_renditions_last() {
    let case = case_named("detect_then_conditional_branches_not_run");
    let answer = Answer::parse(&response_of(&case), &plan_of(&case)).unwrap();
    let names: Vec<&str> = answer
        .branches
        .keys()
        .map(|n: &BranchName| n.name())
        .collect();
    assert_eq!(names, ["detect", "faces", "ocr"]);
}
