//! Owed to the sweep: the Rust client against the native engine on a card, photo by photo, equal
//! to what the Python client read from the same engine
//! (`scripts/bench/frame_engine_client_answers.py` writes the Python side).
//!
//! ```text
//! OP_R1E_URL=127.0.0.1:8001 OP_R1E_LIST=<coco2k list> OP_R1E_EXPECTED=<jsonl of the script> \
//!   cargo test -p op_engine_client --locked --test engine native_engine -- --ignored
//! ```
//!
//! Pass rule: every photo equal, no photo without an answer, and the plan text equal to the one
//! the Python script wrote in its first line.

use std::path::PathBuf;

use op_config::Env;
use op_engine_client::{
    CropVectors, DetectorSpec, EncoderModels, EngineClient, EnginePlan, EngineSettings, IngestPlan,
};
use serde_json::Value;

use crate::answers::{check_detections, check_faces_and_text, check_vectors};

fn read(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

#[ignore = "needs the native engine on a card, a photo list and the Python client's answers for it"]
#[tokio::test]
async fn native_engine_answers_equal_the_python_clients() {
    let env = Env::from_process();
    let url: String = env.require("OP_R1E_URL").unwrap();
    let expected_path: String = env.require("OP_R1E_EXPECTED").unwrap();
    let text = read(&expected_path);
    let mut lines = text.lines();
    let header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    let detector = DetectorSpec {
        model: header["detector"].as_str().unwrap().to_owned(),
        input_size: u32::try_from(header["input_size"].as_u64().unwrap()).unwrap(),
        letterbox_fill: u8::try_from(header["fill"].as_u64().unwrap()).unwrap(),
    };
    let plan = EnginePlan::ingest(
        &IngestPlan {
            detector,
            crop_vectors: CropVectors::Every,
        },
        &EncoderModels::from_env(&env).unwrap(),
    );
    assert_eq!(
        plan.to_json().unwrap(),
        header["plan"].as_str().unwrap(),
        "the plans differ"
    );
    let mut settings = EngineSettings::from_env(&env, 1, 1).unwrap();
    settings.url = url;
    let client = EngineClient::new(&settings, None).unwrap();
    let mut compared = 0usize;
    for line in lines {
        let record: Value = serde_json::from_str(line).unwrap();
        let path = PathBuf::from(record["path"].as_str().unwrap());
        let expected = record
            .get("expected")
            .unwrap_or_else(|| panic!("{}: the Python client had no answer", path.display()));
        let data = std::fs::read(&path).unwrap();
        let answer = client.process(&data, &plan, None).await.unwrap();
        assert_eq!(
            answer.status.as_str(),
            expected["status"],
            "{}",
            path.display()
        );
        assert_eq!(
            answer.decode_route.as_str(),
            expected["decode_route"],
            "{}",
            path.display()
        );
        check_detections(&answer, expected);
        check_vectors(&answer, expected);
        check_faces_and_text(&answer, expected);
        compared += 1;
    }
    assert!(compared > 0, "the expected file holds no photo");
}
