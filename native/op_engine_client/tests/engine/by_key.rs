//! A photo named by a key, from this crate's client to the engine's own front and stub pipeline in
//! this process, over a temporary landing zone: the `source` document the client writes, the
//! answers for a file that is there, gone or changed, and what a refusal is to the client.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use frame_engine::photo::read::KeyedRoots;
use op_engine_client::input::source_document;
use op_engine_client::{
    Branch, CropVectors, DetectorSpec, EncoderModels, EngineClient, EngineError, EnginePlan,
    IngestPlan, PhotoInput, PhotoStatus,
};
use op_local_file::imohash::ImoHash;
use op_local_file::{FileClaim, FileRef};
use tonic::Code;

use crate::support::{Counts, Stub, repo_json, settings};

const PHOTO: &[u8] = b"a plain photo: the stub answers every plain photo alike";

fn plan() -> EnginePlan {
    EnginePlan::ingest(
        &IngestPlan {
            detector: DetectorSpec {
                model: "det".into(),
                input_size: 640,
                letterbox_fill: 114,
            },
            crop_vectors: CropVectors::Every,
        },
        &EncoderModels {
            image_model: "pe".into(),
            crop_model: "pe_crop".into(),
            crop_size: 224,
        },
    )
}

struct Zone {
    _dir: tempfile::TempDir,
    base: std::path::PathBuf,
    stub: Stub,
}

async fn zone() -> Zone {
    let dir = tempfile::tempdir().unwrap();
    let base = fs::canonicalize(dir.path()).unwrap();
    fs::create_dir_all(base.join("landing/7")).unwrap();
    fs::create_dir_all(base.join("source")).unwrap();
    let roots = KeyedRoots::new(
        Some(&base.join("landing")),
        Some(&base.join("source")),
        1 << 20,
    )
    .unwrap();
    Zone {
        stub: Stub::start_reading(None, 8, roots).await,
        base,
        _dir: dir,
    }
}

fn root(zone: &Zone) -> &Path {
    &zone.base
}

fn client(zone: &Zone) -> (EngineClient, Arc<Counts>) {
    let counts = Arc::new(Counts::default());
    let client = EngineClient::new(&settings(zone.stub.addr), Some(counts.clone())).unwrap();
    (client, counts)
}

fn file(key: &str, bytes: &[u8]) -> FileRef {
    FileRef::new(key, FileClaim::of(bytes)).unwrap()
}

#[test]
fn the_source_document_is_the_text_of_every_case_of_the_table() {
    let cases = repo_json("contracts/json/frame_engine_source_cases.json");
    let mut built = 0;
    for case in cases["cases"].as_array().unwrap() {
        let Some(want) = case.get("built") else {
            continue;
        };
        let claim = FileClaim {
            size: case["size"].as_u64().unwrap(),
            imohash: ImoHash::from_hex(case["imohash"].as_str().unwrap()).unwrap(),
        };
        let file = FileRef::new(case["key"].as_str().unwrap(), claim).unwrap();
        assert_eq!(&source_document(&file).unwrap(), want, "{}", case["id"]);
        built += 1;
    }
    assert!(built >= 5, "{built} cases built");
}

#[test]
fn a_key_the_table_refuses_cannot_be_made_into_a_reference_at_all() {
    let cases = repo_json("contracts/json/frame_engine_source_cases.json");
    let claim = FileClaim::of(b"x");
    for case in cases["cases"].as_array().unwrap() {
        let Some(fault) = case["fault"].as_str() else {
            continue;
        };
        if !matches!(fault, "bad_key" | "unknown_root") {
            continue;
        }
        let doc: serde_json::Value = serde_json::from_str(case["text"].as_str().unwrap()).unwrap();
        let made = FileRef::new(doc["key"].as_str().unwrap(), claim);
        assert!(made.is_err(), "{} would be sent", case["id"]);
    }
}

#[tokio::test]
async fn a_photo_named_by_its_key_answers_what_the_same_bytes_answer() {
    let zone = zone().await;
    fs::write(root(&zone).join("landing/7/a.0"), PHOTO).unwrap();
    let (client, counts) = client(&zone);
    let by_key = client
        .process(&file("landing/7/a.0", PHOTO), &plan(), None)
        .await
        .unwrap();
    let by_bytes = client.process(PHOTO, &plan(), None).await.unwrap();
    assert!(by_key.ok());
    assert_eq!(by_key, by_bytes);
    assert!(counts.retried.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_input_is_one_typed_choice() {
    let zone = zone().await;
    fs::write(root(&zone).join("source/b.jpg"), PHOTO).unwrap();
    let (client, _) = client(&zone);
    let key = file("source/b.jpg", PHOTO);
    for input in [PhotoInput::Bytes(PHOTO), PhotoInput::File(&key)] {
        assert!(client.process(input, &plan(), None).await.unwrap().ok());
    }
}

#[tokio::test]
async fn a_file_that_is_gone_or_changed_is_an_answer_about_the_file_and_is_not_sent_again() {
    let zone = zone().await;
    let (client, counts) = client(&zone);
    let gone = client
        .process(&file("source/gone.jpg", PHOTO), &plan(), None)
        .await
        .unwrap();
    assert_eq!(gone.status, PhotoStatus::Missing);
    fs::write(root(&zone).join("source/c.jpg"), b"edited after the claim").unwrap();
    let changed = client
        .process(&file("source/c.jpg", PHOTO), &plan(), None)
        .await
        .unwrap();
    assert_eq!(changed.status, PhotoStatus::Changed);
    for answer in [&gone, &changed] {
        assert!(!answer.ok());
        assert!(answer.status.is_file_problem());
        assert!(!answer.status.is_transient());
        assert_eq!(answer.branches.len(), 3);
    }
    let big = FileRef::new(
        "source/huge.jpg",
        FileClaim {
            size: 2 << 20,
            ..FileClaim::of(b"")
        },
    )
    .unwrap();
    let too_large = client.process(&big, &plan(), None).await.unwrap();
    assert_eq!(too_large.status, PhotoStatus::TooLarge);
    assert!(
        counts.retried.lock().unwrap().is_empty(),
        "no status is sent again"
    );
    assert!(PhotoStatus::ALL.iter().all(|(s, name)| {
        s.is_file_problem() == matches!(*name, "missing" | "changed" | "too_large" | "unreadable")
    }));
    assert_eq!(Branch::ALL.len(), 6);
}

#[tokio::test]
async fn an_engine_with_no_root_for_the_key_rejects_it_and_the_client_does_not_ask_again() {
    let stub = Stub::start(None, 8).await;
    let counts = Arc::new(Counts::default());
    let client = EngineClient::new(&settings(stub.addr), Some(counts.clone())).unwrap();
    let error = client
        .process(&file("landing/7/a.0", PHOTO), &plan(), None)
        .await
        .unwrap_err();
    let EngineError::Rejected { code, message } = &error else {
        panic!("{error:?}");
    };
    assert_eq!(*code, Code::InvalidArgument);
    assert!(
        message.starts_with("root_not_configured: 'landing'"),
        "{message}"
    );
    assert_eq!(error.attempts(), 1);
    assert!(counts.retried.lock().unwrap().is_empty());
}
