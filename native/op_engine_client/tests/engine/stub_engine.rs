//! A whole round trip: this crate's client, the engine's own front and stub pipeline in this
//! process (no card, no model server).

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use op_engine_client::{
    Branch, CropVectors, CropsMode, DecodeRoute, DetectorSpec, EncoderModels, EngineClient,
    EngineError, EnginePlan, FaceSpec, IngestPlan, MODEL, OcrSpec, PhotoStatus, ProbeOutcome,
    RenditionSpec,
};
use tonic::Code;

use crate::support::{Counts, Stub, settings};

const PHOTO: &[u8] = b"a plain photo: the stub answers every plain photo alike";

fn detector() -> DetectorSpec {
    DetectorSpec {
        model: "det".into(),
        input_size: 640,
        letterbox_fill: 114,
    }
}

fn encoders() -> EncoderModels {
    EncoderModels {
        image_model: "pe".into(),
        crop_model: "pe_crop".into(),
        crop_size: 224,
    }
}

fn ingest(crop_vectors: CropVectors) -> EnginePlan {
    EnginePlan::ingest(
        &IngestPlan {
            detector: detector(),
            crop_vectors,
        },
        &encoders(),
    )
}

async fn client(stub: &Stub) -> (EngineClient, Arc<Counts>) {
    let counts = Arc::new(Counts::default());
    let client = EngineClient::new(&settings(stub.addr), Some(counts.clone())).unwrap();
    (client, counts)
}

#[tokio::test]
async fn an_ingested_photo_comes_back_with_its_objects_vectors_and_crops() {
    let stub = Stub::start(None, 8).await;
    let (client, _) = client(&stub).await;
    let answer = client
        .process(PHOTO, &ingest(CropVectors::Every), None)
        .await
        .unwrap();
    assert!(answer.ok());
    assert_eq!(answer.status, PhotoStatus::Ok);
    assert_eq!(answer.decode_route, DecodeRoute::Gpu);
    let size = answer.size.unwrap();
    assert_eq!((size.width, size.height), (96, 64));
    let detections = answer.detections.unwrap();
    assert_eq!(detections.classes, [1, 2]);
    assert_eq!(detections.boxes.len(), 2);
    assert_eq!(detections.blur_sums.len(), 3);
    assert_eq!(answer.whole_embedding.unwrap().len(), 1024);
    let crops = answer.crop_embeddings.unwrap();
    assert_eq!((crops.vectors.rows, crops.vectors.dim), (2, 1024));
    assert_eq!(crops.embedded, [true, true]);
    let jpegs = answer.crop_jpegs.unwrap();
    assert_eq!(jpegs.len(), 2);
    assert!(jpegs.iter().all(|j| j.starts_with(&[0xff, 0xd8])));
}

#[tokio::test]
async fn a_photo_with_stored_boxes_skips_the_detector_and_may_have_no_box_at_all() {
    let stub = Stub::start(None, 8).await;
    let (client, _) = client(&stub).await;
    let plan = EnginePlan::stored_boxes_embed(&encoders());
    let boxes = [
        [0.1, 0.1, 0.5, 0.6],
        [0.5, 0.3, 0.9, 0.9],
        [0.0, 0.0, 1.0, 1.0],
    ];
    let answer = client.process(PHOTO, &plan, Some(&boxes)).await.unwrap();
    assert!(answer.ok() && answer.detections.is_none());
    assert_eq!(answer.crop_embeddings.unwrap().vectors.rows, 3);
    let none = client.process(PHOTO, &plan, Some(&[])).await.unwrap();
    assert_eq!(none.crop_embeddings.unwrap().vectors.rows, 0);
}

#[tokio::test]
async fn faces_text_and_renditions_come_back_typed() {
    let stub = Stub::start(None, 8).await;
    let (client, _) = client(&stub).await;
    let plan = EnginePlan {
        branches: BTreeSet::from([Branch::Detect, Branch::Faces, Branch::FaceEmbed, Branch::Ocr]),
        detector: Some(detector()),
        faces: Some(FaceSpec {
            detector: "scrfd".into(),
            input_size: 640,
            recogniser: "arcface".into(),
            batch: 32,
            score_min: 0.5,
            nms: 0.4,
            max_faces: 128,
            coarse: 0,
            tile_side: 0,
            overlap: 0,
        }),
        ocr: Some(serde_json::from_str::<OcrSpec>(
            r#"{"detector": "d", "recogniser": "r", "batch": 16, "max_lines": 128, "det_side": 960,
                "det_limit": "max", "det_max_side": 1920, "det_antialias": false, "thresh": 0.2,
                "box_thresh": 0.45, "unclip_ratio": 1.4, "taps": 1}"#,
        ).unwrap()),
        derivatives: vec![
            serde_json::from_str::<RenditionSpec>(r#"{"name": "thumb", "side": 256}"#).unwrap(),
        ],
        crops: CropsMode::None,
        ..EnginePlan::default()
    };
    let answer = client.process(PHOTO, &plan, None).await.unwrap();
    assert!(answer.ok());
    let faces = answer.faces.unwrap();
    assert_eq!(faces.boxes.len(), 2);
    assert_eq!(faces.landmarks.len(), 2);
    let embeddings = answer.face_embeddings.unwrap();
    assert_eq!((embeddings.vectors.rows, embeddings.vectors.dim), (2, 512));
    let text = answer.text.unwrap();
    assert_eq!(text.texts, ["STOP", "caf\u{e9}"]);
    assert_eq!(text.quads.len(), 2);
    assert_eq!(answer.derivative_jpegs.unwrap().len(), 1);
}

#[tokio::test]
async fn a_photo_the_engine_cannot_read_is_an_answer_with_a_status() {
    let stub = Stub::start(None, 8).await;
    let (client, _) = client(&stub).await;
    let plan = ingest(CropVectors::Every);
    let corrupt = client
        .process(b"stub:status=corrupt", &plan, None)
        .await
        .unwrap();
    assert_eq!(corrupt.status, PhotoStatus::Corrupt);
    assert_eq!(corrupt.decode_route, DecodeRoute::None);
    assert!(!corrupt.ok() && corrupt.detections.is_none());
    let busy = client
        .process(b"stub:status=gpu_memory", &plan, None)
        .await
        .unwrap();
    assert!(busy.status.is_transient());
}

#[tokio::test]
async fn a_status_the_client_does_not_know_fails_the_photo_loudly() {
    let stub = Stub::start(None, 8).await;
    let (client, _) = client(&stub).await;
    let error = client
        .process(
            b"stub:status=quarantined",
            &ingest(CropVectors::Every),
            None,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, EngineError::Answer(_)), "{error}");
    assert!(error.to_string().contains("quarantined"));
}

#[tokio::test]
async fn the_engines_own_failure_for_a_photo_is_not_sent_again() {
    let stub = Stub::start(None, 8).await;
    let (client, counts) = client(&stub).await;
    let error = client
        .process(b"stub:fail=internal", &ingest(CropVectors::Every), None)
        .await
        .unwrap_err();
    match &error {
        EngineError::Failed {
            code,
            attempts,
            message,
        } => {
            assert_eq!(*code, Code::Internal);
            assert_eq!(*attempts, 1);
            assert!(message.starts_with("frame engine: "), "{message}");
        }
        other => panic!("{other}"),
    }
    assert!(!error.transient());
    assert!(counts.retried.lock().unwrap().is_empty());
    assert_eq!(*counts.exhausted.lock().unwrap(), ["INTERNAL"]);
}

#[tokio::test]
async fn an_engine_that_is_down_is_asked_four_times_and_is_unavailable() {
    let stub = Stub::start(None, 8).await;
    let (client, counts) = client(&stub).await;
    for body in [&b"stub:fail=unavailable"[..], b"stub:fail=context_lost"] {
        let error = client
            .process(body, &ingest(CropVectors::Every), None)
            .await
            .unwrap_err();
        assert!(
            matches!(error, EngineError::Unavailable { attempts: 4, .. }),
            "{error}"
        );
        assert!(error.transient() && !error.overloaded());
        assert_eq!(error.code_name(), "UNAVAILABLE");
    }
    assert_eq!(counts.retried.lock().unwrap().len(), 6);
    assert_eq!(counts.exhausted.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn an_overloaded_engine_is_asked_until_the_photos_deadline() {
    let stub = Stub::start(None, 8).await;
    let mut settings = settings(stub.addr);
    // the first four waits are at least 0.75 s and at most 1.55 s: a fifth call is certain
    settings.deadline = Duration::from_secs(2);
    let client = EngineClient::new(&settings, None).unwrap();
    let started = Instant::now();
    let error = client
        .process(b"stub:fail=overload", &ingest(CropVectors::Every), None)
        .await
        .unwrap_err();
    let attempts = error.attempts();
    assert!(error.overloaded() && error.transient(), "{error}");
    assert!(
        attempts > 4,
        "overload is not capped at the retries of an unavailable engine: {attempts}"
    );
    assert!(started.elapsed() >= Duration::from_secs(2));
    assert!(started.elapsed() < Duration::from_secs(6));
}

#[tokio::test]
async fn a_plan_the_engine_refuses_is_rejected_and_not_retried() {
    let stub = Stub::start(None, 8).await;
    let (client, counts) = client(&stub).await;
    let error = client
        .process(PHOTO, &EnginePlan::default(), None)
        .await
        .unwrap_err();
    match &error {
        EngineError::Rejected { code, message } => {
            assert_eq!(*code, Code::InvalidArgument);
            assert!(message.starts_with("empty_plan"), "{message}");
        }
        other => panic!("{other}"),
    }
    assert!(!error.transient());
    assert!(counts.retried.lock().unwrap().is_empty());
}

#[tokio::test]
async fn the_engine_says_whether_it_is_ready() {
    let flag =
        std::env::temp_dir().join(format!("op_engine_client-{}-not-ready", std::process::id()));
    let stub = Stub::start(Some(flag.clone()), 8).await;
    let (client, _) = client(&stub).await;
    let first = client.probe_ready(MODEL).await;
    assert_eq!((first.outcome, first.attempts), (ProbeOutcome::Ready, 1));
    assert!(first.ready());
    std::fs::write(&flag, b"x").unwrap();
    frame_engine::pipeline::Pipeline::check(&*stub.pipeline).await;
    let not_ready = client.probe_ready(MODEL).await;
    assert_eq!(not_ready.outcome, ProbeOutcome::NotReady);
    assert!(!not_ready.ready());
    std::fs::remove_file(&flag).unwrap();
    frame_engine::pipeline::Pipeline::check(&*stub.pipeline).await;
    assert!(client.probe_ready(MODEL).await.ready());
    let other = client.probe_ready("another_model").await;
    assert_eq!(other.outcome, ProbeOutcome::NotReady);
}

#[tokio::test]
async fn an_engine_that_cannot_be_reached_is_undelivered_after_three_tries() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let client = EngineClient::new(&settings(addr), None).unwrap();
    let started = Instant::now();
    let readiness = client.probe_ready(MODEL).await;
    assert_eq!(readiness.outcome, ProbeOutcome::Undelivered);
    assert_eq!(readiness.attempts, 3);
    assert_eq!(readiness.code, "UNAVAILABLE");
    assert!(!readiness.ready());
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "{:?}",
        started.elapsed()
    );
    let error = client
        .process(PHOTO, &ingest(CropVectors::Every), None)
        .await
        .unwrap_err();
    assert!(
        matches!(error, EngineError::Unavailable { attempts: 4, .. }),
        "{error}"
    );
}

#[tokio::test]
async fn photos_beyond_the_in_flight_cap_wait_their_turn() {
    let stub = Stub::start(None, 8).await;
    let mut settings = settings(stub.addr);
    settings.max_in_flight = 1;
    let client = Arc::new(EngineClient::new(&settings, None).unwrap());
    let started = Instant::now();
    let mut tasks = Vec::new();
    for _ in 0..3 {
        let client = Arc::clone(&client);
        tasks.push(tokio::spawn(async move {
            client
                .process(b"stub:hold_ms=250", &ingest(CropVectors::Every), None)
                .await
                .map(|a| a.ok())
        }));
    }
    let mut peak = 0;
    while !tasks.iter().all(|t| t.is_finished()) {
        peak = peak.max(stub.handle.in_flight());
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    for task in tasks {
        assert!(task.await.unwrap().unwrap());
    }
    assert!(
        started.elapsed() >= Duration::from_millis(700),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(peak, 1, "the engine never saw more than the client's cap");
}

#[tokio::test]
async fn a_photo_that_cannot_get_a_permit_before_its_deadline_is_overloaded() {
    let stub = Stub::start(None, 8).await;
    let mut settings = settings(stub.addr);
    settings.max_in_flight = 1;
    settings.deadline = Duration::from_millis(300);
    let client = Arc::new(EngineClient::new(&settings, None).unwrap());
    let holder = Arc::clone(&client);
    let held = tokio::spawn(async move {
        holder
            .process(b"stub:hold_ms=800", &ingest(CropVectors::Every), None)
            .await
    });
    tokio::time::sleep(Duration::from_millis(100)).await;
    let error = client
        .process(PHOTO, &ingest(CropVectors::Every), None)
        .await
        .unwrap_err();
    assert!(error.overloaded(), "{error}");
    assert!(held.await.unwrap().unwrap().ok());
}
