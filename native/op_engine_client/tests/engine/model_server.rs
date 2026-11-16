//! The model server's readiness and load calls, and the engine's readiness verdict that uses
//! them (`engine_ready` of `src/services/curation/ingest_gpu_decode.py`).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use op_engine_client::{
    DetectorSpec, EngineClient, ModelServer, ModelServerSettings, ProbeOutcome,
};
use tonic::{Code, Status};

use crate::answers::{case_named, plan_of};
use crate::support::{Counts, Fake, Reply, serve_fake, settings};

fn detector() -> DetectorSpec {
    DetectorSpec {
        model: "det".into(),
        input_size: 640,
        letterbox_fill: 114,
    }
}

async fn server_for(fake: &Arc<Fake>) -> ModelServer {
    let addr = serve_fake(Arc::clone(fake)).await;
    ModelServer::connect_lazy(&ModelServerSettings {
        url: addr.to_string(),
    })
    .unwrap()
}

fn mark_ready(fake: &Fake, names: &[&str]) {
    let mut ready = fake.ready.lock().unwrap();
    for name in names {
        ready.insert((*name).to_owned(), true);
    }
}

#[tokio::test]
async fn the_server_says_whether_a_model_is_ready() {
    let fake = Arc::new(Fake::default());
    mark_ready(&fake, &["pe"]);
    let server = server_for(&fake).await;
    assert!(server.is_model_ready("pe").await);
    assert!(!server.is_model_ready("pe_crop").await);
}

#[tokio::test]
async fn a_server_that_cannot_be_asked_has_no_model_ready() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = listener.local_addr().unwrap().to_string();
    drop(listener);
    let server = ModelServer::connect_lazy(&ModelServerSettings { url }).unwrap();
    assert!(!server.is_model_ready("pe").await);
    let error = server.load_model("pe").await.unwrap_err();
    assert_eq!(error.code, Code::Unavailable);
    assert_eq!(error.model, "pe");
}

#[tokio::test]
async fn a_load_that_the_server_refuses_is_an_error_with_its_status() {
    let fake = Arc::new(Fake::default());
    *fake.load_fails.lock().unwrap() = Some(Status::not_found("no such model in the repository"));
    let server = server_for(&fake).await;
    let error = server.load_model("ghost").await.unwrap_err();
    assert_eq!(error.code, Code::NotFound);
    assert!(error.to_string().contains("ghost"), "{error}");
    assert!(error.message.contains("no such model"));
}

async fn pair() -> (Arc<Fake>, Arc<Fake>, EngineClient, ModelServer, Arc<Counts>) {
    let engine = Arc::new(Fake::default());
    mark_ready(&engine, &["frame_engine"]);
    let models = Arc::new(Fake::default());
    let server = server_for(&models).await;
    let addr = serve_fake(Arc::clone(&engine)).await;
    let counts = Arc::new(Counts::default());
    let client = EngineClient::new(&settings(addr), Some(counts.clone())).unwrap();
    (engine, models, client, server, counts)
}

#[tokio::test]
async fn a_ready_engine_loads_the_models_the_server_does_not_have_and_the_verdict_is_kept() {
    let (engine, models, client, server, counts) = pair().await;
    mark_ready(&models, &["pe_image_encoder"]);
    let names = ["det", "pe_image_encoder", "pe_crop_encoder"];
    assert!(client.ready_for(&detector(), &names, &server).await);
    assert_eq!(*models.loads.lock().unwrap(), ["det", "pe_crop_encoder"]);
    assert_eq!(*counts.probed.lock().unwrap(), [ProbeOutcome::Ready]);
    // within five seconds the engine is not asked again
    assert!(client.ready_for(&detector(), &names, &server).await);
    assert_eq!(engine.probes.load(Ordering::SeqCst), 1);
    // another detector shape is its own verdict
    let other = DetectorSpec {
        input_size: 1280,
        ..detector()
    };
    assert!(client.ready_for(&other, &names, &server).await);
    assert_eq!(engine.probes.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn a_failed_call_drops_the_verdict_so_the_next_photo_asks_the_engine_again() {
    let (engine, models, client, server, _) = pair().await;
    mark_ready(&models, &["det"]);
    assert!(client.ready_for(&detector(), &["det"], &server).await);
    engine.then(Reply::Fail(Status::internal("frame engine: boom")));
    let case = case_named("frame_only");
    assert!(
        client
            .process(b"photo", &plan_of(&case), None)
            .await
            .is_err()
    );
    assert!(client.ready_for(&detector(), &["det"], &server).await);
    assert_eq!(engine.probes.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn an_engine_that_says_not_ready_is_not_ready_and_nothing_is_kept() {
    let (engine, models, client, server, counts) = pair().await;
    engine
        .ready
        .lock()
        .unwrap()
        .insert("frame_engine".into(), false);
    mark_ready(&models, &["det"]);
    assert!(!client.ready_for(&detector(), &["det"], &server).await);
    assert!(!client.ready_for(&detector(), &["det"], &server).await);
    assert_eq!(engine.probes.load(Ordering::SeqCst), 2);
    assert_eq!(
        *counts.probed.lock().unwrap(),
        [ProbeOutcome::NotReady, ProbeOutcome::NotReady]
    );
}

#[tokio::test]
async fn a_model_the_server_cannot_load_keeps_the_engine_from_taking_photos() {
    let (engine, models, client, server, _) = pair().await;
    *models.load_fails.lock().unwrap() = Some(Status::internal("engine build failed"));
    assert!(!client.ready_for(&detector(), &["det"], &server).await);
    assert!(!client.ready_for(&detector(), &["det"], &server).await);
    assert_eq!(
        engine.probes.load(Ordering::SeqCst),
        2,
        "no verdict was kept"
    );
}
