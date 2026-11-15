//! The client against an in-process engine that answers protocol version 1 with canned tensors:
//! what goes on the wire, how each kind of answer is classified, and a whole measured row.

#![allow(
    clippy::unwrap_used,
    reason = "the in-process engine's handlers and helpers are test code, outside #[test] functions"
)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use frame_engine_loadgen::client::{self, EngineClient, MODEL, pack_bytes, unpack_bytes};
use frame_engine_loadgen::exchange::{Exchange, Outcome};
use frame_engine_loadgen::list::PhotoList;
use frame_engine_loadgen::measure::{Setup, measure};
use frame_engine_loadgen::plan::PlanFile;
use frame_engine_loadgen::proto::grpc_inference_service_server::{
    GrpcInferenceService, GrpcInferenceServiceServer,
};
use frame_engine_loadgen::proto::model_infer_response::InferOutputTensor;
use frame_engine_loadgen::proto::{
    InferStatistics, ModelInferRequest, ModelInferResponse, ModelReadyRequest, ModelReadyResponse,
    ModelStatistics, ModelStatisticsRequest, ModelStatisticsResponse, ServerLiveRequest,
    ServerLiveResponse, ServerReadyRequest, ServerReadyResponse, StatisticDuration,
};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

const PLAN: &str =
    r#"{"version": 1, "branches": ["detect"], "detector": {"model": "d", "input_size": 640}}"#;

/// Answers photo `n` by its bytes: `corrupt` for a photo that starts with `C`, a gRPC refusal for
/// one that starts with `R`, a failed branch for `B`, a wrong echo for `E`.
#[derive(Default)]
struct FakeEngine {
    requests: Mutex<Vec<ModelInferRequest>>,
    infers: AtomicU64,
}

fn bytes_output(name: &str, item: &str) -> (InferOutputTensor, Bytes) {
    (
        InferOutputTensor {
            name: name.to_string(),
            datatype: "BYTES".into(),
            shape: vec![1],
            ..Default::default()
        },
        pack_bytes(item.as_bytes()),
    )
}

#[tonic::async_trait]
impl GrpcInferenceService for Shared {
    async fn server_live(
        &self,
        _: Request<ServerLiveRequest>,
    ) -> Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _: Request<ServerReadyRequest>,
    ) -> Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: true }))
    }

    async fn model_ready(
        &self,
        r: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        Ok(Response::new(ModelReadyResponse {
            ready: r.into_inner().name == MODEL,
        }))
    }

    async fn model_infer(
        &self,
        r: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let request = r.into_inner();
        self.0.infers.fetch_add(1, Ordering::Relaxed);
        let image = unpack_bytes(&request.raw_input_contents[0]).unwrap()[0].to_vec();
        let id = request.id.clone();
        self.0.requests.lock().unwrap().push(request);
        let (status, branches, echo) = match image.first() {
            Some(b'R') => return Err(Status::resource_exhausted("full")),
            Some(b'C') => (
                "corrupt",
                r#"{"detect": {"status": "error", "reason": "image_failed"}}"#,
                id,
            ),
            Some(b'B') => (
                "ok",
                r#"{"detect": {"status": "error", "reason": "model_error"}}"#,
                id,
            ),
            Some(b'E') => ("ok", r#"{"detect": {"status": "ok"}}"#, "wrong".to_string()),
            _ => ("ok", r#"{"detect": {"status": "ok"}}"#, id),
        };
        let outputs = [
            bytes_output("status", status),
            bytes_output("decode_route", "gpu"),
            bytes_output("branch_status", branches),
        ];
        Ok(Response::new(ModelInferResponse {
            id: echo,
            model_name: MODEL.into(),
            model_version: "1".into(),
            outputs: outputs.iter().map(|(o, _)| o.clone()).collect(),
            raw_output_contents: outputs.into_iter().map(|(_, b)| b).collect(),
            ..Default::default()
        }))
    }

    async fn model_statistics(
        &self,
        _: Request<ModelStatisticsRequest>,
    ) -> Result<Response<ModelStatisticsResponse>, Status> {
        // 4 items per execution, 2 ms of queue per item
        let n = self.0.infers.load(Ordering::Relaxed);
        let step = |count: u64, ns: u64| Some(StatisticDuration { count, ns });
        Ok(Response::new(ModelStatisticsResponse {
            model_stats: vec![ModelStatistics {
                name: "det".into(),
                inference_count: n,
                execution_count: n / 4,
                inference_stats: Some(InferStatistics {
                    queue: step(n, n * 2_000_000),
                    compute_infer: step(n / 4, n / 4 * 8_000_000),
                    ..Default::default()
                }),
                ..Default::default()
            }],
        }))
    }
}

/// The service type: a handle to the engine, since the trait is foreign to this test crate.
struct Shared(Arc<FakeEngine>);

async fn serve(engine: Arc<FakeEngine>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap().to_string();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(GrpcInferenceServiceServer::new(Shared(engine)))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    address
}

fn photos() -> Vec<Bytes> {
    [
        "ok-photo",
        "Rejected",
        "Corrupt",
        "Branch",
        "Echo",
        "second-ok",
    ]
    .iter()
    .map(|s| Bytes::from(s.as_bytes().to_vec()))
    .collect()
}

async fn client_for(address: &str, photos: &[Bytes]) -> EngineClient {
    let clients = client::connect(address, 2, Duration::from_secs(5))
        .await
        .unwrap();
    EngineClient::new(clients, photos, PLAN, Duration::from_secs(5))
}

#[tokio::test]
async fn the_request_is_protocol_version_1_exactly() {
    let engine = Arc::new(FakeEngine::default());
    let address = serve(engine.clone()).await;
    let photos = photos();
    let c = client_for(&address, &photos).await;
    assert_eq!(
        c.exchange(0).await,
        Outcome::Ok {
            route: "gpu".into(),
            bytes: 45
        }
    );
    let sent = engine.requests.lock().unwrap().remove(0);
    assert_eq!(sent.model_name, MODEL);
    assert_eq!(sent.id, "0");
    assert!(
        sent.outputs.is_empty(),
        "no outputs filter, as the API sends"
    );
    let names: Vec<_> = sent
        .inputs
        .iter()
        .map(|i| (i.name.as_str(), i.datatype.as_str(), i.shape.clone()))
        .collect();
    assert_eq!(
        names,
        [("image", "BYTES", vec![1]), ("plan", "BYTES", vec![1])]
    );
    assert_eq!(
        unpack_bytes(&sent.raw_input_contents[0]).unwrap(),
        [b"ok-photo".as_slice()]
    );
    assert_eq!(
        unpack_bytes(&sent.raw_input_contents[1]).unwrap(),
        [PLAN.as_bytes()]
    );
}

#[tokio::test]
async fn each_kind_of_answer_has_its_own_failure_key() {
    let engine = Arc::new(FakeEngine::default());
    let address = serve(engine).await;
    let photos = photos();
    let c = client_for(&address, &photos).await;
    let key = |o: Outcome| match o {
        Outcome::Failed { key } => key,
        ok => panic!("expected a failure, got {ok:?}"),
    };
    assert_eq!(key(c.exchange(1).await), "grpc:RESOURCE_EXHAUSTED");
    assert_eq!(key(c.exchange(2).await), "status:corrupt");
    assert_eq!(key(c.exchange(3).await), "branch:detect:model_error");
    assert_eq!(key(c.exchange(4).await), "protocol:id_echo:wrong");
}

#[tokio::test]
async fn readiness_is_server_and_model() {
    let address = serve(Arc::new(FakeEngine::default())).await;
    let mut clients = client::connect(&address, 1, Duration::from_secs(5))
        .await
        .unwrap();
    assert!(
        client::is_ready(&mut clients[0], Duration::from_secs(5))
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn a_whole_row_is_measured_through_the_client_and_the_statistics() {
    let engine = Arc::new(FakeEngine::default());
    let address = serve(engine.clone()).await;
    let dir = std::env::temp_dir().join(format!("op-fe-loadgen-row-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let files: Vec<_> = ["ok-photo", "second-ok"]
        .iter()
        .map(|n| {
            let p = dir.join(n);
            std::fs::write(&p, n.as_bytes()).unwrap();
            p.display().to_string()
        })
        .collect();
    std::fs::write(dir.join("list.txt"), files.join("\n")).unwrap();
    let list = PhotoList::load(&dir.join("list.txt")).unwrap();
    let c = Arc::new(client_for(&address, &list.photos).await);
    let mut stats = client::connect(&address, 1, Duration::from_secs(5))
        .await
        .unwrap();
    let setup = Setup {
        mode: "engine",
        label: "test".into(),
        command: vec!["op-fe-loadgen".into()],
        plan: PlanFile::parse(PLAN).unwrap(),
        plan_path: "plan.json".into(),
        in_flight: 4,
        warmup: Duration::from_millis(200),
        window: Duration::from_millis(600),
        cgroup: None,
        stats: Some((address.clone(), stats.remove(0))),
        gpu: None,
        engine: Some(frame_engine_loadgen::row::EngineEvidence {
            address,
            channels: 2,
            timeout_s: 5.0,
        }),
        z: None,
        list,
    };
    let row = measure(setup, c).await;
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(row.failed, 0, "{:?}", row.failures_by_status);
    assert!(row.photos > 20, "{} photos", row.photos);
    assert_eq!(
        row.decode_routes.get("gpu").copied(),
        Some(row.photos as u64)
    );
    assert!(row.photos_per_s > 30.0);
    let det = row.model_server.unwrap().models.remove("det").unwrap();
    // the fake counts 4 items per execution and 2 ms of queue per item
    assert!((det.mean_batch - 4.0).abs() < 0.5, "{}", det.mean_batch);
    assert!((det.queue_ms_per_request - 2.0).abs() < 1e-6);
}
