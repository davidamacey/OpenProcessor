//! What the suites share: the repository's files, a scripted gRPC server, the engine's own stub.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use frame_engine::front::{Front, FrontConfig, FrontHandle, ServeConfig, serve};
use frame_engine::metrics::Metrics;
use frame_engine::stub::StubPipeline;
use op_engine_client::proto::grpc_inference_service_server::{
    GrpcInferenceService, GrpcInferenceServiceServer,
};
use op_engine_client::proto::{
    ModelInferRequest, ModelInferResponse, ModelReadyRequest, ModelReadyResponse,
    RepositoryModelLoadRequest, RepositoryModelLoadResponse,
};
use op_engine_client::{CallObserver, EngineSettings, ProbeOutcome};
use serde_json::Value;
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{Request, Response, Status};

/// A file of the repository, by its path from the root.
pub fn repo_file(path: &str) -> String {
    let full = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(path);
    std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("{}: {e}", full.display()))
}

pub fn repo_json(path: &str) -> Value {
    serde_json::from_str(&repo_file(path)).expect("the file is JSON")
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// What a scripted call does.
pub enum Reply {
    Answer(ModelInferResponse),
    Fail(Status),
}

/// A gRPC server that answers `ModelInfer` from a script (then from `fallback`), holds each call
/// for `hold`, and counts them; `ModelReady` and `RepositoryModelLoad` answer from a table.
#[derive(Default)]
pub struct Fake {
    script: Mutex<VecDeque<Reply>>,
    fallback: Mutex<Option<ModelInferResponse>>,
    pub hold: Mutex<Duration>,
    pub calls: AtomicU32,
    current: AtomicU32,
    pub peak: AtomicU32,
    pub ready: Mutex<HashMap<String, bool>>,
    pub loads: Mutex<Vec<String>>,
    pub load_fails: Mutex<Option<Status>>,
    pub probes: AtomicU32,
    pub requests: Mutex<Vec<ModelInferRequest>>,
}

impl Fake {
    pub fn then(&self, reply: Reply) {
        self.script.lock().unwrap().push_back(reply);
    }

    pub fn answering(&self, answer: ModelInferResponse) {
        *self.fallback.lock().unwrap() = Some(answer);
    }
}

/// The trait is foreign to this test crate: serve a handle to the fake.
pub struct Shared(pub Arc<Fake>);

#[tonic::async_trait]
impl GrpcInferenceService for Shared {
    async fn model_ready(
        &self,
        request: Request<ModelReadyRequest>,
    ) -> Result<Response<ModelReadyResponse>, Status> {
        self.0.probes.fetch_add(1, Ordering::Relaxed);
        let name = request.into_inner().name;
        let ready = self
            .0
            .ready
            .lock()
            .unwrap()
            .get(&name)
            .copied()
            .unwrap_or(false);
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn model_infer(
        &self,
        request: Request<ModelInferRequest>,
    ) -> Result<Response<ModelInferResponse>, Status> {
        let fake = &self.0;
        fake.calls.fetch_add(1, Ordering::SeqCst);
        fake.requests.lock().unwrap().push(request.into_inner());
        let now = fake.current.fetch_add(1, Ordering::SeqCst) + 1;
        fake.peak.fetch_max(now, Ordering::SeqCst);
        let hold = *fake.hold.lock().unwrap();
        tokio::time::sleep(hold).await;
        fake.current.fetch_sub(1, Ordering::SeqCst);
        let scripted = fake.script.lock().unwrap().pop_front();
        match scripted {
            Some(Reply::Answer(answer)) => Ok(Response::new(answer)),
            Some(Reply::Fail(status)) => Err(status),
            None => fake
                .fallback
                .lock()
                .unwrap()
                .clone()
                .map(Response::new)
                .ok_or_else(|| Status::internal("frame engine: nothing scripted")),
        }
    }

    async fn repository_model_load(
        &self,
        request: Request<RepositoryModelLoadRequest>,
    ) -> Result<Response<RepositoryModelLoadResponse>, Status> {
        let model = request.into_inner().model_name;
        if let Some(status) = self.0.load_fails.lock().unwrap().clone() {
            return Err(status);
        }
        self.0.ready.lock().unwrap().insert(model.clone(), true);
        self.0.loads.lock().unwrap().push(model);
        Ok(Response::new(RepositoryModelLoadResponse {}))
    }
}

/// A fake served on a free local port.
pub async fn serve_fake(fake: Arc<Fake>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(GrpcInferenceServiceServer::new(Shared(fake)))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    address
}

/// The engine's own front over its own stub pipeline, in this process: no card, no model server.
pub struct Stub {
    pub addr: SocketAddr,
    pub pipeline: Arc<StubPipeline>,
    pub handle: FrontHandle<StubPipeline>,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Stub {
    pub async fn start(not_ready_file: Option<PathBuf>, max_in_flight: u32) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let pipeline = Arc::new(StubPipeline::new(not_ready_file));
        let config = FrontConfig {
            max_in_flight,
            request_deadline_seconds: 30.0,
        };
        let (front, handle) = Front::new(Arc::clone(&pipeline), Metrics::new(), config);
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let serve_config = ServeConfig {
            max_message_mib: 64,
            drain: Duration::from_secs(1),
        };
        let served = handle.clone();
        tokio::spawn(async move {
            serve(listener, front, served, serve_config, async {
                let _ = stopped.await;
            })
            .await
            .unwrap();
        });
        Self {
            addr,
            pipeline,
            handle,
            stop: Some(stop),
        }
    }
}

impl Drop for Stub {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
    }
}

/// Settings for a test: short deadlines so a failure path takes a second, not two minutes.
pub fn settings(addr: SocketAddr) -> EngineSettings {
    EngineSettings {
        url: addr.to_string(),
        channels: 2,
        max_in_flight: 4,
        call_timeout: Duration::from_secs(5),
        deadline: Duration::from_secs(2),
    }
}

/// Counts what the client reports.
#[derive(Default)]
pub struct Counts {
    pub retried: Mutex<Vec<String>>,
    pub exhausted: Mutex<Vec<String>>,
    pub probed: Mutex<Vec<ProbeOutcome>>,
}

impl CallObserver for Counts {
    fn retried(&self, code: &str) {
        self.retried.lock().unwrap().push(code.to_owned());
    }
    fn exhausted(&self, code: &str) {
        self.exhausted.lock().unwrap().push(code.to_owned());
    }
    fn probed(&self, outcome: ProbeOutcome) {
        self.probed.lock().unwrap().push(outcome);
    }
}
