//! The health tick (`_health` in `frame_engine/__main__.py`): every `OP_FRAME_ENGINE_HEALTH_SECONDS`,
//! and at once after a failed model call, the pipeline is probed (model server, region
//! re-registration), so readiness follows the model server coming and going without traffic. A
//! lost CUDA context ends the tick, and with it the process: a lost context does not recover, the
//! restart policy starts a fresh one.
//!
//! [`probe`] is the other side, for `op-frame-engine health` (the container's health check): it
//! asks a running engine `ModelReady` for the model `frame_engine`, as the API does before every
//! photo under the ingest load profile.

use std::future::Future;
use std::time::Duration;

use crate::pipeline::Pipeline;
use crate::proto::ModelReadyRequest;
use crate::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::wire::contract::contract;

/// Why the tick ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ended {
    /// `stop` completed.
    Stopped,
    /// The pipeline reports a lost CUDA context.
    ContextLost,
}

/// Probes `pipeline` until `stop` completes or its context is lost.
pub async fn run<P: Pipeline>(
    pipeline: &P,
    every: Duration,
    stop: impl Future<Output = ()>,
) -> Ended {
    tokio::pin!(stop);
    loop {
        if pipeline.lost() {
            return Ended::ContextLost;
        }
        pipeline.check().await;
        tokio::select! {
            () = &mut stop => return Ended::Stopped,
            () = pipeline.wake().notified() => {}
            () = tokio::time::sleep(every) => {}
        }
    }
}

/// Whether the engine listening on `127.0.0.1:port` answers `ModelReady` true, within `timeout`
/// for the connection and again for the call.
pub async fn probe(port: u16, timeout: Duration) -> Result<bool, String> {
    let channel = tonic::transport::Endpoint::from_shared(format!("http://127.0.0.1:{port}"))
        .map_err(|e| e.to_string())?
        .connect_timeout(timeout)
        .timeout(timeout)
        .connect()
        .await
        .map_err(|e| format!("no engine on port {port}: {e}"))?;
    let model = &contract().model;
    let answer = GrpcInferenceServiceClient::new(channel)
        .model_ready(ModelReadyRequest {
            name: model.name.clone(),
            version: model.version.clone(),
        })
        .await
        .map_err(|e| format!("ModelReady: {e}"))?;
    Ok(answer.into_inner().ready)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::Instant;

    use tokio::sync::Notify;

    use super::*;
    use crate::pipeline::{Photo, PhotoAnswer, PipelineError};

    #[derive(Default)]
    struct Probed {
        checks: AtomicU32,
        lost: AtomicBool,
        wake: Notify,
    }

    impl Pipeline for Probed {
        fn ready(&self) -> bool {
            true
        }
        fn lost(&self) -> bool {
            self.lost.load(Ordering::SeqCst)
        }
        fn wake(&self) -> &Notify {
            &self.wake
        }
        async fn check(&self) {
            self.checks.fetch_add(1, Ordering::SeqCst);
        }
        async fn process(&self, _: Photo) -> Result<PhotoAnswer, PipelineError> {
            Err(PipelineError::Engine("no photos here".into()))
        }
    }

    #[tokio::test]
    async fn it_probes_on_its_period_and_at_once_when_woken() {
        let p = Arc::new(Probed::default());
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let tick = tokio::spawn({
            let p = Arc::clone(&p);
            async move {
                run(&*p, Duration::from_millis(40), async {
                    let _ = stopped.await;
                })
                .await
            }
        });
        tokio::time::sleep(Duration::from_millis(130)).await;
        let periodic = p.checks.load(Ordering::SeqCst);
        assert!(
            (3..=5).contains(&periodic),
            "{periodic} probes in 130 ms at a 40 ms period"
        );
        let t = Instant::now();
        let before = p.checks.load(Ordering::SeqCst);
        p.wake.notify_one();
        while p.checks.load(Ordering::SeqCst) == before && t.elapsed() < Duration::from_secs(1) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(
            t.elapsed() < Duration::from_millis(30),
            "a woken tick does not wait out its period"
        );
        stop.send(()).expect("stop");
        assert_eq!(tick.await.expect("joined"), Ended::Stopped);
    }

    #[tokio::test]
    async fn a_lost_context_ends_the_tick() {
        let p = Probed::default();
        p.lost.store(true, Ordering::SeqCst);
        let ended = run(&p, Duration::from_secs(60), std::future::pending()).await;
        assert_eq!(ended, Ended::ContextLost);
        assert_eq!(p.checks.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn the_probe_fails_when_no_engine_listens() {
        let free = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = free.local_addr().expect("addr").port();
        drop(free);
        let answer = probe(port, Duration::from_millis(500)).await;
        assert!(answer.expect_err("nothing there").contains("no engine"));
    }
}
