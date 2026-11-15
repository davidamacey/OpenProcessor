//! The health tick (`_health` in `frame_engine/__main__.py`): every `OP_FRAME_ENGINE_HEALTH_SECONDS`,
//! and at once after a failed model call, the pipeline is probed (model server, region
//! re-registration), so readiness follows the model server coming and going without traffic. A
//! lost CUDA context ends the tick, and with it the process: a lost context does not recover, the
//! restart policy starts a fresh one.

use std::future::Future;
use std::time::Duration;

use crate::pipeline::Pipeline;

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
}
