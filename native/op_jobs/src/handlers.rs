//! The two handlers the runner ships with: the jobs the live checks run through the whole path
//! (`src/tasks/infra.py`).

use std::time::Duration;

use serde_json::{Map, Value, json};

use crate::handler::{Handlers, JobContext, Outcome, RegisterError};
use crate::store::JobStore;

/// The shortest `sleep_noop` run, in seconds.
pub const MIN_SLEEP_S: u64 = 1;
/// The longest `sleep_noop` run, in seconds.
pub const MAX_SLEEP_S: u64 = 600;

fn object(value: &Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

/// Does nothing: proves a started job is claimed, run and finished through the wrapper.
pub async fn noop<S: JobStore>(ctx: JobContext<S>) -> Outcome {
    ctx.progress(1, Some(1), Some("noop"), json!({})).await;
    Outcome::Completed(object(&json!({"noop": true})))
}

/// Sleeps `params["seconds"]` (1 to 600), one progress snapshot a second, and stops at a cancel or
/// a shutdown. The live checks use it as a job that is still running when a worker is hurt.
pub async fn sleep_noop<S: JobStore>(ctx: JobContext<S>) -> Outcome {
    let seconds = ctx
        .params()
        .get("seconds")
        .and_then(Value::as_u64)
        .filter(|s| (MIN_SLEEP_S..=MAX_SLEEP_S).contains(s));
    let Some(seconds) = seconds else {
        return Outcome::failed(
            "handler_error",
            format!("seconds must be an integer from {MIN_SLEEP_S} to {MAX_SLEEP_S}"),
        );
    };
    let mut slept = 0;
    while slept < seconds && !ctx.cancelled() && !ctx.shutting_down() {
        tokio::time::sleep(Duration::from_secs(1)).await;
        slept += 1;
        ctx.progress(slept, Some(seconds), Some("sleeping"), json!({}))
            .await;
    }
    if slept < seconds && ctx.shutting_down() && !ctx.cancelled() {
        return Outcome::Interrupted;
    }
    Outcome::Completed(object(
        &json!({"slept": slept, "cancelled": slept < seconds}),
    ))
}

/// Register `noop` and `sleep_noop`.
///
/// # Errors
///
/// [`RegisterError`] when either kind is not marked `rust` in the kind table, or is registered.
pub fn register_builtin<S: JobStore>(handlers: &mut Handlers<S>) -> Result<(), RegisterError> {
    handlers.register("noop", noop::<S>)?;
    handlers.register("sleep_noop", sleep_noop::<S>)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::sync::{mpsc, watch};

    use super::*;
    use crate::fake::FakeStore;
    use crate::runner::{RunnerConfig, run_worker};
    use crate::state::JobStatus;
    use crate::watch::LeaseTiming;
    use op_db::ListenerEvent;

    fn config() -> RunnerConfig {
        RunnerConfig {
            owner: "w".into(),
            max_running: 2,
            poll_interval: Duration::from_secs(2),
            timing: LeaseTiming {
                ttl: Duration::from_secs(120),
                tick: Duration::from_secs(15),
                fence_margin: Duration::from_secs(30),
            },
            progress_min_interval: Duration::from_secs(1),
            drain: Duration::from_secs(30),
        }
    }

    fn worker(store: &Arc<FakeStore>) -> (watch::Sender<bool>, mpsc::Sender<ListenerEvent>) {
        let mut handlers = Handlers::new();
        register_builtin(&mut handlers).unwrap();
        let (stop, shutdown) = watch::channel(false);
        let (wake, events) = mpsc::channel(4);
        tokio::spawn(run_worker(
            Arc::clone(store),
            handlers,
            config(),
            events,
            shutdown,
        ));
        (stop, wake)
    }

    #[test]
    fn the_builtin_handlers_are_exactly_the_kinds_the_contract_marks_rust() {
        let mut handlers = Handlers::<FakeStore>::new();
        register_builtin(&mut handlers).unwrap();
        let from_contract: Vec<String> = crate::kinds::contract()
            .kinds
            .iter()
            .filter(|k| k.runtime == crate::kinds::Runtime::Rust)
            .map(|k| k.name.clone())
            .collect();
        assert_eq!(handlers.kinds(), from_contract);
    }

    #[test]
    fn a_kind_the_contract_gives_to_python_cannot_be_registered_here() {
        let mut handlers = Handlers::<FakeStore>::new();
        let run = |_: JobContext<FakeStore>| async { Outcome::Interrupted };
        assert_eq!(
            handlers.register("reprocess", run),
            Err(RegisterError::NotRustKind("reprocess".into()))
        );
        assert_eq!(
            handlers.register("no_such_kind", run),
            Err(RegisterError::UnknownKind("no_such_kind".into()))
        );
        handlers.register("noop", run).unwrap();
        assert_eq!(
            handlers.register("noop", run),
            Err(RegisterError::Twice("noop".into()))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn noop_completes_with_its_result_and_one_progress_snapshot() {
        let store = Arc::new(FakeStore::new());
        let id = store.insert("noop", json!({}), None).unwrap();
        let _w = worker(&store);
        tokio::time::sleep(Duration::from_secs(1)).await;
        let job = store.job(id);
        assert_eq!(job.status, JobStatus::Completed);
        assert_eq!(job.result.unwrap(), json!({"noop": true}));
        assert_eq!(job.progress.unwrap()["stage"], "noop");
    }

    #[tokio::test(start_paused = true)]
    async fn sleep_noop_sleeps_reports_each_second_and_validates_its_seconds() {
        let store = Arc::new(FakeStore::new());
        let ok = store
            .insert("sleep_noop", json!({"seconds": 3}), None)
            .unwrap();
        let bad = store
            .insert("sleep_noop", json!({"seconds": 601}), None)
            .unwrap();
        let missing = store.insert("sleep_noop", json!({}), None).unwrap();
        let _w = worker(&store);
        tokio::time::sleep(Duration::from_secs(5)).await;
        let done = store.job(ok);
        assert_eq!(done.status, JobStatus::Completed);
        assert_eq!(
            done.result.unwrap(),
            json!({"slept": 3, "cancelled": false})
        );
        assert_eq!(done.progress.unwrap()["done"], 3);
        for id in [bad, missing] {
            let job = store.job(id);
            assert_eq!(job.status, JobStatus::Failed);
            assert_eq!(job.error_code.as_deref(), Some("handler_error"));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn sleep_noop_stops_on_a_cancel_and_interrupts_on_a_shutdown() {
        let store = Arc::new(FakeStore::new());
        let cancelled = store
            .insert("sleep_noop", json!({"seconds": 600}), None)
            .unwrap();
        let (stop, _wake) = worker(&store);
        tokio::time::sleep(Duration::from_secs(5)).await;
        store.cancel(cancelled);
        tokio::time::sleep(Duration::from_secs(20)).await;
        let job = store.job(cancelled);
        assert_eq!(job.status, JobStatus::Cancelled);
        assert!(job.result.unwrap()["slept"].as_u64().unwrap() < 600);

        let draining = store
            .insert("sleep_noop", json!({"seconds": 600}), None)
            .unwrap();
        tokio::time::sleep(Duration::from_secs(4)).await;
        stop.send(true).unwrap();
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert_eq!(store.status(draining), JobStatus::Interrupted);
    }
}
