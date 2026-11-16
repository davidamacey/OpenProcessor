//! The closed loop: `in_flight` tasks, each sending a request, waiting for its answer and
//! sending the next, so exactly `in_flight` requests are outstanding until the window closes.
//! The edges of the window call the same [`Probe`] the engine load generator uses, so every
//! counter a row reads (cgroup CPU, this process's CPU, load) is read at the instants the rate is
//! measured over.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use frame_engine_loadgen::exchange::Probe;
use tokio::time::Instant;

use crate::record::RequestRecord;
use crate::send::Sender;
use crate::summary::Window;

/// The loop's shape.
#[derive(Debug, Clone, Copy)]
pub struct LoopConfig {
    /// Requests outstanding at all times.
    pub in_flight: usize,
    /// Length of the unmeasured warm-up.
    pub warmup: Duration,
    /// Length of the measured window.
    pub window: Duration,
}

/// What the loop produced.
#[derive(Debug)]
pub struct LoopOutput {
    /// Every request, whenever it ended.
    pub records: Vec<RequestRecord>,
    /// The window as the loop's clock read it at the probes.
    pub window: Window,
    /// Requests started in total (the sequence numbers handed out).
    pub requests_started: u64,
}

/// Run the loop, from the sender's `t0`.
///
/// # Panics
///
/// Panics when a sending task itself panicked, so a defective instrument reports no rate.
pub async fn run<P: Probe>(sender: Arc<Sender>, config: LoopConfig, probe: &mut P) -> LoopOutput {
    let t0 = tokio::time::Instant::from_std(sender.t0);
    let open = t0 + config.warmup;
    let close = open + config.window;
    let next = Arc::new(AtomicU64::new(0));
    let tasks: Vec<_> = (0..config.in_flight)
        .map(|_| {
            let (sender, next) = (Arc::clone(&sender), Arc::clone(&next));
            tokio::spawn(async move {
                let mut done = Vec::new();
                while Instant::now() < close {
                    let seq = next.fetch_add(1, Ordering::Relaxed);
                    done.push(sender.send(seq).await);
                }
                done
            })
        })
        .collect();
    tokio::time::sleep_until(open).await;
    probe.window_open().await;
    let open_ms = (Instant::now() - t0).as_secs_f64() * 1e3;
    tokio::time::sleep_until(close).await;
    probe.window_close().await;
    let close_ms = (Instant::now() - t0).as_secs_f64() * 1e3;
    let mut records = Vec::new();
    for task in tasks {
        records.extend(task.await.expect("a sending task panicked"));
    }
    records.sort_by_key(|r| r.seq);
    LoopOutput {
        records,
        window: Window { open_ms, close_ms },
        requests_started: next.load(Ordering::Relaxed),
    }
}
