//! What one photo's round trip is, whichever engine or models answer it, and the closed loop that
//! drives it. The loop knows nothing of gRPC: the engine client and the models-alone mode each
//! implement [`Exchange`], so the loop's arithmetic is tested against a fake and the client
//! against an in-process engine.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tokio::time::Instant;

/// How one photo ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The engine answered `status` `ok` and no branch failed. `route` is the decode route it
    /// reported; `bytes` the size of the answer on the wire.
    Ok { route: String, bytes: u64 },
    /// Anything else, under a key a reader can group by: `grpc:<CODE>`, `status:<photo status>`,
    /// `branch:<name>:<reason>`, `protocol:<what>`.
    Failed { key: String },
}

/// One photo's round trip.
pub trait Exchange: Send + Sync + 'static {
    /// Send photo `index` of the list and wait for its answer.
    fn exchange(&self, index: usize) -> impl Future<Output = Outcome> + Send;
}

/// The loop's shape: `in_flight` photos at all times, a warm-up that is not measured, then a
/// window that is.
#[derive(Debug, Clone, Copy)]
pub struct LoopConfig {
    pub in_flight: usize,
    pub warmup: Duration,
    pub window: Duration,
    /// How many photos the list holds: the loop walks it round robin.
    pub list_len: usize,
}

/// Called at the two edges of the window, from the loop's own clock, so every counter a row
/// reads (CPU, model server statistics, load) is read at the instants the rate is measured over.
pub trait Probe: Send {
    fn window_open(&mut self) -> impl Future<Output = ()> + Send;
    fn window_close(&mut self) -> impl Future<Output = ()> + Send;
}

/// What the window held.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoopResult {
    /// Latencies in milliseconds of the photos that completed inside the window with `ok`.
    pub latencies_ms: Vec<f64>,
    /// Failures by key, for photos that completed inside the window.
    pub failures: BTreeMap<String, u64>,
    /// Failures that completed during the warm-up (reported, not measured).
    pub warmup_failures: u64,
    /// Photos that completed after the window closed (not counted anywhere else).
    pub after_window: u64,
    pub routes: BTreeMap<String, u64>,
    pub response_bytes: u64,
    /// The window's length on the loop's clock.
    pub window_seconds: f64,
}

impl LoopResult {
    pub fn ok(&self) -> usize {
        self.latencies_ms.len()
    }

    pub fn failed(&self) -> u64 {
        self.failures.values().sum()
    }
}

struct Done {
    finished: Instant,
    latency: Duration,
    outcome: Outcome,
}

/// Run the closed loop. Each of `in_flight` tasks sends a photo, waits for the answer and sends
/// the next, so exactly `in_flight` are outstanding until the window closes. A photo counts in the
/// window by when it completed.
pub async fn run<E: Exchange, P: Probe>(
    exchange: Arc<E>,
    config: LoopConfig,
    probe: &mut P,
) -> LoopResult {
    let start = Instant::now();
    let open = start + config.warmup;
    let close = open + config.window;
    let next = Arc::new(AtomicUsize::new(0));
    let tasks: Vec<_> = (0..config.in_flight)
        .map(|_| {
            let exchange = Arc::clone(&exchange);
            let next = Arc::clone(&next);
            tokio::spawn(async move {
                let mut done = Vec::new();
                while Instant::now() < close {
                    let index = next.fetch_add(1, Ordering::Relaxed) % config.list_len;
                    let sent = Instant::now();
                    let outcome = exchange.exchange(index).await;
                    let finished = Instant::now();
                    done.push(Done {
                        finished,
                        latency: finished - sent,
                        outcome,
                    });
                }
                done
            })
        })
        .collect();
    tokio::time::sleep_until(open).await;
    probe.window_open().await;
    let opened = Instant::now();
    tokio::time::sleep_until(close).await;
    probe.window_close().await;
    let closed = Instant::now();
    let mut result = LoopResult {
        window_seconds: (closed - opened).as_secs_f64(),
        ..LoopResult::default()
    };
    for task in tasks {
        // A panic in an exchange is a defect of the instrument: stop, do not report a rate.
        let done = task.await.expect("an exchange task panicked");
        for d in done {
            if d.finished < open {
                if matches!(d.outcome, Outcome::Failed { .. }) {
                    result.warmup_failures += 1;
                }
            } else if d.finished >= close {
                result.after_window += 1;
            } else {
                match d.outcome {
                    Outcome::Ok { route, bytes } => {
                        result.latencies_ms.push(d.latency.as_secs_f64() * 1e3);
                        *result.routes.entry(route).or_default() += 1;
                        result.response_bytes += bytes;
                    }
                    Outcome::Failed { key } => *result.failures.entry(key).or_default() += 1,
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Answers after a fixed delay; photo `fail_every`, 2 * `fail_every`, ... fail.
    struct Fake {
        delay: Duration,
        fail_every: usize,
        seen: Mutex<Vec<usize>>,
        outstanding: AtomicUsize,
        peak: AtomicUsize,
    }

    impl Fake {
        fn new(delay: Duration, fail_every: usize) -> Arc<Fake> {
            Arc::new(Fake {
                delay,
                fail_every,
                seen: Mutex::new(Vec::new()),
                outstanding: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            })
        }
    }

    impl Exchange for Fake {
        async fn exchange(&self, index: usize) -> Outcome {
            let now = self.outstanding.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            self.seen.lock().unwrap().push(index);
            tokio::time::sleep(self.delay).await;
            self.outstanding.fetch_sub(1, Ordering::SeqCst);
            if self.fail_every > 0 && index % self.fail_every == self.fail_every - 1 {
                Outcome::Failed {
                    key: "grpc:UNAVAILABLE".into(),
                }
            } else {
                Outcome::Ok {
                    route: "gpu".into(),
                    bytes: 100,
                }
            }
        }
    }

    #[derive(Default)]
    struct Marks {
        opened: Option<Instant>,
        closed: Option<Instant>,
    }

    impl Probe for Marks {
        async fn window_open(&mut self) {
            self.opened = Some(Instant::now());
        }
        async fn window_close(&mut self) {
            self.closed = Some(Instant::now());
        }
    }

    fn config(in_flight: usize, list_len: usize) -> LoopConfig {
        LoopConfig {
            in_flight,
            warmup: Duration::from_secs(2),
            window: Duration::from_secs(10),
            list_len,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn the_window_counts_completions_inside_it_and_nothing_else() {
        // 8 in flight, 100 ms each: 80 photos a second, 800 in a 10 s window
        let fake = Fake::new(Duration::from_millis(100), 0);
        let mut marks = Marks::default();
        let r = run(fake.clone(), config(8, 1000), &mut marks).await;
        assert_eq!(r.ok(), 800);
        assert_eq!(r.failed(), 0);
        assert!((r.window_seconds - 10.0).abs() < 1e-6);
        assert!((crate::arith::rate(r.ok(), r.window_seconds) - 80.0).abs() < 1e-6);
        assert_eq!(
            fake.peak.load(Ordering::SeqCst),
            8,
            "closed loop: never more than N"
        );
        assert_eq!(
            r.after_window, 8,
            "each lane's photo sent at 11.9 s completes at 12.0 s"
        );
        assert_eq!(r.routes["gpu"], 800);
        assert_eq!(r.response_bytes, 80_000);
        let mut l = r.latencies_ms.clone();
        assert!((crate::arith::percentile(&mut l, 99.0) - 100.0).abs() < 1.0);
    }

    #[tokio::test(start_paused = true)]
    async fn probe_edges_sit_on_the_warm_up_and_window_boundaries() {
        let fake = Fake::new(Duration::from_millis(100), 0);
        let mut marks = Marks::default();
        let t0 = Instant::now();
        let _ = run(fake, config(4, 10), &mut marks).await;
        assert_eq!(marks.opened.unwrap() - t0, Duration::from_secs(2));
        assert_eq!(marks.closed.unwrap() - t0, Duration::from_secs(12));
    }

    #[tokio::test(start_paused = true)]
    async fn failures_are_counted_by_key_and_are_not_in_the_rate() {
        // every 4th photo fails: with a 4-photo list, photo 3 each lap
        let fake = Fake::new(Duration::from_millis(100), 4);
        let mut marks = Marks::default();
        let r = run(fake, config(1, 4), &mut marks).await;
        assert_eq!(r.ok() + r.failed() as usize, 100);
        assert_eq!(r.failures["grpc:UNAVAILABLE"], 25);
        assert_eq!(r.ok(), 75);
        assert_eq!(
            r.warmup_failures, 4,
            "photos 3, 7, 11 and 15 completed during the warm-up"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_list_is_walked_round_robin_and_photos_finishing_after_close_are_not_counted() {
        // 3.7 s a photo against a 10 s window and 3 in flight: the last ones finish after close
        let fake = Fake::new(Duration::from_millis(3700), 0);
        let mut marks = Marks::default();
        let r = run(fake.clone(), config(3, 5), &mut marks).await;
        let seen = fake.seen.lock().unwrap().clone();
        assert_eq!(seen[..7], [0, 1, 2, 3, 4, 0, 1]);
        // each of 3 lanes completes at 3.7, 7.4 and 11.1 s (all inside 2..12 s), and its next
        // photo, sent at 11.1 s, would complete at 14.8 s
        assert_eq!((r.ok(), r.after_window), (9, 3));
    }
}
