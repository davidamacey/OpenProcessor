//! Seconds per stage of one photo (`StageClock` of `frame_engine/stage_metrics.py`): the names
//! are the contract's, each stage entered is told to the request (the front logs the stage of a
//! request answered at its deadline), and the seconds are published when the photo ends, also
//! when it ends early.
//!
//! A stage's seconds are the time the photo spent in it: on the photo's task for routing, the
//! waits and the settles; on a pool thread for the kernels, the decode and the encodes, measured
//! there and added here. A model call stage counts the wait for its answer, not the call's
//! latency (`detect`, `embed_pe`). With `OP_FRAME_ENGINE_STAGE_SYNC` the pool threads wait for
//! their stream at the end of the kernel stages ([`SYNCED`]), so the card's time lands where it
//! ran; that removes overlap inside a photo and is for attribution runs only.

use std::future::Future;
use std::sync::Arc;
use std::time::Instant;

use crate::metrics::Metrics;
use crate::pipeline::RequestState;

pub const QUEUE: &str = "queue";
pub const BUDGET: &str = "budget";
pub const DECODE: &str = "decode";
pub const LETTERBOX: &str = "letterbox";
pub const DETECT: &str = "detect";
pub const EMBED_FRAME: &str = "embed_frame";
pub const EMBED_CROPS: &str = "embed_crops";
pub const EMBED_PE: &str = "embed_pe";
pub const CROP_COPY: &str = "crop_copy";
pub const CROP_JPEG: &str = "crop_jpeg";
pub const BLUR: &str = "blur";

/// Every stage this engine records, in the contract's order.
pub const ALL: [&str; 11] = [
    QUEUE,
    BUDGET,
    DECODE,
    LETTERBOX,
    DETECT,
    EMBED_FRAME,
    EMBED_CROPS,
    EMBED_PE,
    CROP_COPY,
    CROP_JPEG,
    BLUR,
];

/// The kernel stages whose end waits for the stream under `OP_FRAME_ENGINE_STAGE_SYNC`
/// (`stage_metrics.ASYNC_STAGES`, the ones of N0's branches).
pub const SYNCED: [&str; 4] = [LETTERBOX, EMBED_FRAME, EMBED_CROPS, BLUR];

/// One photo's seconds per stage, published to the metrics when dropped.
pub struct StageClock {
    request: Arc<RequestState>,
    metrics: Arc<Metrics>,
    seconds: Vec<(&'static str, f64)>,
}

impl StageClock {
    pub fn new(request: Arc<RequestState>, metrics: Arc<Metrics>) -> Self {
        Self {
            request,
            metrics,
            seconds: Vec::new(),
        }
    }

    pub fn add(&mut self, stage: &'static str, seconds: f64) {
        match self.seconds.iter_mut().find(|(s, _)| *s == stage) {
            Some((_, total)) => *total += seconds,
            None => self.seconds.push((stage, seconds)),
        }
    }

    /// Stages measured on a pool thread.
    pub fn merge(&mut self, stages: impl IntoIterator<Item = (&'static str, f64)>) {
        for (stage, seconds) in stages {
            self.add(stage, seconds);
        }
    }

    /// Tell the request it is in `stage`.
    pub fn enter(&self, stage: &'static str) {
        self.request.enter(stage);
    }

    pub fn time<T>(&mut self, stage: &'static str, f: impl FnOnce() -> T) -> T {
        self.enter(stage);
        let start = Instant::now();
        let out = f();
        self.add(stage, start.elapsed().as_secs_f64());
        out
    }

    pub async fn wait<T>(&mut self, stage: &'static str, f: impl Future<Output = T>) -> T {
        self.enter(stage);
        let start = Instant::now();
        let out = f.await;
        self.add(stage, start.elapsed().as_secs_f64());
        out
    }

    /// The seconds so far, in the order the stages were first entered.
    pub fn seconds(&self) -> &[(&'static str, f64)] {
        &self.seconds
    }
}

impl Drop for StageClock {
    fn drop(&mut self) {
        self.metrics
            .publish_stages(self.seconds.iter().map(|(s, v)| (*s, *v)));
    }
}

/// Stages measured on a pool thread, each entered on the request as it starts.
pub struct JobClock {
    request: Arc<RequestState>,
    pub stages: Vec<(&'static str, f64)>,
}

impl JobClock {
    pub fn new(request: Arc<RequestState>) -> Self {
        Self {
            request,
            stages: Vec::new(),
        }
    }

    pub fn time<T>(&mut self, stage: &'static str, f: impl FnOnce() -> T) -> T {
        self.request.enter(stage);
        let start = Instant::now();
        let out = f();
        self.stages.push((stage, start.elapsed().as_secs_f64()));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::RequestClock;
    use crate::wire::contract::contract;

    #[test]
    fn every_stage_is_a_contract_stage_in_the_contracts_order() {
        let stages = &contract().stages;
        let at: Vec<usize> = ALL
            .iter()
            .map(|s| {
                stages
                    .iter()
                    .position(|c| c == s)
                    .expect("a contract stage")
            })
            .collect();
        assert!(at.windows(2).all(|w| w[0] < w[1]));
        assert!(SYNCED.iter().all(|s| ALL.contains(s)));
    }

    #[test]
    fn the_clock_sums_by_stage_tells_the_request_and_publishes_once_when_dropped() {
        let metrics = Metrics::new();
        let request = Arc::new(RequestState::new(RequestClock::start(Instant::now(), 5.0)));
        let mut clock = StageClock::new(Arc::clone(&request), Arc::clone(&metrics));
        clock.add(QUEUE, 0.25);
        clock.time(DECODE, || ());
        assert_eq!(request.stage(), DECODE);
        clock.merge([(LETTERBOX, 0.5), (DECODE, 1.0), (LETTERBOX, 0.25)]);
        let names: Vec<&str> = clock.seconds().iter().map(|(s, _)| *s).collect();
        assert_eq!(names, [QUEUE, DECODE, LETTERBOX]);
        assert_eq!(clock.seconds()[2].1, 0.75);
        drop(clock);
        let text = metrics.render();
        assert!(text.contains("op_frame_engine_stage_observations_total{stage=\"letterbox\"} 1"));
        assert!(text.contains("op_frame_engine_stage_seconds_total{stage=\"letterbox\"} 0.75"));
    }
}
