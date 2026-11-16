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

/// Wait from arrival to the start of work.
pub const QUEUE: &str = "queue";
/// Wait for the decoded-frame budget.
pub const BUDGET: &str = "budget";
/// Decode, orientation and size check on a lane.
pub const DECODE: &str = "decode";
/// Resample of the detector's letterboxed input.
pub const LETTERBOX: &str = "letterbox";
/// The detector call.
pub const DETECT: &str = "detect";
/// Resample and call of the whole-frame embedding.
pub const EMBED_FRAME: &str = "embed_frame";
/// Resample and calls of the crop embeddings.
pub const EMBED_CROPS: &str = "embed_crops";
/// The perception-encoder embedding call.
pub const EMBED_PE: &str = "embed_pe";
/// Copy of each crop rectangle out of the frame.
pub const CROP_COPY: &str = "crop_copy";
/// JPEG encode of each crop.
pub const CROP_JPEG: &str = "crop_jpeg";
/// Laplacian sums for the blur measure.
pub const BLUR: &str = "blur";
/// Resample of the face detector's passes.
pub const FACE_LETTERBOX: &str = "face_letterbox";
/// The face detector's calls and their decode and NMS.
pub const FACE_DETECT: &str = "face_detect";
/// Warp of the aligned faces into the recogniser batches.
pub const FACE_ALIGN: &str = "face_align";
/// The face recogniser's calls.
pub const FACE_EMBED: &str = "face_embed";
/// Laplacian sums of the face boxes.
pub const FACE_BLUR: &str = "face_blur";

/// Every stage this engine records, in the contract's order.
pub const ALL: [&str; 16] = [
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
    FACE_LETTERBOX,
    FACE_DETECT,
    FACE_ALIGN,
    FACE_EMBED,
    FACE_BLUR,
];

/// The kernel stages whose end waits for the stream under `OP_FRAME_ENGINE_STAGE_SYNC`
/// (`stage_metrics.ASYNC_STAGES`, the ones of the branches built here).
pub const SYNCED: [&str; 7] = [
    LETTERBOX,
    EMBED_FRAME,
    EMBED_CROPS,
    BLUR,
    FACE_LETTERBOX,
    FACE_ALIGN,
    FACE_BLUR,
];

/// One photo's seconds per stage, published to the metrics when dropped.
pub struct StageClock {
    request: Arc<RequestState>,
    metrics: Arc<Metrics>,
    seconds: Vec<(&'static str, f64)>,
}

impl StageClock {
    /// A clock for `request` that publishes to `metrics` when dropped.
    pub fn new(request: Arc<RequestState>, metrics: Arc<Metrics>) -> Self {
        Self {
            request,
            metrics,
            seconds: Vec::new(),
        }
    }

    /// Adds `seconds` to `stage`'s total.
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

    /// Runs `f`, entering `stage` first and adding its wall time to the stage.
    pub fn time<T>(&mut self, stage: &'static str, f: impl FnOnce() -> T) -> T {
        self.enter(stage);
        let start = Instant::now();
        let out = f();
        self.add(stage, start.elapsed().as_secs_f64());
        out
    }

    /// Awaits `f`, entering `stage` first and adding its wall time to the stage.
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
    /// Seconds per stage in the order the stages ran.
    pub stages: Vec<(&'static str, f64)>,
}

impl JobClock {
    /// A clock for `request` with no stage yet.
    pub fn new(request: Arc<RequestState>) -> Self {
        Self {
            request,
            stages: Vec::new(),
        }
    }

    /// Runs `f` on the pool thread, entering `stage` on the request first and recording its wall time.
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
