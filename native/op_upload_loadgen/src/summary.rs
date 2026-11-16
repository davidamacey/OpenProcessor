//! The row's arithmetic on the requests' records: no clock, no network, so a reader recomputes
//! every figure from the stamps file, and the unit tests fix the stamps by hand.
//!
//! A request counts in the window by when its answer ended, as the engine load generator counts a
//! photo (plan 17 section 5). Latencies are of requests answered 2xx; every other request is
//! counted under its failure key and never dropped.

use std::collections::BTreeMap;

use frame_engine_loadgen::arith::{percentile, rate};
use serde::Serialize;

use crate::record::{Outcome, RequestRecord};

/// The measured window on the loop's clock, in milliseconds from the loop's start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    /// When the window opened.
    pub open_ms: f64,
    /// When the window closed.
    pub close_ms: f64,
}

impl Window {
    /// The window's length in seconds.
    pub fn seconds(&self) -> f64 {
        (self.close_ms - self.open_ms) / 1e3
    }

    /// `warmup`, `window` or `after` for a request that ended at `finished_ms`.
    pub fn phase_of(&self, finished_ms: f64) -> &'static str {
        if finished_ms < self.open_ms {
            "warmup"
        } else if finished_ms >= self.close_ms {
            "after"
        } else {
            "window"
        }
    }
}

/// Percentiles and mean of a list of milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Spread {
    /// Median.
    pub p50: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Mean.
    pub mean: f64,
    /// Largest.
    pub max: f64,
}

impl Spread {
    fn of(mut values: Vec<f64>) -> Spread {
        if values.is_empty() {
            return Spread::default();
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let p50 = percentile(&mut values, 50.0);
        let p99 = percentile(&mut values, 99.0);
        Spread {
            p50,
            p99,
            mean,
            max: values.last().copied().unwrap_or(0.0),
        }
    }
}

/// What the window held.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Summary {
    /// The window's length on the loop's clock, in seconds.
    pub window_seconds_measured: f64,
    /// Requests answered 2xx inside the window.
    pub requests_ok: u64,
    /// Requests that failed inside the window, by key.
    pub request_failures_by_key: BTreeMap<String, u64>,
    /// Photos the server reports `failed` inside 2xx answers, by `error_kind`.
    pub photo_failures_by_kind: BTreeMap<String, u64>,
    /// Photos reported `success`.
    pub photos_success: u64,
    /// Photos reported `duplicate`.
    pub photos_duplicate: u64,
    /// Photos in 2xx answers that carried no per-photo results.
    pub photos_unread: u64,
    /// Photos in requests that failed.
    pub photos_in_failed_requests: u64,
    /// `photos_success + photos_duplicate + photos_unread`, per second of the window.
    pub photos_per_s: f64,
    /// Photo bytes of the 2xx requests per second of the window.
    pub photo_bytes_per_s: f64,
    /// Whole body bytes of the 2xx requests per second of the window.
    pub body_bytes_per_s: f64,
    /// Response time of the 2xx requests: request start to the end of the answer.
    pub request_ms: Spread,
    /// Per photo part: request start to the connection taking the part's last frame.
    pub part_taken_ms: Spread,
    /// Per request: the last part taken to the end of the answer, the server's tail.
    pub tail_after_last_part_ms: Spread,
    /// Requests that failed during the warm-up (reported, not measured).
    pub warmup_failed_requests: u64,
    /// Requests that ended after the window closed (counted nowhere else).
    pub after_window_requests: u64,
}

impl Summary {
    /// Requests that failed inside the window.
    pub fn failed_requests(&self) -> u64 {
        self.request_failures_by_key.values().sum()
    }

    /// Photos reported `failed` inside 2xx answers.
    pub fn failed_photos(&self) -> u64 {
        self.photo_failures_by_kind.values().sum()
    }
}

/// Summarise the records of a run against the window.
pub fn summarize(records: &[RequestRecord], window: Window) -> Summary {
    let mut s = Summary {
        window_seconds_measured: window.seconds(),
        ..Summary::default()
    };
    let (mut request_ms, mut part_ms, mut tail_ms) = (Vec::new(), Vec::new(), Vec::new());
    let (mut photo_bytes, mut body_bytes) = (0u64, 0u64);
    for r in records {
        match window.phase_of(r.finished_ms) {
            "warmup" => {
                s.warmup_failed_requests += u64::from(matches!(r.outcome, Outcome::Failed { .. }));
            }
            "after" => s.after_window_requests += 1,
            _ => match &r.outcome {
                Outcome::Failed { key } => {
                    *s.request_failures_by_key.entry(key.clone()).or_default() += 1;
                    s.photos_in_failed_requests += r.photos as u64;
                }
                Outcome::Answered {
                    success,
                    duplicate,
                    failed_by_kind,
                    read,
                } => {
                    s.requests_ok += 1;
                    s.photos_success += success;
                    s.photos_duplicate += duplicate;
                    if !read {
                        s.photos_unread += r.photos as u64;
                    }
                    for (kind, n) in failed_by_kind {
                        *s.photo_failures_by_kind.entry(kind.clone()).or_default() += n;
                    }
                    photo_bytes += r.photo_bytes;
                    body_bytes += r.body_bytes;
                    request_ms.push(r.response_ms);
                    part_ms.extend_from_slice(&r.parts_taken_ms);
                    if let Some(last) = r.parts_taken_ms.last() {
                        tail_ms.push(r.response_ms - last);
                    }
                }
            },
        }
    }
    let seconds = s.window_seconds_measured;
    let answered = (s.photos_success + s.photos_duplicate + s.photos_unread) as usize;
    s.photos_per_s = rate(answered, seconds);
    s.photo_bytes_per_s = if seconds > 0.0 {
        photo_bytes as f64 / seconds
    } else {
        0.0
    };
    s.body_bytes_per_s = if seconds > 0.0 {
        body_bytes as f64 / seconds
    } else {
        0.0
    };
    s.request_ms = Spread::of(request_ms);
    s.part_taken_ms = Spread::of(part_ms);
    s.tail_after_last_part_ms = Spread::of(tail_ms);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(
        seq: u64,
        finished: f64,
        parts: &[f64],
        response: f64,
        kinds: &[(&str, u64)],
    ) -> RequestRecord {
        RequestRecord {
            seq,
            pack: 0,
            photos: parts.len(),
            photo_bytes: 1000 * parts.len() as u64,
            body_bytes: 1100 * parts.len() as u64,
            started_ms: finished - response,
            finished_ms: finished,
            parts_taken_ms: parts.to_vec(),
            response_ms: response,
            outcome: Outcome::Answered {
                success: parts.len() as u64 - kinds.iter().map(|k| k.1).sum::<u64>(),
                duplicate: 0,
                failed_by_kind: kinds.iter().map(|(k, n)| ((*k).to_string(), *n)).collect(),
                read: true,
            },
        }
    }

    fn failed(seq: u64, finished: f64, photos: usize, key: &str) -> RequestRecord {
        RequestRecord {
            seq,
            pack: 0,
            photos,
            photo_bytes: 0,
            body_bytes: 0,
            started_ms: finished - 10.0,
            finished_ms: finished,
            parts_taken_ms: vec![],
            response_ms: 10.0,
            outcome: Outcome::Failed { key: key.into() },
        }
    }

    const WINDOW: Window = Window {
        open_ms: 1000.0,
        close_ms: 3000.0,
    };

    #[test]
    fn a_request_counts_in_the_window_by_when_it_ended() {
        let records = [
            ok(0, 999.9, &[10.0], 20.0, &[]),
            ok(1, 1000.0, &[10.0], 20.0, &[]),
            ok(2, 2999.9, &[10.0], 20.0, &[]),
            ok(3, 3000.0, &[10.0], 20.0, &[]),
            failed(4, 500.0, 3, "http:503"),
            failed(5, 2000.0, 3, "http:503"),
        ];
        let s = summarize(&records, WINDOW);
        assert_eq!(s.requests_ok, 2);
        assert_eq!(s.after_window_requests, 1);
        assert_eq!(s.warmup_failed_requests, 1);
        assert_eq!(s.request_failures_by_key["http:503"], 1);
        assert_eq!(s.photos_in_failed_requests, 3);
        assert_eq!(s.window_seconds_measured, 2.0);
    }

    #[test]
    fn rates_and_spreads_on_fixed_stamps() {
        // two requests in a 2 s window: 4 photos (one of them failed inside its answer) and 2
        let records = [
            ok(
                0,
                1500.0,
                &[100.0, 200.0, 300.0, 400.0],
                500.0,
                &[("decode_failed", 1)],
            ),
            ok(1, 2500.0, &[100.0, 200.0], 300.0, &[]),
        ];
        let s = summarize(&records, WINDOW);
        assert_eq!(s.photos_success, 5);
        assert_eq!(s.photo_failures_by_kind["decode_failed"], 1);
        assert_eq!(s.failed_photos(), 1);
        assert_eq!(s.photos_per_s, 2.5);
        assert_eq!(s.photo_bytes_per_s, 3000.0);
        assert_eq!(s.body_bytes_per_s, 3300.0);
        assert_eq!(s.request_ms.p50, 400.0);
        assert_eq!(s.request_ms.mean, 400.0);
        assert_eq!(s.request_ms.max, 500.0);
        // parts: 100,100,200,200,300,400 -> median 200; tails: 500-400 = 100 and 300-200 = 100
        assert_eq!(s.part_taken_ms.p50, 200.0);
        assert_eq!(s.part_taken_ms.max, 400.0);
        assert_eq!(s.tail_after_last_part_ms.p50, 100.0);
        assert_eq!(s.tail_after_last_part_ms.p99, 100.0);
    }

    #[test]
    fn answers_without_per_photo_results_are_unread_not_success() {
        let mut r = ok(0, 1500.0, &[1.0, 2.0], 5.0, &[]);
        r.outcome = Outcome::Answered {
            success: 0,
            duplicate: 0,
            failed_by_kind: BTreeMap::new(),
            read: false,
        };
        let s = summarize(&[r], WINDOW);
        assert_eq!((s.photos_success, s.photos_unread), (0, 2));
        assert_eq!(s.photos_per_s, 1.0);
    }

    #[test]
    fn an_empty_window_divides_by_nothing() {
        let s = summarize(
            &[],
            Window {
                open_ms: 5.0,
                close_ms: 5.0,
            },
        );
        assert_eq!((s.photos_per_s, s.photo_bytes_per_s), (0.0, 0.0));
        assert_eq!(s.request_ms, Spread::default());
    }

    #[test]
    fn duplicates_are_answered_photos_and_stay_apart() {
        let mut r = ok(0, 1500.0, &[1.0, 2.0, 3.0], 5.0, &[]);
        r.outcome = Outcome::Answered {
            success: 1,
            duplicate: 2,
            failed_by_kind: BTreeMap::new(),
            read: true,
        };
        let s = summarize(&[r], WINDOW);
        assert_eq!((s.photos_success, s.photos_duplicate), (1, 2));
        assert_eq!(s.photos_per_s, 1.5);
    }
}
