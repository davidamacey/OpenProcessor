//! The arithmetic of a row, with no clock, no network and no GPU: percentiles, rates and the
//! model server's statistics deltas. Everything a reader needs to recompute a row from its
//! recorded counters is here.

use std::collections::BTreeMap;

use serde::Serialize;

/// The `p`-th percentile (0 to 100) of `values` by linear interpolation between the closest
/// ranks, numpy's default (`numpy.percentile`, method "linear"), so a row can be recomputed with
/// numpy from the recorded latencies. `values` is sorted in place; empty input gives 0.
pub fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let rank = p / 100.0 * (values.len() - 1) as f64;
    let (lo, hi) = (rank.floor() as usize, rank.ceil() as usize);
    values[lo] + (values[hi] - values[lo]) * (rank - lo as f64)
}

/// Photos per second: completions in a window of `seconds`.
pub fn rate(completions: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        completions as f64 / seconds
    }
}

/// One model's cumulative counters in the model server's statistics answer
/// (<https://github.com/triton-inference-server/server/blob/r26.10/docs/protocol/extension_statistics.md>):
/// `inference_count` counts batch items, `execution_count` counts executions (a dynamic batch is
/// one), `queue` is the requests' cumulative wait in the scheduler, `compute_infer` the cumulative
/// execution time. All grow from the server's start, so a window is the difference of two reads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Counters {
    /// Requests the model server counted as inferences (batch items) since it started.
    pub inference_count: u64,
    /// Executions the model server ran since it started; a dynamic batch is one execution.
    pub execution_count: u64,
    /// Requests whose scheduler wait is counted in `queue_ns`.
    pub queue_count: u64,
    /// Cumulative time requests waited in the scheduler, in nanoseconds.
    pub queue_ns: u64,
    /// Executions whose time is counted in `compute_infer_ns`.
    pub compute_infer_count: u64,
    /// Cumulative model execution time, in nanoseconds.
    pub compute_infer_ns: u64,
}

/// What a model did inside a window.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ModelWindow {
    /// Batch items the model inferred inside the window.
    pub inferences: u64,
    /// Executions the model ran inside the window.
    pub executions: u64,
    /// Batch items per execution.
    pub mean_batch: f64,
    /// Scheduler wait per request, milliseconds.
    pub queue_ms_per_request: f64,
    /// Execution time per execution, milliseconds.
    pub compute_infer_ms_per_execution: f64,
}

fn per(total: u64, count: u64) -> f64 {
    if count == 0 {
        0.0
    } else {
        total as f64 / count as f64
    }
}

/// The window between two reads of one model's counters; `None` when the counters went backwards
/// (the server restarted inside the window: the row must say so, not invent a delta).
pub fn model_window(before: &Counters, after: &Counters) -> Option<ModelWindow> {
    let d = |a: u64, b: u64| b.checked_sub(a);
    let inferences = d(before.inference_count, after.inference_count)?;
    let executions = d(before.execution_count, after.execution_count)?;
    let queue_count = d(before.queue_count, after.queue_count)?;
    let queue_ns = d(before.queue_ns, after.queue_ns)?;
    let compute_count = d(before.compute_infer_count, after.compute_infer_count)?;
    let compute_ns = d(before.compute_infer_ns, after.compute_infer_ns)?;
    Some(ModelWindow {
        inferences,
        executions,
        mean_batch: per(inferences, executions),
        queue_ms_per_request: per(queue_ns, queue_count) / 1e6,
        compute_infer_ms_per_execution: per(compute_ns, compute_count) / 1e6,
    })
}

/// Every model that ran in the window (`executions > 0`), by name. A model missing from
/// `before` started inside the window and counts from zero; one that went backwards is left out
/// and named in the second list.
pub fn windows(
    before: &BTreeMap<String, Counters>,
    after: &BTreeMap<String, Counters>,
) -> (BTreeMap<String, ModelWindow>, Vec<String>) {
    let mut ran = BTreeMap::new();
    let mut reset = Vec::new();
    for (name, now) in after {
        let then = before.get(name).copied().unwrap_or_default();
        match model_window(&then, now) {
            Some(w) if w.executions > 0 => {
                ran.insert(name.clone(), w);
            }
            Some(_) => {}
            None => reset.push(name.clone()),
        }
    }
    (ran, reset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_match_numpy_linear() {
        // numpy.percentile([1, 2, 3, 4, 10], [0, 25, 50, 90, 99, 100]) with method "linear"
        let mut v = vec![10.0, 1.0, 4.0, 2.0, 3.0];
        let got: Vec<f64> = [0.0, 25.0, 50.0, 90.0, 99.0, 100.0]
            .iter()
            .map(|p| percentile(&mut v, *p))
            .collect();
        let want = [1.0, 2.0, 3.0, 7.6, 9.76, 10.0];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-9, "{got:?} against {want:?}");
        }
    }

    #[test]
    fn a_percentile_of_nothing_or_of_one_value_is_defined() {
        assert_eq!(percentile(&mut [], 99.0), 0.0);
        assert_eq!(percentile(&mut [7.0], 99.0), 7.0);
    }

    #[test]
    fn the_rate_is_completions_over_the_window() {
        assert_eq!(rate(300, 20.0), 15.0);
        assert_eq!(rate(5, 0.0), 0.0);
    }

    fn counters(inf: u64, exec: u64, q_n: u64, q_ns: u64, c_n: u64, c_ns: u64) -> Counters {
        Counters {
            inference_count: inf,
            execution_count: exec,
            queue_count: q_n,
            queue_ns: q_ns,
            compute_infer_count: c_n,
            compute_infer_ns: c_ns,
        }
    }

    #[test]
    fn a_window_is_the_difference_of_two_reads() {
        let before = counters(1_000, 100, 1_000, 5_000_000, 100, 200_000_000);
        // 300 items in 10 executions: mean batch 30; 300 requests waited 12 ms in all; 10 runs
        // took 50 ms in all
        let after = counters(1_300, 110, 1_300, 17_000_000, 110, 250_000_000);
        let w = model_window(&before, &after).unwrap();
        assert_eq!((w.inferences, w.executions), (300, 10));
        assert!((w.mean_batch - 30.0).abs() < 1e-9);
        assert!((w.queue_ms_per_request - 12.0 / 300.0).abs() < 1e-9);
        assert!((w.compute_infer_ms_per_execution - 5.0).abs() < 1e-9);
    }

    #[test]
    fn counters_that_went_backwards_give_no_window() {
        let before = counters(1_000, 100, 1_000, 1, 100, 1);
        let after = counters(10, 1, 10, 1, 1, 1);
        assert_eq!(model_window(&before, &after), None);
        let (ran, reset) = windows(
            &BTreeMap::from([("m".to_string(), before)]),
            &BTreeMap::from([("m".to_string(), after)]),
        );
        assert!(ran.is_empty());
        assert_eq!(reset, ["m"]);
    }

    #[test]
    fn only_models_that_executed_appear_and_a_new_model_counts_from_zero() {
        let before = BTreeMap::from([
            ("idle".to_string(), counters(5, 5, 5, 5, 5, 5)),
            ("busy".to_string(), counters(10, 10, 10, 10, 10, 10)),
        ]);
        let after = BTreeMap::from([
            ("idle".to_string(), counters(5, 5, 5, 5, 5, 5)),
            ("busy".to_string(), counters(14, 12, 14, 10, 12, 10)),
            (
                "fresh".to_string(),
                counters(8, 2, 8, 8_000_000, 2, 4_000_000),
            ),
        ]);
        let (ran, reset) = windows(&before, &after);
        assert!(reset.is_empty());
        assert_eq!(
            ran.keys().map(String::as_str).collect::<Vec<_>>(),
            ["busy", "fresh"]
        );
        assert!((ran["busy"].mean_batch - 2.0).abs() < 1e-9);
        assert!((ran["fresh"].mean_batch - 4.0).abs() < 1e-9);
    }
}
