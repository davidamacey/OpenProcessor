//! The model server's statistics (`ModelStatistics`, the statistics extension), read at the
//! window's edges. The counters are cumulative from the server's start; the window is their
//! difference ([`crate::arith::windows`]).

use std::collections::BTreeMap;
use std::time::Duration;

use crate::arith::Counters;
use crate::client::Client;
use crate::proto::{ModelStatisticsRequest, ModelStatisticsResponse};

/// Counters of every model the server holds (an empty name asks for all), summed over versions.
pub fn counters_of(response: &ModelStatisticsResponse) -> BTreeMap<String, Counters> {
    let mut out: BTreeMap<String, Counters> = BTreeMap::new();
    for m in &response.model_stats {
        let stats = m.inference_stats.as_ref();
        let pick =
            |f: fn(&crate::proto::InferStatistics) -> Option<&crate::proto::StatisticDuration>| {
                stats.and_then(f).map_or((0, 0), |d| (d.count, d.ns))
            };
        let (queue_count, queue_ns) = pick(|s| s.queue.as_ref());
        let (compute_count, compute_ns) = pick(|s| s.compute_infer.as_ref());
        let c = out.entry(m.name.clone()).or_default();
        c.inference_count += m.inference_count;
        c.execution_count += m.execution_count;
        c.queue_count += queue_count;
        c.queue_ns += queue_ns;
        c.compute_infer_count += compute_count;
        c.compute_infer_ns += compute_ns;
    }
    out
}

/// Read every model's counters from the server.
pub async fn fetch(
    client: &mut Client,
    timeout: Duration,
) -> Result<BTreeMap<String, Counters>, String> {
    let mut request = tonic::Request::new(ModelStatisticsRequest::default());
    request.set_timeout(timeout);
    let response = client
        .model_statistics(request)
        .await
        .map_err(|e| format!("ModelStatistics: {e}"))?;
    Ok(counters_of(&response.into_inner()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::{InferStatistics, ModelStatistics, StatisticDuration};

    fn model(
        name: &str,
        inf: u64,
        exec: u64,
        queue: (u64, u64),
        compute: (u64, u64),
    ) -> ModelStatistics {
        ModelStatistics {
            name: name.into(),
            version: "1".into(),
            inference_count: inf,
            execution_count: exec,
            inference_stats: Some(InferStatistics {
                queue: Some(StatisticDuration {
                    count: queue.0,
                    ns: queue.1,
                }),
                compute_infer: Some(StatisticDuration {
                    count: compute.0,
                    ns: compute.1,
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn counters_are_read_per_model_and_summed_over_versions() {
        let response = ModelStatisticsResponse {
            model_stats: vec![
                model("det", 10, 5, (10, 1_000), (5, 2_000)),
                model("det", 2, 1, (2, 100), (1, 50)),
                ModelStatistics {
                    name: "bare".into(),
                    inference_count: 3,
                    ..Default::default()
                },
            ],
        };
        let got = counters_of(&response);
        assert_eq!(
            got["det"],
            Counters {
                inference_count: 12,
                execution_count: 6,
                queue_count: 12,
                queue_ns: 1_100,
                compute_infer_count: 6,
                compute_infer_ns: 2_050,
            }
        );
        assert_eq!(got["bare"].inference_count, 3);
        assert_eq!(got["bare"].queue_ns, 0);
    }
}
