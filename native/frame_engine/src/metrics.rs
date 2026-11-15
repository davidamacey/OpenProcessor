//! The engine's Prometheus series: the names, kinds and labels of `frame_engine/stage_metrics.py`
//! (the generated contract lists them: `metrics` and `stages`), plus the process series of
//! Python's `ProcessCollector`, read from `/proc/self`. Served on `OP_FRAME_ENGINE_METRICS_PORT`.
//!
//! The text is OpenMetrics (`prometheus-client` 0.26 writes no other): a counter family `x`
//! appears as `x_total`, as in the Python engine's, but without its `_created` series.

use std::convert::Infallible;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use prometheus_client::collector::Collector;
use prometheus_client::encoding::text::encode;
use prometheus_client::encoding::{DescriptorEncoder, EncodeLabelSet, EncodeMetric};
use prometheus_client::metrics::counter::{ConstCounter, Counter};
use prometheus_client::metrics::family::Family;
use prometheus_client::metrics::gauge::{ConstGauge, Gauge};
use prometheus_client::registry::Registry;
use tokio::net::TcpListener;

/// The content type of [`Metrics::render`] (OpenMetrics 1.0).
pub const CONTENT_TYPE: &str = "application/openmetrics-text; version=1.0.0; charset=utf-8";

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ByStage {
    pub stage: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ByOutcome {
    pub outcome: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ByCause {
    pub cause: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ByCode {
    pub code: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ByModelCall {
    pub model: String,
    pub code: String,
    pub reason: String,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
pub struct ByImage {
    pub status: String,
    pub decode_route: String,
}

type Seconds = Counter<f64, AtomicU64>;

/// Every series of one engine process. The fields are public: the front writes the ones it owns
/// (`in_flight`, `overloads`, `deadlines`), the pipeline the rest.
pub struct Metrics {
    registry: Registry,
    pub stage_seconds: Family<ByStage, Seconds>,
    pub stage_observations: Family<ByStage, Counter>,
    pub in_flight: Gauge,
    pub slab_in_use: Gauge,
    pub slab_peak: Gauge,
    pub slab_waits: Gauge,
    pub budget_in_use: Gauge,
    pub budget_waits: Family<ByOutcome, Counter>,
    pub registrations: Family<ByCause, Counter>,
    pub triton_errors: Family<ByModelCall, Counter>,
    pub deadlines: Family<ByStage, Counter>,
    pub outcomes: Family<ByImage, Counter>,
    pub context_lost: Family<ByCode, Counter>,
    pub overloads: Family<ByCause, Counter>,
    pub colour: Family<ByOutcome, Counter>,
    pub colour_table_bytes: Gauge,
}

/// Registers a family; the encoder ends each help text with a full stop, so the text given
/// does not.
fn register(
    registry: &mut Registry,
    name: &str,
    help: &str,
    metric: impl prometheus_client::registry::Metric,
) {
    registry.register(name, help.trim_end_matches('.'), metric);
}

impl Metrics {
    pub fn new() -> Arc<Self> {
        let mut registry = Registry::default();
        let m = Self {
            stage_seconds: Family::default(),
            stage_observations: Family::default(),
            in_flight: Gauge::default(),
            slab_in_use: Gauge::default(),
            slab_peak: Gauge::default(),
            slab_waits: Gauge::default(),
            budget_in_use: Gauge::default(),
            budget_waits: Family::default(),
            registrations: Family::default(),
            triton_errors: Family::default(),
            deadlines: Family::default(),
            outcomes: Family::default(),
            context_lost: Family::default(),
            overloads: Family::default(),
            colour: Family::default(),
            colour_table_bytes: Gauge::default(),
            registry: Registry::default(),
        };
        register(
            &mut registry,
            "op_frame_engine_stage_seconds",
            "Wall seconds the worker thread spent in a frame engine stage; a model call in \
             flight counts only while the worker waits for its answer (crop_jpeg: host-thread \
             encode seconds summed over the crops).",
            m.stage_seconds.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_stage_observations",
            "Images that ran a frame engine stage.",
            m.stage_observations.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_in_flight",
            "Requests admitted and not yet answered.",
            m.in_flight.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_slab_in_use_bytes",
            "Bytes of the region handed out.",
            m.slab_in_use.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_slab_peak_bytes",
            "Most bytes of the region handed out.",
            m.slab_peak.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_slab_waits",
            "Requests that waited for room in the region.",
            m.slab_waits.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_budget_in_use_bytes",
            "Bytes of the decoded-frame budget held by requests.",
            m.budget_in_use.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_budget_waits",
            "Requests that waited for device memory, by outcome (admitted or timeout).",
            m.budget_waits.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_region_registrations",
            "Shared-memory region registrations with the model server, by cause.",
            m.registrations.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_triton_call_errors",
            "Failed model-server calls, by model, gRPC status and reason (deadline: no answer \
             in time; unavailable: the channel or server went away; client: the client library \
             failed the call; rejected: the server answered an error).",
            m.triton_errors.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_request_deadlines",
            "Requests answered DEADLINE_EXCEEDED at the request deadline, by the stage they \
             were last in.",
            m.deadlines.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_images",
            "Images answered, by status (ok or the reason) and decode route.",
            m.outcomes.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_context_lost",
            "CUDA contexts lost to a sticky CUDA error, by cudaError code; the process then \
             exits for a restart (a dying process may not be scraped: changes() over \
             process_start_time_seconds counts the restarts).",
            m.context_lost.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_overloads",
            "Requests answered with the overload status, by cause.",
            m.overloads.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_colour",
            "Decoded frames by what the embedded colour profile asked: srgb (untagged or sRGB, \
             untouched), converted (to sRGB on the GPU) or ignored (not an RGB profile).",
            m.colour.clone(),
        );
        register(
            &mut registry,
            "op_frame_engine_colour_table_bytes",
            "Device bytes held by the cached profile-to-sRGB tables.",
            m.colour_table_bytes.clone(),
        );
        registry.register_collector(Box::new(ProcessSeries));
        Arc::new(Self { registry, ..m })
    }

    /// The exposition text of every series.
    pub fn render(&self) -> String {
        let mut text = String::new();
        encode(&mut text, &self.registry).expect("writing to a String does not fail");
        text
    }

    /// Adds one image's seconds per stage (`StageClock.publish`): each stage's seconds and one
    /// observation.
    pub fn publish_stages<'a>(&self, seconds: impl IntoIterator<Item = (&'a str, f64)>) {
        for (stage, secs) in seconds {
            let label = ByStage {
                stage: stage.to_owned(),
            };
            self.stage_seconds.get_or_create(&label).inc_by(secs);
            self.stage_observations.get_or_create(&label).inc();
        }
    }

    /// Counts a request answered `DEADLINE_EXCEEDED` in `stage`.
    pub fn count_deadline(&self, stage: &str) {
        self.deadlines
            .get_or_create(&ByStage {
                stage: stage.to_owned(),
            })
            .inc();
    }

    /// Counts a request answered with the overload status for `cause`.
    pub fn count_overload(&self, cause: &str) {
        self.overloads
            .get_or_create(&ByCause {
                cause: cause.to_owned(),
            })
            .inc();
    }
}

/// The `process_*` series of Python's `ProcessCollector`, read from `/proc/self` at each scrape
/// (proc(5): `stat` for the CPU time, start time and memory, `fd` and `limits` for the
/// descriptors). A series whose source cannot be read is left out, as there.
#[derive(Debug)]
struct ProcessSeries;

impl Collector for ProcessSeries {
    fn encode(&self, mut encoder: DescriptorEncoder) -> Result<(), std::fmt::Error> {
        let Ok(me) = procfs::process::Process::myself() else {
            return Ok(());
        };
        if let Ok(stat) = me.stat() {
            let ticks = procfs::ticks_per_second() as f64;
            let page = procfs::page_size();
            let started = procfs::boot_time_secs().map_or(0.0, |boot| boot as f64)
                + stat.starttime as f64 / ticks;
            let gauges = [
                (
                    "process_virtual_memory_bytes",
                    "Virtual memory size in bytes.",
                    stat.vsize as f64,
                ),
                (
                    "process_resident_memory_bytes",
                    "Resident memory size in bytes.",
                    (stat.rss * page) as f64,
                ),
                (
                    "process_start_time_seconds",
                    "Start time of the process since unix epoch in seconds.",
                    started,
                ),
            ];
            for (name, help, value) in gauges {
                let gauge = ConstGauge::new(value);
                gauge.encode(encoder.encode_descriptor(
                    name,
                    help,
                    None,
                    gauge.metric_type(),
                )?)?;
            }
            let cpu = ConstCounter::new((stat.utime + stat.stime) as f64 / ticks);
            cpu.encode(encoder.encode_descriptor(
                "process_cpu_seconds",
                "Total user and system CPU time spent in seconds.",
                None,
                cpu.metric_type(),
            )?)?;
        }
        if let Ok(open) = me.fd_count() {
            let gauge = ConstGauge::new(open as f64);
            gauge.encode(encoder.encode_descriptor(
                "process_open_fds",
                "Number of open file descriptors.",
                None,
                gauge.metric_type(),
            )?)?;
        }
        if let Ok(limits) = me.limits()
            && let procfs::process::LimitValue::Value(max) = limits.max_open_files.soft_limit
        {
            let gauge = ConstGauge::new(max as f64);
            gauge.encode(encoder.encode_descriptor(
                "process_max_fds",
                "Maximum number of open file descriptors.",
                None,
                gauge.metric_type(),
            )?)?;
        }
        Ok(())
    }
}

/// Serves the exposition on `listener` to every GET, whatever the path (as the Python engine's
/// `start_http_server` does), until `stop` completes.
pub async fn serve(listener: TcpListener, metrics: Arc<Metrics>, stop: impl Future<Output = ()>) {
    tokio::pin!(stop);
    loop {
        let stream = tokio::select! {
            () = &mut stop => return,
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(_) => continue,
            },
        };
        let metrics = Arc::clone(&metrics);
        tokio::spawn(async move {
            let answer = service_fn(move |_request: Request<hyper::body::Incoming>| {
                let body = Bytes::from(metrics.render());
                async move {
                    Ok::<_, Infallible>(
                        Response::builder()
                            .header("content-type", CONTENT_TYPE)
                            .body(Full::new(body))
                            .expect("a response with one header"),
                    )
                }
            });
            // a scraper that hangs up mid-answer is its own business
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), answer)
                .await;
        });
    }
}

/// `(sample name, labels)` of every line of an exposition, for tests: `name{a="b"} value` is
/// `("name", "a=\"b\"")`.
#[cfg(test)]
fn samples(text: &str) -> Vec<(String, String, String)> {
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let (series, value) = l.rsplit_once(' ').expect("a value");
            match series.split_once('{') {
                Some((name, rest)) => (
                    name.to_owned(),
                    rest.trim_end_matches('}').to_owned(),
                    value.to_owned(),
                ),
                None => (series.to_owned(), String::new(), value.to_owned()),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::contract;

    #[test]
    fn counters_and_gauges_appear_under_the_python_names() {
        let m = Metrics::new();
        m.publish_stages([("detect", 0.25), ("decode", 1.5)]);
        m.publish_stages([("detect", 0.25)]);
        m.in_flight.set(3);
        m.count_deadline("embed_pe");
        m.count_overload("in_flight");
        let text = m.render();
        let s = samples(&text);
        let value = |name: &str, labels: &str| {
            s.iter()
                .find(|(n, l, _)| n == name && l == labels)
                .map(|(_, _, v)| v.clone())
        };
        assert_eq!(
            value("op_frame_engine_stage_seconds_total", "stage=\"detect\"").as_deref(),
            Some("0.5")
        );
        assert_eq!(
            value(
                "op_frame_engine_stage_observations_total",
                "stage=\"detect\""
            )
            .as_deref(),
            Some("2")
        );
        assert_eq!(
            value(
                "op_frame_engine_stage_observations_total",
                "stage=\"decode\""
            )
            .as_deref(),
            Some("1")
        );
        assert_eq!(value("op_frame_engine_in_flight", "").as_deref(), Some("3"));
        assert_eq!(
            value(
                "op_frame_engine_request_deadlines_total",
                "stage=\"embed_pe\""
            )
            .as_deref(),
            Some("1")
        );
        assert_eq!(
            value("op_frame_engine_overloads_total", "cause=\"in_flight\"").as_deref(),
            Some("1")
        );
        assert!(text.ends_with("# EOF\n"));
    }

    #[test]
    fn every_family_of_the_contract_is_exposed_with_its_kind() {
        let text = used().render();
        for family in &contract().metrics {
            let kind = &family.kind;
            assert!(
                text.contains(&format!("# TYPE {} {kind}\n", family.name)),
                "{} ({kind}) is missing:\n{text}",
                family.name
            );
        }
    }

    /// A registry with one sample in every labelled family (a family with no child has no line).
    fn used() -> Arc<Metrics> {
        let m = Metrics::new();
        m.budget_waits
            .get_or_create(&ByOutcome {
                outcome: "admitted".into(),
            })
            .inc();
        m.registrations
            .get_or_create(&ByCause {
                cause: "start".into(),
            })
            .inc();
        m.triton_errors
            .get_or_create(&ByModelCall {
                model: "det".into(),
                code: "UNAVAILABLE".into(),
                reason: "deadline".into(),
            })
            .inc();
        m.outcomes
            .get_or_create(&ByImage {
                status: "ok".into(),
                decode_route: "gpu".into(),
            })
            .inc();
        m.context_lost
            .get_or_create(&ByCode { code: "700".into() })
            .inc();
        m.colour
            .get_or_create(&ByOutcome {
                outcome: "srgb".into(),
            })
            .inc();
        m.publish_stages([("queue", 0.0)]);
        m.count_deadline("queue");
        m.count_overload("slab");
        m
    }

    #[test]
    fn the_labels_of_each_family_are_the_contracts() {
        let m = used();
        let s = samples(&m.render());
        for family in &contract().metrics {
            let suffix = if family.kind == "counter" {
                "_total"
            } else {
                ""
            };
            let name = format!("{}{suffix}", family.name);
            let seen: Vec<&(String, String, String)> =
                s.iter().filter(|(n, _, _)| *n == name).collect();
            if family.labels.is_empty() {
                continue;
            }
            assert!(!seen.is_empty(), "{name} has no sample");
            for (_, labels, _) in seen {
                let keys: Vec<&str> = labels
                    .split(',')
                    .map(|kv| kv.split('=').next().unwrap_or(""))
                    .collect();
                assert_eq!(keys, family.labels, "{name}");
            }
        }
    }

    #[test]
    fn the_process_series_are_read_from_proc() {
        // CPU time is counted in clock ticks: burn a few so the counter is above zero
        let start = std::time::Instant::now();
        let mut x = 0u64;
        while start.elapsed() < std::time::Duration::from_millis(60) {
            x = std::hint::black_box(x.wrapping_add(1));
        }
        let text = Metrics::new().render();
        let s = samples(&text);
        let number = |name: &str| -> f64 {
            s.iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, _, v)| v.parse().expect("a number"))
                .unwrap_or_else(|| panic!("{name}:\n{text}"))
        };
        assert!(number("process_resident_memory_bytes") > 1_000_000.0);
        assert!(number("process_virtual_memory_bytes") >= number("process_resident_memory_bytes"));
        assert!(number("process_open_fds") >= 3.0);
        assert!(number("process_max_fds") >= number("process_open_fds"));
        assert!(number("process_cpu_seconds_total") > 0.0);
        let started = number("process_start_time_seconds");
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_secs_f64();
        assert!(
            started > 1.0e9 && started <= now,
            "start {started} now {now}"
        );
    }

    #[test]
    fn every_stage_of_the_contract_can_be_published() {
        let m = Metrics::new();
        m.publish_stages(contract().stages.iter().map(|stage| (stage.as_str(), 0.5)));
        let text = m.render();
        assert_eq!(contract().stages.len(), 24);
        for stage in &contract().stages {
            assert!(
                text.contains(&format!(
                    "op_frame_engine_stage_seconds_total{{stage=\"{stage}\"}} 0.5"
                )),
                "{stage}"
            );
        }
    }

    #[tokio::test]
    async fn the_endpoint_answers_a_scrape_with_the_exposition() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("addr");
        let m = Metrics::new();
        m.in_flight.set(7);
        let (stop, stopped) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(serve(listener, m, async {
            let _ = stopped.await;
        }));
        let mut client = tokio::net::TcpStream::connect(addr).await.expect("connect");
        client
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await
            .expect("send");
        let mut reply = String::new();
        client.read_to_string(&mut reply).await.expect("read");
        assert!(reply.starts_with("HTTP/1.1 200 OK"), "{reply}");
        assert!(
            reply
                .to_lowercase()
                .contains("content-type: application/openmetrics-text"),
            "{reply}"
        );
        assert!(reply.contains("op_frame_engine_in_flight 7"), "{reply}");
        stop.send(()).expect("stop");
        server.await.expect("the server ends when told to");
    }
}
