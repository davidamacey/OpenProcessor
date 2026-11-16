//! The engine's metric families against the Python engine's (`/metrics` must keep the names, kinds
//! and label names dashboards and alerts read). The Python side is
//! `native/frame_engine/tests/golden/metrics_families.json`, taken from `EngineMetrics`' registry
//! by `scripts/codegen/frame_engine_parity_goldens.py`, with the reviewed differences and their
//! reasons. The native side is the exposition of a registry with one sample in every family.

use std::collections::BTreeMap;

use frame_engine::metrics::{ByCause, ByCode, ByImage, ByModelCall, ByOutcome, Metrics};
use serde_json::Value;

const FAMILIES: &str = include_str!("golden/metrics_families.json");

/// A family as the exposition shows it: kind and the label names of its samples.
type Seen = BTreeMap<String, (String, Option<Vec<String>>)>;

/// A registry with a sample in every labelled family, so its label names show.
fn used() -> std::sync::Arc<Metrics> {
    let m = Metrics::new();
    let by_cause = |c: &str| ByCause { cause: c.into() };
    let by_outcome = |o: &str| ByOutcome { outcome: o.into() };
    m.budget_waits.get_or_create(&by_outcome("admitted")).inc();
    m.registrations.get_or_create(&by_cause("start")).inc();
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
    m.colour.get_or_create(&by_outcome("srgb")).inc();
    m.publish_stages([("queue", 0.0)]);
    m.count_deadline("queue");
    m.count_overload("slab");
    m
}

fn label_names(sample: &str) -> Vec<String> {
    let Some((_, rest)) = sample.split_once('{') else {
        return Vec::new();
    };
    let inside = rest.split_once('}').map_or(rest, |(inside, _)| inside);
    inside
        .split("\",")
        .filter_map(|pair| pair.split('=').next())
        .map(str::to_owned)
        .collect()
}

fn exposition(text: &str) -> Seen {
    let mut seen = Seen::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, kind) = rest.split_once(' ').expect("name and kind");
            seen.insert(name.to_owned(), (kind.to_owned(), None));
        }
    }
    for line in text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let series = line.rsplit_once(' ').map_or(line, |(s, _)| s);
        let sample = series.split('{').next().unwrap_or(series);
        let family = seen
            .keys()
            .filter(|f| sample == f.as_str() || sample.starts_with(&format!("{f}_")))
            .max_by_key(|f| f.len())
            .cloned()
            .unwrap_or_else(|| panic!("a sample of no family: {line}"));
        let labels = label_names(series);
        let entry = seen.get_mut(&family).expect("known");
        assert!(
            entry.1.as_ref().is_none_or(|l| *l == labels),
            "{family} has samples with different label names"
        );
        entry.1 = Some(labels);
    }
    seen
}

#[test]
fn every_family_is_the_python_engines_or_a_reviewed_difference() {
    let doc: Value = serde_json::from_str(FAMILIES).expect("the committed families are JSON");
    let native = exposition(&used().render());
    let native_only = doc["native_only"].as_object().expect("native_only");
    let python_only = doc["python_only"].as_object().expect("python_only");
    let mut problems = Vec::new();
    for family in doc["families"].as_array().expect("families") {
        let name = family["name"].as_str().expect("name");
        assert!(
            family.get("buckets").is_none(),
            "{name}: compare buckets here"
        );
        if python_only.contains_key(name) {
            continue;
        }
        let Some((kind, labels)) = native.get(name) else {
            problems.push(format!("{name} is missing here"));
            continue;
        };
        if kind != family["type"].as_str().expect("type") {
            problems.push(format!("{name}: kind {kind}, python {}", family["type"]));
        }
        let want: Vec<String> = family["labels"]
            .as_array()
            .expect("labels")
            .iter()
            .map(|l| l.as_str().expect("label").to_owned())
            .collect();
        match labels {
            Some(got) if *got != want => {
                problems.push(format!("{name}: labels {got:?}, python {want:?}"))
            }
            None if !want.is_empty() => {
                problems.push(format!("{name}: no sample to read labels from"))
            }
            _ => {}
        }
    }
    let python: Vec<&str> = doc["families"]
        .as_array()
        .expect("families")
        .iter()
        .filter_map(|f| f["name"].as_str())
        .collect();
    for name in native.keys() {
        if !python.contains(&name.as_str()) && !native_only.contains_key(name) {
            problems.push(format!(
                "{name} exists only here and is not a reviewed difference"
            ));
        }
    }
    for name in native_only.keys() {
        if !native.contains_key(name) {
            problems.push(format!("reviewed native-only {name} is not exposed"));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn counters_are_exposed_with_the_total_suffix_the_python_engine_writes() {
    let text = used().render();
    for family in [
        "op_frame_engine_stage_seconds",
        "op_frame_engine_images",
        "process_cpu_seconds",
    ] {
        assert!(
            text.contains(&format!("{family}_total")),
            "{family}:\n{text}"
        );
    }
}
