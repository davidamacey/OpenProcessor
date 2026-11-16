"""The Rust front's HTTP latency histogram is the Python one: same name, labels and buckets
(``native/op_api/src/metrics.rs``; dashboards and alerts read ``http_request_duration_seconds``)."""

from __future__ import annotations

import re
from pathlib import Path

from src.core.metrics import HTTP_REQUEST_DURATION_SECONDS


RUST = (Path(__file__).resolve().parents[2] / 'native' / 'op_api' / 'src' / 'metrics.rs').read_text(
    encoding='utf-8'
)


def test_the_rust_histogram_has_the_python_name() -> None:
    assert re.search(r'"http_request_duration_seconds",\s*"HTTP request latency', RUST)
    assert HTTP_REQUEST_DURATION_SECONDS._name == 'http_request_duration_seconds'


def test_the_rust_histogram_has_the_python_buckets() -> None:
    constant = re.search(r'const BUCKETS: \[f64; (\d+)\] = \[(.*?)\];', RUST, flags=re.S)
    assert constant is not None
    rust = [float(v) for v in re.findall(r'[0-9.]+', constant.group(2))]
    python = list(HTTP_REQUEST_DURATION_SECONDS._upper_bounds)[:-1]  # the last is +Inf
    assert int(constant.group(1)) == len(rust)
    assert rust == python


def test_the_rust_histogram_has_the_python_labels() -> None:
    labels = re.search(r'struct HttpLabels \{(.*?)\}', RUST, flags=re.S)
    assert labels is not None
    rust = re.findall(r'^\s*(\w+): String,', labels.group(1), flags=re.M)
    assert rust == list(HTTP_REQUEST_DURATION_SECONDS._labelnames)
