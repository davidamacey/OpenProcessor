"""Pin every metric name referenced by the Grafana dashboards and
Prometheus alert rules to a fixture of metrics this stack's Triton,
DCGM-exporter and node-exporter actually export.

Bug this guards against (installer plan, Wave 0 follow-up): the Triton
unified dashboard's "Model Ready" and "GPU Temperature" panels, and the
`ModelNotReady` alert, queried metric names this Triton version's
``/metrics`` endpoint never exposes (``nv_model_ready_state``,
``nv_gpu_temperature``) -- panels were always empty and the alert could
never fire. A third instance of the same bug class was also found while
fixing this: the "Model Track Latency Comparison" panel's P95/P99 lines
used ``histogram_quantile(..., nv_inference_request_duration_us_bucket)``,
but ``nv_inference_request_duration_us`` is a plain counter (confirmed
via ``# TYPE``), not a histogram -- Triton exposes no ``_bucket`` series
for it, so those two lines always returned no data too.

The fixture below was captured 2026-09-26 from a live, already-running
stack's exposition endpoints (read-only ``curl``/``docker exec wget``,
never a container mutation):
- Triton ``/metrics`` (this repo's own ``triton-server``/Dockerfile.triton
  image, scraped on the running opfinal deployment's Triton port);
- dcgm-exporter ``/metrics`` (``nvcr.io/nvidia/k8s/dcgm-exporter:3.3.5-3.4.0-ubuntu22.04``,
  the exact image pinned in docker-compose.yml's ``dcgm-exporter`` service);
- node-exporter ``/metrics`` (``prom/node-exporter:v1.10.2``, the exact
  image pinned in docker-compose.yml's ``node-exporter`` service).

Fixing a real drift here (a Triton/DCGM/node-exporter version bump adding
or renaming metrics) means re-capturing this fixture against the new
image, not just adding names to make the test pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_PATHS = tuple(sorted((REPO_ROOT / 'monitoring' / 'dashboards').glob('*.json')))
ALERT_PATH = REPO_ROOT / 'monitoring' / 'alerts' / 'triton-alerts.yml'

# Captured from Triton's own /metrics (this repo's Dockerfile.triton image,
# tritonserver 26.06 branch) -- every `# HELP nv_*` / `up` name it exposes.
_TRITON_METRICS = frozenset(
    {
        'up',
        'nv_cpu_memory_total_bytes',
        'nv_cpu_memory_used_bytes',
        'nv_cpu_utilization',
        'nv_energy_consumption',
        'nv_gpu_memory_total_bytes',
        'nv_gpu_memory_used_bytes',
        'nv_gpu_power_limit',
        'nv_gpu_power_usage',
        'nv_gpu_utilization',
        'nv_inference_compute_infer_duration_us',
        'nv_inference_compute_input_duration_us',
        'nv_inference_compute_output_duration_us',
        'nv_inference_count',
        'nv_inference_exec_count',
        'nv_inference_pending_request_count',
        'nv_inference_queue_duration_us',
        'nv_inference_request_duration_us',
        'nv_inference_request_failure',
        'nv_inference_request_success',
        'nv_model_load_duration_secs',
        'nv_pinned_memory_pool_total_bytes',
        'nv_pinned_memory_pool_used_bytes',
    }
)

# Captured from dcgm-exporter:3.3.5-3.4.0-ubuntu22.04's /metrics.
_DCGM_METRICS = frozenset(
    {
        'DCGM_FI_DEV_MEMORY_TEMP',
        'DCGM_FI_DEV_GPU_TEMP',
        'DCGM_FI_DEV_POWER_USAGE',
        'DCGM_FI_DEV_GPU_UTIL',
        'DCGM_FI_DEV_MEM_COPY_UTIL',
        'DCGM_FI_DEV_ENC_UTIL',
        'DCGM_FI_DEV_DEC_UTIL',
        'DCGM_FI_DEV_VGPU_LICENSE_STATUS',
        'DCGM_FI_DEV_FB_USED',
        # Re-checked 2026-10-03 against the live dcgm-exporter /metrics.
        'DCGM_FI_DEV_FB_FREE',
        'DCGM_FI_DEV_SM_CLOCK',
        'DCGM_FI_DEV_XID_ERRORS',
        'DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION',
    }
)

# Captured from node-exporter:v1.10.2's /metrics (a small, stable
# well-known subset -- only the ones this repo's dashboards actually use).
_NODE_EXPORTER_METRICS = frozenset(
    {
        'node_cpu_seconds_total',
        'node_load1',
        'node_memory_MemTotal_bytes',
        'node_memory_MemAvailable_bytes',
        'node_memory_Buffers_bytes',
        'node_memory_Cached_bytes',
    }
)

# Prometheus-side names for the API (op_* custom counters/histograms and the
# HTTP instrumentation histogram), the node-exporter extras and the synthetic
# ALERTS series. Captured 2026-10-03 from the live Prometheus label values.
_API_METRICS = frozenset(
    {
        'http_request_duration_seconds_count',
        'http_request_duration_seconds_bucket',
        'op_pipeline_stage_seconds_count',
        'op_pipeline_stage_seconds_sum',
        'op_pipeline_stage_seconds_bucket',
        'op_pipeline_stage_bytes_total',
        'op_source_image_decode_count_total',
        'op_source_image_prefetch_hits_total',
        'op_source_image_prefetch_misses_total',
        'op_thumbnail_cache_hits_total',
        'op_thumbnail_cache_misses_total',
        'op_shm_crop_cache_hits_total',
        'op_shm_crop_cache_misses_total',
        'op_shm_crop_cache_evictions_total',
        'op_open_vocab_items_written_total',
        'op_ingest_occ_final_conflict_total',
        'op_vlm_call_combined_count_total',
        'op_vlm_call_separate_count_total',
        'op_vlm_combined_parse_failure_total',
        # Issue #94 series. Defined in src/services/curation/ops_metrics.py and
        # pinned there (tests/curation/test_ops_metrics.py); not yet captured
        # from a live Prometheus -- confirm on the next deploy.
        'op_ingest_images_total',
        'op_ingest_items_total',
        'op_queue_depth',
        'op_queue_oldest_item_age_seconds',
        'op_worker_last_heartbeat_timestamp_seconds',
        'op_worker_up',
        'op_embedding_state_items',
        'op_region_segmenter_request_seconds_bucket',
        'op_region_segmenter_request_seconds_count',
        'op_region_segmenter_request_seconds_sum',
        'op_region_segmenter_requests_total',
        'op_detection_worker_items_total',
        'op_vlm_tokens_total',
        'op_vlm_request_seconds_bucket',
        'op_vlm_request_seconds_count',
        'op_vlm_request_seconds_sum',
        'op_opensearch_shards',
        'op_opensearch_store_bytes',
        'node_load5',
        'node_load15',
        'ALERTS',
    }
)

REAL_METRIC_NAMES = _TRITON_METRICS | _DCGM_METRICS | _NODE_EXPORTER_METRICS | _API_METRICS

# PromQL functions/aggregators/keywords that look like bare identifiers in
# an `expr` string but are never metric names.
_PROMQL_KEYWORDS = frozenset(
    {
        'rate',
        'irate',
        'sum',
        'avg',
        'min',
        'max',
        'count',
        'by',
        'without',
        'on',
        'group_left',
        'group_right',
        'histogram_quantile',
        'humanize',
        'humanizePercentage',
        'or',
        'vector',
        'clamp_min',
        'topk',
        'increase',
        'le',
        'mode',  # only ever a label key (node_cpu_seconds_total{mode=...}), never a metric
    }
)

_IDENTIFIER_RE = re.compile(r'[A-Za-z_:][A-Za-z0-9_:]*')
_LABEL_MATCHER_RE = re.compile(r'\{[^}]*\}')
_RANGE_VECTOR_RE = re.compile(r'\[[^\]]*\]')  # e.g. "[5m]" -- not an identifier
_AGGREGATION_CLAUSE_RE = re.compile(r'\b(?:by|without)\s*\([^)]*\)')


def _metric_names_in_expr(expr: str) -> set[str]:
    """Extract bare metric-name identifiers from a PromQL expression.

    Strips label-matcher blocks (`{model=~".*"}`), `by (...)`/`without
    (...)` aggregation clauses (label names, not metrics) and range-vector
    durations (`[5m]` -- the trailing `m`/`h`/`s` would otherwise look like
    a stray one-letter identifier) before tokenizing, then drops PromQL
    functions/keywords and anything immediately followed by `(` (a
    function call, not a metric).
    """
    stripped = _LABEL_MATCHER_RE.sub(' ', expr)
    stripped = _AGGREGATION_CLAUSE_RE.sub(' ', stripped)
    stripped = _RANGE_VECTOR_RE.sub(' ', stripped)
    stripped = re.sub(r'\b\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\b', ' ', stripped)  # numeric literals
    names: set[str] = set()
    for m in _IDENTIFIER_RE.finditer(stripped):
        name = m.group(0)
        if name in _PROMQL_KEYWORDS:
            continue
        if name[0].isdigit():
            continue
        # A function call: the identifier is immediately followed by '('.
        if stripped[m.end() : m.end() + 1] == '(':
            continue
        names.add(name)
    return names


def _dashboard_exprs(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    exprs: list[str] = []
    for panel in data.get('panels', []):
        for target in panel.get('targets', []) or []:
            expr = target.get('expr')
            ds = target.get('datasource')
            if isinstance(ds, dict) and ds.get('type') == 'loki':
                continue  # LogQL, not PromQL
            if expr:
                exprs.append(expr)
    return exprs


def _alert_exprs(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text())
    exprs: list[str] = []
    for group in data.get('groups', []):
        for rule in group.get('rules', []):
            expr = rule.get('expr')
            if expr:
                exprs.append(expr)
    return exprs


def test_fixture_is_nonempty_and_disjoint_from_promql_keywords() -> None:
    assert REAL_METRIC_NAMES
    assert not (REAL_METRIC_NAMES & _PROMQL_KEYWORDS)


def test_dashboard_metric_names_are_real() -> None:
    unknown: dict[str, set[str]] = {}
    for path in DASHBOARD_PATHS:
        for expr in _dashboard_exprs(path):
            bad = _metric_names_in_expr(expr) - REAL_METRIC_NAMES
            if bad:
                unknown.setdefault(f'{path.name}: {expr}', set()).update(bad)
    assert not unknown, (
        'dashboard panel(s) query metric(s) not in the real-metrics fixture '
        f'(always-empty panel risk): {unknown}'
    )


def test_alert_metric_names_are_real() -> None:
    unknown: dict[str, set[str]] = {}
    for expr in _alert_exprs(ALERT_PATH):
        bad = _metric_names_in_expr(expr) - REAL_METRIC_NAMES
        if bad:
            unknown.setdefault(expr, set()).update(bad)
    assert not unknown, (
        'alert rule(s) query metric(s) not in the real-metrics fixture '
        f'(never-fires risk): {unknown}'
    )


def test_model_not_ready_alert_is_gone() -> None:
    """Regression guard for the specific bug: no *rule* may query
    nv_model_ready_state (never exported), and no dashboard panel may be
    titled "Model Ready" backed by it. Checks parsed rule `expr` values,
    not the raw file text, since the file's own explanatory comment about
    this fix legitimately mentions the retired metric name."""
    for expr in _alert_exprs(ALERT_PATH):
        assert 'nv_model_ready_state' not in expr

    dashboard = json.loads(
        (REPO_ROOT / 'monitoring' / 'dashboards' / 'triton-unified-dashboard.json').read_text()
    )
    titles = {p.get('title') for p in dashboard.get('panels', [])}
    assert 'Model Ready' not in titles


def test_gpu_temperature_panel_uses_dcgm() -> None:
    dashboard = json.loads(
        (REPO_ROOT / 'monitoring' / 'dashboards' / 'triton-unified-dashboard.json').read_text()
    )
    panels = {p.get('title'): p for p in dashboard.get('panels', [])}
    assert 'GPU Temperature' in panels
    exprs = [t['expr'] for t in panels['GPU Temperature']['targets']]
    assert any('DCGM_FI_DEV_GPU_TEMP' in e for e in exprs)
    assert not any('nv_gpu_temperature' in e for e in exprs)


def test_every_dashboard_is_provisionable_and_linked() -> None:
    uids: set[str] = set()
    for path in DASHBOARD_PATHS:
        data = json.loads(path.read_text())
        assert data['uid'] not in uids, path.name
        uids.add(data['uid'])
        assert data['title'], path.name
        assert data['panels'], path.name
        assert 'openprocessor' in data.get('tags', []), path.name
        assert any(link.get('type') == 'dashboards' for link in data.get('links', [])), path.name
