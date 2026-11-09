"""The job wrapper observes duration and failures (6B-3).

Docs: https://prometheus.github.io/client_python/instrumenting/histogram/ and
https://prometheus.io/docs/practices/alerting/ (alert on ``increase()`` of a counter).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from prometheus_client import REGISTRY

from src.domain import job_states as js
from src.tasks.runtime import job_task
from src.tasks.runtime.job_metrics import observe_delivery


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_a_failed_delivery_counts_one_failure_and_one_duration() -> None:
    labels = {'kind': 'op.test.metrics_a'}
    before_fail = _sample('op_jobs_failed_total', **labels)
    before_n = _sample('op_job_duration_seconds_count', outcome=js.FAILED, **labels)
    observe_delivery('op.test.metrics_a', js.FAILED, 12.5)
    assert _sample('op_jobs_failed_total', **labels) == before_fail + 1
    assert _sample('op_job_duration_seconds_count', outcome=js.FAILED, **labels) == before_n + 1
    assert _sample('op_job_duration_seconds_sum', outcome=js.FAILED, **labels) >= 12.5


def test_a_completed_delivery_is_not_a_failure() -> None:
    labels = {'kind': 'op.test.metrics_b'}
    observe_delivery('op.test.metrics_b', js.COMPLETED, 1.0)
    assert _sample('op_jobs_failed_total', **labels) == 0.0
    assert _sample('op_job_duration_seconds_count', outcome=js.COMPLETED, **labels) == 1.0


def test_the_wrapper_reports_through_the_metrics_module() -> None:
    source = (REPO_ROOT / 'src' / 'tasks' / 'runtime' / 'job_task.py').read_text()
    assert 'observe_delivery(' in source
    assert hasattr(job_task, 'job_metrics')


def test_failure_burst_alert_reads_the_counter() -> None:
    rules = yaml.safe_load((REPO_ROOT / 'monitoring' / 'alerts' / 'jobs-alerts.yml').read_text())
    burst = next(r for g in rules['groups'] for r in g['rules'] if r['alert'] == 'JobFailureBurst')
    assert burst['expr'] == 'sum by (kind) (increase(op_jobs_failed_total[1h])) > 5'


def test_dashboard_has_a_duration_percentile_panel() -> None:
    dashboard = json.loads(
        (REPO_ROOT / 'monitoring' / 'dashboards' / 'jobs-queue.json').read_text()
    )
    exprs = [t['expr'] for p in dashboard['panels'] for t in p['targets']]
    assert any(
        'histogram_quantile' in e and 'op_job_duration_seconds_bucket' in e and 'le' in e
        for e in exprs
    )
    ids = [p['id'] for p in dashboard['panels']]
    assert len(ids) == len(set(ids))
