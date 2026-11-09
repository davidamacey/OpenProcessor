"""Celery workers serve their metrics and Prometheus scrapes them (6B-3).

Docs: https://prometheus.github.io/client_python/multiprocess/ (merged registry served by the
main process) and the dns_sd_config section of the Prometheus configuration reference.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from src.tasks import worker_metrics


if TYPE_CHECKING:
    import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SERVICES = (
    'jobs-worker',
    'periodic-worker',
    'open-vocab-worker',
    'cluster-worker',
    'curation-auto-label-worker',
    'curation-assist-loop-worker',
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def test_scrape_job_covers_every_celery_worker_by_dns() -> None:
    config = yaml.safe_load((REPO_ROOT / 'monitoring' / 'prometheus.yml').read_text())
    job = next(j for j in config['scrape_configs'] if j['job_name'] == 'celery-workers')
    (dns,) = job['dns_sd_configs']
    assert dns['type'] == 'A'
    assert dns['port'] == int(worker_metrics.DEFAULT_PORT)
    assert set(dns['names']) == set(WORKER_SERVICES)
    assert any(r.get('target_label') == 'service' for r in job['relabel_configs'])
    compose = (REPO_ROOT / 'docker-compose.yml').read_text()
    for service in WORKER_SERVICES:
        block = compose.split(f'\n  {service}:\n', 1)[1].split('\n  ', 1)[0]
        assert (
            'PROMETHEUS_MULTIPROC_DIR=/run/op-prom-worker'
            in compose.split(f'\n  {service}:\n', 1)[1].split('\n\n  ', 1)[0]
        ), block


def test_every_jobs_queue_dashboard_metric_is_defined_in_code() -> None:
    dashboard = json.loads(
        (REPO_ROOT / 'monitoring' / 'dashboards' / 'jobs-queue.json').read_text()
    )
    exprs = [t['expr'] for p in dashboard['panels'] for t in p['targets']]
    names = {n for e in exprs for n in re.findall(r'\b(op_[a-z0-9_]+)', e)}
    assert names
    source = ''.join(p.read_text() for p in (REPO_ROOT / 'src').rglob('*.py'))
    for name in names:
        base = re.sub(r'_(bucket|sum|count)$', '', name)
        assert f"'{base}'" in source, f'{name} is not exported by any code'


def test_server_serves_the_merged_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    port = _free_port()
    monkeypatch.setenv('PROMETHEUS_MULTIPROC_DIR', str(tmp_path))
    monkeypatch.setenv('OP_WORKER_METRICS_PORT', str(port))
    worker_metrics._started.clear()
    # prometheus_client picks file-backed values at import time, so a child writes the sample.
    subprocess.run(
        [
            sys.executable,
            '-c',
            "from prometheus_client import Counter;Counter('op_worker_probe_total','p').inc(3)",
        ],
        env={**os.environ, 'PROMETHEUS_MULTIPROC_DIR': str(tmp_path)},
        check=True,
    )
    assert worker_metrics.start_server() == port
    body = urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5).read().decode()
    assert 'op_worker_probe_total 3.0' in body


def test_server_stays_off_without_a_multiproc_dir_or_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    worker_metrics._started.clear()
    monkeypatch.delenv('PROMETHEUS_MULTIPROC_DIR', raising=False)
    assert worker_metrics.start_server() is None
    monkeypatch.setenv('PROMETHEUS_MULTIPROC_DIR', str(tmp_path))
    monkeypatch.setenv('OP_WORKER_METRICS_PORT', '0')
    assert worker_metrics.start_server() is None
