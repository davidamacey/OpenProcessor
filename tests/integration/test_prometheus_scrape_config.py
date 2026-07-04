"""Pins for monitoring/prometheus.yml scrape topology.

Guards against scrape jobs silently disappearing (dashboards go blank
without an error anywhere) and against jobs pointing at services that
do not exist in docker-compose.yml.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def prometheus_config() -> dict:
    with (_REPO_ROOT / 'monitoring' / 'prometheus.yml').open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope='module')
def compose_services() -> set[str]:
    with (_REPO_ROOT / 'docker-compose.yml').open() as fh:
        compose = yaml.safe_load(fh)
    return set(compose.get('services', {}))


def _jobs(prometheus_config: dict) -> dict[str, dict]:
    return {job['job_name']: job for job in prometheus_config['scrape_configs']}


def test_expected_scrape_jobs_present(prometheus_config: dict) -> None:
    jobs = _jobs(prometheus_config)
    for expected in ('triton', 'yolo-api', 'node', 'dcgm', 'loki'):
        assert expected in jobs, f'scrape job {expected!r} missing from prometheus.yml'


def test_scrape_targets_reference_compose_services(
    prometheus_config: dict, compose_services: set[str]
) -> None:
    """Every static target host must be a service defined in compose."""
    for job in prometheus_config['scrape_configs']:
        for static in job.get('static_configs', []):
            for target in static.get('targets', []):
                host = target.split(':')[0]
                assert host in compose_services, (
                    f'job {job["job_name"]!r} scrapes {host!r}, '
                    'which is not a docker-compose service'
                )
