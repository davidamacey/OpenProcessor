"""6B-6 / R-12 / RC-8: replicas are scraped one by one, logs are shipped for every service, and
dashboards key on `service`, never on `container` or `instance`.

Docs: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#dns_sd_config,
https://grafana.com/docs/alloy/latest/reference/components/discovery/discovery.relabel/.
"""

from __future__ import annotations

import json
import re

import yaml

from compose._helpers import REPO_ROOT, _services
from compose.test_consumers_after_b2 import consumers


MONITORING = REPO_ROOT / 'monitoring'
ARBITER = {'gpu-arbiter-worker'}  # only in the overlay render


def _prometheus_jobs() -> dict[str, dict]:
    config = yaml.safe_load((MONITORING / 'prometheus.yml').read_text())
    return {job['job_name']: job for job in config['scrape_configs']}


def _logged_services() -> set[str]:
    alloy = (MONITORING / 'alloy-config.alloy').read_text()
    kept: set[str] = set()
    for match in re.finditer(
        r'__meta_docker_container_label_com_docker_compose_service"\]\s*'
        r'regex\s*=\s*"([^"]+)"\s*action\s*=\s*"keep"',
        alloy,
    ):
        kept |= set(match.group(1).split('|'))
    return kept


def test_api_and_triton_are_scraped_by_dns_so_every_replica_is_a_target() -> None:
    jobs = _prometheus_jobs()
    for name, service, port in (('api', 'api', 8000), ('triton', 'triton-server', 8002)):
        dns = jobs[name]['dns_sd_configs']
        assert dns == [{'names': [service], 'type': 'A', 'port': port}]
        assert 'static_configs' not in jobs[name]
        assert any(rule.get('target_label') == 'service' for rule in jobs[name]['relabel_configs'])


def test_every_queue_consumer_and_api_has_a_log_rule() -> None:
    wanted = {name for names in consumers(_services()).values() for name in names}
    wanted |= {'api', 'triton-server', 'postgres', 'valkey', *ARBITER}
    assert wanted <= _logged_services(), sorted(wanted - _logged_services())


def test_every_logged_service_label_is_a_compose_service() -> None:
    known = set(_services()) | ARBITER | {'segmenter'}
    assert _logged_services() <= known, sorted(_logged_services() - known)


def test_log_streams_carry_the_service_label() -> None:
    alloy = (MONITORING / 'alloy-config.alloy').read_text()
    assert alloy.count('target_label  = "service"') == alloy.count('discovery.relabel "')


def test_the_valkey_exporter_is_scraped_and_alerted_on() -> None:
    assert 'valkey-exporter' in _services()
    assert _prometheus_jobs()['valkey']['static_configs'][0]['targets'] == ['valkey-exporter:9121']
    rules = yaml.safe_load((MONITORING / 'alerts' / 'jobs-alerts.yml').read_text())['groups'][0][
        'rules'
    ]
    valkey = next(r for r in rules if r['alert'] == 'ValkeyMemoryNearMax')
    assert 'redis_memory_used_bytes' in valkey['expr']
    assert 'redis_memory_max_bytes' in valkey['expr']


def test_the_jobs_alerts_are_the_owner_d83_set() -> None:
    rules = yaml.safe_load((MONITORING / 'alerts' / 'jobs-alerts.yml').read_text())['groups'][0][
        'rules'
    ]
    by_name = {r['alert']: r for r in rules}
    assert {'OldestQueuedJob', 'JobFailureBurst', 'DiskLow'} <= set(by_name)
    assert '900' in by_name['OldestQueuedJob']['expr']
    assert by_name['OldestQueuedJob']['for'] == '5m'
    assert 'op_jobs_failed_total' in by_name['JobFailureBurst']['expr']
    assert '> 5' in by_name['JobFailureBurst']['expr']
    assert 'op_disk_free_ratio' in by_name['DiskLow']['expr']
    assert '0.10' in by_name['DiskLow']['expr']


def test_the_jobs_queue_dashboard_reads_the_job_gauges() -> None:
    dashboard = json.loads((MONITORING / 'dashboards' / 'jobs-queue.json').read_text())
    exprs = ' '.join(t['expr'] for p in dashboard['panels'] for t in p['targets'])
    for metric in ('op_jobs{', 'op_jobs_oldest_queued_seconds', 'op_jobs_failed_24h'):
        assert metric in exprs


def test_no_dashboard_query_selects_by_container_or_instance() -> None:
    offenders = []
    for path in sorted((MONITORING / 'dashboards').glob('*.json')):
        dashboard = json.loads(path.read_text())
        panels = [
            *dashboard['panels'],
            *(q for p in dashboard['panels'] for q in p.get('panels', [])),
        ]
        for panel in panels:
            offenders.extend(
                (path.name, panel.get('title'))
                for target in panel.get('targets', [])
                if re.search(r'\b(container|instance)\b', target.get('expr', ''))
            )
    assert not offenders, offenders
