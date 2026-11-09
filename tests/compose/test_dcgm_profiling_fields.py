"""6B-8 / S-22: the monitoring profile's DCGM exporter reads a field list with the profiling fields.

Docs: https://github.com/NVIDIA/dcgm-exporter (custom counters CSV, DCGM_EXPORTER_COLLECTORS).
"""

from __future__ import annotations

import json
import re

from compose._helpers import REPO_ROOT, _services, render


PROFILING = (
    'DCGM_FI_PROF_SM_ACTIVE',
    'DCGM_FI_PROF_SM_OCCUPANCY',
    'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE',
    'DCGM_FI_PROF_DRAM_ACTIVE',
    'DCGM_FI_PROF_PCIE_TX_BYTES',
    'DCGM_FI_PROF_PCIE_RX_BYTES',
)
CSV = REPO_ROOT / 'monitoring' / 'dcgm-counters.csv'


def _fields() -> list[str]:
    return [
        line.split(',')[0].strip()
        for line in CSV.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]


def test_the_field_list_has_the_profiling_fields() -> None:
    assert set(PROFILING) <= set(_fields())


def test_every_field_a_dashboard_or_alert_reads_is_still_exported() -> None:
    text = ' '.join(
        p.read_text()
        for p in [
            *(REPO_ROOT / 'monitoring' / 'dashboards').glob('*.json'),
            *(REPO_ROOT / 'monitoring').glob('*.yml'),
            *(REPO_ROOT / 'monitoring' / 'alerts').glob('*.yml'),
        ]
    )
    used = set(re.findall(r'DCGM_FI_(?:DEV|PROF)_[A-Z_0-9]+', text))
    assert used <= set(_fields()), sorted(used - set(_fields()))


def test_the_exporter_mounts_and_reads_the_list() -> None:
    service = render({})['dcgm-exporter']
    assert service['environment']['DCGM_EXPORTER_COLLECTORS'] == '/etc/dcgm-exporter/counters.csv'
    assert any(
        v['target'] == '/etc/dcgm-exporter/counters.csv' and v.get('read_only')
        for v in service['volumes']
    )
    assert service['profiles'] == ['monitoring']


def test_the_gpu_dashboard_shows_them_beside_utilization() -> None:
    dashboard = json.loads(
        (REPO_ROOT / 'monitoring' / 'dashboards' / 'gpu-metrics-dashboard.json').read_text()
    )
    exprs = ' '.join(t['expr'] for p in dashboard['panels'] for t in p.get('targets', []))
    assert all(field in exprs for field in PROFILING)
    assert 'DCGM_FI_DEV_GPU_UTIL' in exprs


def test_the_service_is_defined_once_in_the_base_file() -> None:
    assert 'dcgm-exporter' in _services()
