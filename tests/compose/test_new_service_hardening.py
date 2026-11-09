"""Public hardening of the services this wave adds (compose `security_opt` and `cap_drop`).

Docs: https://docs.docker.com/reference/compose-file/services/#security_opt and #cap_drop.
Capability sets come from spike S0 (postgres and valkey) and from the images running as appuser
(the Python services need none).
"""

from __future__ import annotations

from typing import Any

import pytest

from compose._helpers import _services, render


PYTHON_SERVICES = (
    'db-migrate',
    'jobs-worker',
    'periodic-worker',
    'open-vocab-worker',
    'jobs-beat',
    'flower',
    'cluster-worker',
    'curation-auto-label-worker',
    'curation-assist-loop-worker',
)
# The GPU consumers keep the default capability set the NVIDIA runtime expects; they still get
# no-new-privileges.
GPU_CONSUMERS = ('curation-trainer', 'model-export-worker', 'curation-evaluator')
# Spike S0: the postgres entrypoint starts as root, chowns PGDATA, reads the mode-600 secret
# (DAC_OVERRIDE) and drops to postgres (SETUID, SETGID). Without CHOWN or DAC_OVERRIDE it does not start.
POSTGRES_CAPS = {'CHOWN', 'DAC_OVERRIDE', 'FOWNER', 'SETGID', 'SETUID'}


def _svc(name: str) -> dict[str, Any]:
    return _services()[name]


@pytest.mark.parametrize('name', ['postgres', 'valkey', *PYTHON_SERVICES, *GPU_CONSUMERS])
def test_every_new_service_blocks_privilege_escalation(name: str) -> None:
    assert 'no-new-privileges:true' in _svc(name)['security_opt']
    assert not _svc(name).get('privileged')


@pytest.mark.parametrize('name', PYTHON_SERVICES)
def test_python_services_run_with_no_capabilities(name: str) -> None:
    svc = _svc(name)
    assert svc['cap_drop'] == ['ALL']
    assert not svc.get('cap_add')


def test_postgres_keeps_only_the_capabilities_the_spike_measured() -> None:
    svc = _svc('postgres')
    assert svc['cap_drop'] == ['ALL']
    assert set(svc['cap_add']) == POSTGRES_CAPS


def test_valkey_runs_as_a_plain_user_with_no_capability() -> None:
    svc = _svc('valkey')
    assert svc['cap_drop'] == ['ALL']
    assert not svc.get('cap_add')


@pytest.mark.parametrize('name', ['gpu-arbiter-worker', 'pgbouncer'])
def test_the_overlay_services_are_hardened_and_publish_no_port(name: str) -> None:
    svc = render(
        {}, 'docker-compose.yml', 'docker-compose.gpu-arbiter.yml', 'docker-compose.pgbouncer.yml'
    )[name]
    assert 'no-new-privileges:true' in svc['security_opt']
    assert svc['cap_drop'] == ['ALL']
    assert not svc.get('ports')


@pytest.mark.parametrize(
    'name', [n for n in PYTHON_SERVICES if n != 'flower'] + list(GPU_CONSUMERS)
)
def test_no_worker_publishes_a_port(name: str) -> None:
    assert not _svc(name).get('ports')
