"""Public hardening of the services this wave adds (compose `security_opt` and `cap_drop`).

Docs: https://docs.docker.com/reference/compose-file/services/#security_opt and #cap_drop.
Capability sets come from spike S0 (postgres and valkey) and from the images running as appuser
(the Python services need none).
"""

from __future__ import annotations

from typing import Any

import pytest

from compose._helpers import _services


PYTHON_SERVICES = (
    'db-migrate',
    'jobs-worker',
    'periodic-worker',
    'open-vocab-worker',
    'jobs-beat',
    'flower',
    'cluster-worker',
)
# Spike S0: the postgres entrypoint starts as root, chowns PGDATA, reads the mode-600 secret
# (DAC_OVERRIDE) and drops to postgres (SETUID, SETGID). Without CHOWN or DAC_OVERRIDE it does not start.
POSTGRES_CAPS = {'CHOWN', 'DAC_OVERRIDE', 'FOWNER', 'SETGID', 'SETUID'}
# Valkey's wrapper copies the conf as root (DAC_OVERRIDE, CHOWN) and the image drops to valkey.
VALKEY_CAPS = {'CHOWN', 'DAC_OVERRIDE', 'SETGID', 'SETUID'}


def _svc(name: str) -> dict[str, Any]:
    return _services()[name]


@pytest.mark.parametrize('name', ['postgres', 'valkey', *PYTHON_SERVICES])
def test_every_new_service_blocks_privilege_escalation(name: str) -> None:
    assert 'no-new-privileges:true' in _svc(name)['security_opt']
    assert not _svc(name).get('privileged')


@pytest.mark.parametrize('name', PYTHON_SERVICES)
def test_python_services_run_with_no_capabilities(name: str) -> None:
    svc = _svc(name)
    assert svc['cap_drop'] == ['ALL']
    assert not svc.get('cap_add')


@pytest.mark.parametrize(('name', 'caps'), [('postgres', POSTGRES_CAPS), ('valkey', VALKEY_CAPS)])
def test_postgres_and_valkey_keep_only_the_capabilities_the_spike_measured(
    name: str, caps: set[str]
) -> None:
    svc = _svc(name)
    assert svc['cap_drop'] == ['ALL']
    assert set(svc['cap_add']) == caps
