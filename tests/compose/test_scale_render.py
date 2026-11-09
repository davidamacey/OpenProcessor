"""D-54 / D-50: the stack renders scaled, and the beat stays one replica.

Docs: https://docs.docker.com/reference/compose-file/deploy/#replicas,
https://docs.docker.com/reference/cli/docker/compose/config/ (offline render: no build, no up).
"""

from __future__ import annotations

import shutil

import pytest

from compose._helpers import render


pytestmark = pytest.mark.skipif(shutil.which('docker') is None, reason='docker CLI not installed')


def test_api_and_jobs_worker_render_with_two_replicas() -> None:
    services = render({'OP_API_REPLICAS': '2', 'OP_JOBS_REPLICAS': '2'})
    for name in ('api', 'jobs-worker'):
        assert services[name]['deploy']['replicas'] == 2, name
        assert 'container_name' not in services[name], name


def test_the_beat_is_one_replica_whatever_the_scale_variables_say() -> None:
    services = render({'OP_API_REPLICAS': '2', 'OP_JOBS_REPLICAS': '2'})
    assert services['jobs-beat']['deploy']['replicas'] == 1


def test_the_default_render_is_one_replica_of_each() -> None:
    services = render({})
    assert services['api']['deploy']['replicas'] == 1
    assert services['jobs-worker']['deploy']['replicas'] == 1
