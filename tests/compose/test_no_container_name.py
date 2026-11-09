"""D-54: no service of any compose file sets container_name, so every service can be scaled.

Docs: https://docs.docker.com/reference/compose-file/services/#container_name (a custom name
prevents scaling), https://docs.docker.com/reference/compose-file/services/#labels.
"""

from __future__ import annotations

import pytest

from compose._helpers import REPO_ROOT, _load_yaml


FILES = sorted(
    p for p in [*REPO_ROOT.glob('docker-compose*.yml'), *REPO_ROOT.glob('docker/test/*.yml')]
)


@pytest.mark.parametrize('path', FILES, ids=lambda p: p.name)
def test_no_service_has_a_container_name(path) -> None:
    services = _load_yaml(path).get('services') or {}
    named = sorted(n for n, s in services.items() if 'container_name' in s)
    assert not named, f'{path.name}: {named}'
