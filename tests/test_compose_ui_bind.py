"""OP_UI_BIND_ADDRESS opens only the human-facing UIs (the four monitoring UIs and Cropwright).

Rendered with ``docker compose config`` so the real interpolation is
exercised, not the raw YAML text.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_SERVICES = {'grafana', 'prometheus', 'opensearch-dashboards', 'curation-mlflow', 'cropwright'}

pytestmark = pytest.mark.skipif(shutil.which('docker') is None, reason='docker CLI not installed')


def _hosts(tmp_path: Path, env: dict[str, str]) -> dict[str, set[str]]:
    env_file = tmp_path / 'env'
    env_file.write_text(''.join(f'{k}={v}\n' for k, v in env.items()))
    out = subprocess.run(
        [
            'docker',
            'compose',
            '--env-file',
            str(env_file),
            '-f',
            str(REPO_ROOT / 'docker-compose.yml'),
            '--profile',
            '*',
            'config',
            '--format',
            'json',
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': str(tmp_path)},
    ).stdout
    services = json.loads(out)['services']
    return {
        name: {p.get('host_ip', '') for p in spec.get('ports') or []}
        for name, spec in services.items()
        if spec.get('ports')
    }


def test_default_binds_everything_to_loopback(tmp_path: Path) -> None:
    hosts = _hosts(tmp_path, {})
    assert hosts.keys() >= UI_SERVICES
    assert all(h == {'127.0.0.1'} for h in hosts.values()), hosts


def test_ui_variable_moves_only_the_uis(tmp_path: Path) -> None:
    hosts = _hosts(tmp_path, {'OP_UI_BIND_ADDRESS': '0.0.0.0'})
    for name, h in hosts.items():
        expected = {'0.0.0.0'} if name in UI_SERVICES else {'127.0.0.1'}
        assert h == expected, (name, h)


def test_ui_variable_defaults_to_bind_address(tmp_path: Path) -> None:
    hosts = _hosts(tmp_path, {'OP_BIND_ADDRESS': '10.1.2.3'})
    assert all(h == {'10.1.2.3'} for h in hosts.values()), hosts


def test_cropwright_variable_moves_only_cropwright(tmp_path: Path) -> None:
    """The installer opens just the web UI to a LAN (homelab default); the API, OpenSearch and the
    monitoring UIs stay on loopback."""
    hosts = _hosts(tmp_path, {'CROPWRIGHT_BIND_ADDRESS': '0.0.0.0'})
    for name, h in hosts.items():
        expected = {'0.0.0.0'} if name == 'cropwright' else {'127.0.0.1'}
        assert h == expected, (name, h)


def test_cropwright_variable_wins_over_the_ui_variable(tmp_path: Path) -> None:
    hosts = _hosts(tmp_path, {'OP_UI_BIND_ADDRESS': '0.0.0.0', 'CROPWRIGHT_BIND_ADDRESS': '127.0.0.1'})
    assert hosts['cropwright'] == {'127.0.0.1'}
    assert hosts['grafana'] == {'0.0.0.0'}
