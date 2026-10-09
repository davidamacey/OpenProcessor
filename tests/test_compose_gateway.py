"""Gateway mode (OP_GATEWAY_SUBPATHS) in docker-compose.yml.

Rendered with ``docker compose config`` so the real interpolation is exercised,
and the wrapper script each UI service starts through is extracted from the
rendered config and RUN (``sh -c <script> gateway-wrap <tag> <command...>`` with
``env`` or ``echo`` as the command), so what it exports, appends and refuses is
tested, not its text.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_SERVICES = ('grafana', 'prometheus', 'opensearch-dashboards', 'curation-mlflow')
TAGS = {
    'grafana': 'grafana',
    'prometheus': 'prometheus',
    'opensearch-dashboards': 'dashboards',
    'curation-mlflow': 'mlflow',
}

pytestmark = pytest.mark.skipif(shutil.which('docker') is None, reason='docker CLI not installed')


def _services(tmp_path: Path, env: dict[str, str]) -> dict[str, dict]:
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
            'monitoring',
            '--profile',
            'training',
            '--profile',
            'cropwright',
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
    return json.loads(out)['services']


def _wrap(
    services: dict[str, dict], name: str, command: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run service ``name``'s wrapper with ``command`` standing in for the real process."""
    entrypoint = services[name]['entrypoint']
    assert entrypoint[:2] == ['/bin/sh', '-c'], entrypoint
    script, tag = entrypoint[2].replace('$$', '$'), entrypoint[4]  # config re-escapes $
    assert tag == TAGS[name]
    container_env = {k: v for k, v in services[name]['environment'].items() if k.startswith('OP_')}
    # The service's own real entrypoint args (entrypoint[5:]) are replaced by `command`.
    return subprocess.run(
        ['/bin/sh', '-c', script, 'gateway-wrap', tag, *command],
        check=False,
        capture_output=True,
        text=True,
        env={'PATH': '/usr/bin:/bin', **container_env, **env},
    )


def test_gateway_switch_defaults_off_everywhere(tmp_path: Path) -> None:
    services = _services(tmp_path, {})
    for name in (*UI_SERVICES, 'cropwright', 'yolo-api'):
        assert services[name]['environment']['OP_GATEWAY_SUBPATHS'] == 'false', name


def test_gateway_switch_reaches_the_api_the_gateway_and_each_ui(tmp_path: Path) -> None:
    services = _services(tmp_path, {'OP_GATEWAY_SUBPATHS': 'true'})
    for name in (*UI_SERVICES, 'cropwright', 'yolo-api'):
        assert services[name]['environment']['OP_GATEWAY_SUBPATHS'] == 'true', name


def test_ui_ports_stay_on_loopback_in_gateway_mode(tmp_path: Path) -> None:
    services = _services(
        tmp_path, {'OP_GATEWAY_SUBPATHS': 'true', 'CROPWRIGHT_BIND_ADDRESS': '0.0.0.0'}
    )
    for name in UI_SERVICES:
        assert {p['host_ip'] for p in services[name]['ports']} == {'127.0.0.1'}, name
    assert {p['host_ip'] for p in services['cropwright']['ports']} == {'0.0.0.0'}


def test_every_ui_starts_through_the_wrapper_with_its_real_entrypoint(tmp_path: Path) -> None:
    services = _services(tmp_path, {})
    real = {
        'grafana': ['/run.sh'],
        'prometheus': ['/bin/prometheus'],
        'opensearch-dashboards': ['./opensearch-dashboards-docker-entrypoint.sh'],
        'curation-mlflow': [],
    }
    for name, tail in real.items():
        assert services[name]['entrypoint'][5:] == tail, name
    assert services['curation-mlflow']['command'][:2] == ['mlflow', 'server']
    assert services['opensearch-dashboards']['command'] == ['opensearch-dashboards']


@pytest.mark.parametrize('off', ['', 'false', 'FALSE', '0', 'no', 'off'])
def test_off_is_a_plain_exec_with_no_added_flags(tmp_path: Path, off: str) -> None:
    services = _services(tmp_path, {})
    for name in UI_SERVICES:
        r = _wrap(services, name, ['echo', 'ARGS'], {'OP_GATEWAY_SUBPATHS': off})
        assert (r.returncode, r.stdout.strip()) == (0, 'ARGS'), (name, r.stderr)


@pytest.mark.parametrize('on', ['true', 'TRUE', '1', 'yes', 'on'])
def test_on_adds_each_uis_sub_path_settings(tmp_path: Path, on: str) -> None:
    services = _services(tmp_path, {})
    gw = {'OP_GATEWAY_SUBPATHS': on, 'OP_UI_PUBLISH_ADDRESS': '127.0.0.1'}

    prom = _wrap(services, 'prometheus', ['echo', 'ARGS'], gw)
    assert (
        prom.stdout.strip()
        == 'ARGS --web.external-url=/prometheus/ --web.route-prefix=/prometheus/'
    )

    # MLflow stays rooted (--static-prefix would also prefix /health and /api,
    # breaking the in-network MLFLOW_TRACKING_URI); the gateway strips /mlflow.
    mlflow = _wrap(services, 'curation-mlflow', ['echo', 'ARGS'], gw)
    assert (mlflow.returncode, mlflow.stdout.strip()) == (0, 'ARGS')

    graf = _wrap(services, 'grafana', ['env'], {**gw, 'GF_SECURITY_ADMIN_PASSWORD': 'not-admin'})
    assert graf.returncode == 0, graf.stderr
    assert 'GF_SERVER_SERVE_FROM_SUB_PATH=true' in graf.stdout.splitlines()
    assert 'GF_SERVER_ROOT_URL=%(protocol)s://%(domain)s/grafana/' in graf.stdout.splitlines()

    osd = _wrap(services, 'opensearch-dashboards', ['env'], gw)
    assert osd.returncode == 0, osd.stderr
    assert {'SERVER_BASEPATH=/dashboards', 'SERVER_REWRITEBASEPATH=true'} <= set(
        osd.stdout.splitlines()
    )


def test_off_does_not_export_the_sub_path_settings(tmp_path: Path) -> None:
    services = _services(tmp_path, {})
    graf = _wrap(services, 'grafana', ['env'], {'OP_GATEWAY_SUBPATHS': 'false'})
    assert 'GF_SERVER_SERVE_FROM_SUB_PATH' not in graf.stdout
    osd = _wrap(services, 'opensearch-dashboards', ['env'], {'OP_GATEWAY_SUBPATHS': 'false'})
    assert 'SERVER_BASEPATH' not in osd.stdout


@pytest.mark.parametrize('password', ['', 'admin'])
def test_grafana_refuses_the_default_or_empty_password_in_gateway_mode(
    tmp_path: Path, password: str
) -> None:
    services = _services(tmp_path, {})
    env = {
        'OP_GATEWAY_SUBPATHS': 'true',
        'OP_UI_PUBLISH_ADDRESS': '127.0.0.1',
        'GF_SECURITY_ADMIN_PASSWORD': password,
    }
    r = _wrap(services, 'grafana', ['echo', 'STARTED'], env)
    assert r.returncode != 0
    assert 'STARTED' not in r.stdout
    assert 'GF_SECURITY_ADMIN_PASSWORD' in r.stderr


def test_grafana_default_password_is_still_allowed_without_the_gateway(tmp_path: Path) -> None:
    services = _services(tmp_path, {})
    env = {'OP_GATEWAY_SUBPATHS': 'false', 'GF_SECURITY_ADMIN_PASSWORD': 'admin'}
    assert _wrap(services, 'grafana', ['echo', 'STARTED'], env).stdout.strip() == 'STARTED'


@pytest.mark.parametrize('address', ['0.0.0.0', '192.168.1.5', '10.1.2.3', ''])
def test_every_ui_refuses_to_start_published_beyond_loopback_in_gateway_mode(
    tmp_path: Path, address: str
) -> None:
    services = _services(tmp_path, {})
    for name in UI_SERVICES:
        env = {
            'OP_GATEWAY_SUBPATHS': 'true',
            'OP_UI_PUBLISH_ADDRESS': address,
            'GF_SECURITY_ADMIN_PASSWORD': 'not-admin',
        }
        r = _wrap(services, name, ['echo', 'STARTED'], env)
        assert r.returncode != 0, name
        assert 'STARTED' not in r.stdout, name
        assert 'OP_UI_BIND_ADDRESS' in r.stderr, name


def test_the_wrapper_receives_the_publish_address_from_the_bind_variables(tmp_path: Path) -> None:
    for env, expected in (
        ({}, '127.0.0.1'),
        ({'OP_BIND_ADDRESS': '10.1.2.3'}, '10.1.2.3'),
        ({'OP_BIND_ADDRESS': '10.1.2.3', 'OP_UI_BIND_ADDRESS': '127.0.0.1'}, '127.0.0.1'),
    ):
        services = _services(tmp_path, env)
        for name in UI_SERVICES:
            assert services[name]['environment']['OP_UI_PUBLISH_ADDRESS'] == expected, name


def test_an_unparseable_switch_is_refused(tmp_path: Path) -> None:
    services = _services(tmp_path, {})
    r = _wrap(services, 'prometheus', ['echo', 'STARTED'], {'OP_GATEWAY_SUBPATHS': 'maybe'})
    assert r.returncode != 0
    assert 'STARTED' not in r.stdout
