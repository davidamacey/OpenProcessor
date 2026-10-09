"""Pin the Grafana Alloy log-shipper config's container filters.

History: the first filters matched container names (``/triton-server.*``),
which never matched this compose's ``${COMPOSE_PROJECT_NAME}-triton``
names; the Wave 0 fix matched the ``-triton`` / ``-api`` suffix, which
also matched every other OpenProcessor stack on the host, so one install
collected another's logs (installer acceptance item K-4).

The filter is now the ``com.docker.compose.project`` label compared with
``OP_LOG_PROJECT`` (docker-compose.yml sets it from COMPOSE_PROJECT_NAME),
then the ``com.docker.compose.service`` label. This test parses the
relabel blocks (no HCL dependency: the file is flat), simulates Alloy's
keep rules (fully anchored RE2) against containers of this and other
stacks, and checks the rendered compose config wires the variable.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOY_CONFIG_PATH = REPO_ROOT / 'monitoring' / 'alloy-config.alloy'

_BLOCK_RE = re.compile(r'discovery\.relabel\s+"(?P<name>[^"]+)"\s*\{(?P<body>.*?)\n\}', re.DOTALL)
_RULE_RE = re.compile(r'rule\s*\{(?P<body>.*?)\}', re.DOTALL)
_PROJECT_LABEL = '__meta_docker_container_label_com_docker_compose_project'
_SERVICE_LABEL = '__meta_docker_container_label_com_docker_compose_service'


def _read_config() -> str:
    return ALLOY_CONFIG_PATH.read_text(encoding='utf-8')


def test_config_braces_are_balanced() -> None:
    depth = 0
    for i, ch in enumerate(_read_config()):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            assert depth >= 0, f"unbalanced '}}' at offset {i}"
    assert depth == 0


def _keep_rules(block: str) -> list[tuple[str, str]]:
    """(source_label, regex) of every `action = "keep"` rule; a regex given
    as sys.env("X") is returned as the placeholder ENV:X."""
    rules = []
    for m in _RULE_RE.finditer(block):
        body = m.group('body')
        if 'action' not in body or '"keep"' not in body:
            continue
        label = re.search(r'source_labels\s*=\s*\["([^"]+)"\]', body).group(1)  # type: ignore[union-attr]
        env = re.search(r'regex\s*=\s*sys\.env\("([A-Z_]+)"\)', body)
        lit = re.search(r'regex\s*=\s*"([^"]+)"', body)
        rules.append((label, f'ENV:{env.group(1)}' if env else lit.group(1)))  # type: ignore[union-attr]
    return rules


def _blocks() -> dict[str, str]:
    blocks = {m.group('name'): m.group('body') for m in _BLOCK_RE.finditer(_read_config())}
    assert set(blocks) == {'triton', 'fastapi', 'workers'}, blocks
    return blocks


def _kept(block: str, labels: dict[str, str], env: dict[str, str]) -> bool:
    for label, regex in _keep_rules(block):
        pattern = env[regex[4:]] if regex.startswith('ENV:') else regex
        if not re.fullmatch(pattern, labels.get(label, '')):
            return False
    return True


def _container(project: str, service: str) -> dict[str, str]:
    return {
        _PROJECT_LABEL: project,
        _SERVICE_LABEL: service,
        '__meta_docker_container_name': f'/{project}-{"triton" if service == "triton-server" else "api"}',
    }


@pytest.mark.parametrize('project', ['openprocessor', 'opinst-w0', 'op_fresh2'])
def test_only_this_projects_triton_and_api_are_kept(project: str) -> None:
    blocks = _blocks()
    env = {'OP_LOG_PROJECT': project}
    assert _kept(blocks['triton'], _container(project, 'triton-server'), env)
    assert _kept(blocks['fastapi'], _container(project, 'api'), env)
    for other in ('openprocessor', 'opfinal', 'opinst-other'):
        if other == project:
            continue
        # Same suffixes, another stack: must not be collected.
        assert not _kept(blocks['triton'], _container(other, 'triton-server'), env)
        assert not _kept(blocks['fastapi'], _container(other, 'api'), env)
    assert not _kept(blocks['triton'], _container(project, 'api'), env)
    assert not _kept(blocks['fastapi'], _container(project, 'triton-server'), env)
    assert not _kept(blocks['fastapi'], _container(project, 'curation-detection-worker'), env)


_WORKER_SERVICES = (
    'curation-detection-worker',
    'curation-vlm-worker',
    'curation-auto-label-worker',
    'segmenter',
)


@pytest.mark.parametrize('project', ['openprocessor', 'opinst-w0'])
def test_worker_and_segmenter_logs_are_kept_for_this_project_only(project: str) -> None:
    block = _blocks()['workers']
    env = {'OP_LOG_PROJECT': project}
    for service in _WORKER_SERVICES:
        assert _kept(block, _container(project, service), env), service
        assert not _kept(block, _container('opfinal', service), env), service
    for service in ('api', 'triton-server', 'opensearch', 'curation-cluster-refresh'):
        assert not _kept(block, _container(project, service), env), service
    # fullmatch: a look-alike service name must not slip through
    assert not _kept(block, _container(project, 'segmenter-extra'), env)


def test_worker_streams_get_their_own_job_label() -> None:
    assert 'replacement  = "worker"' in _read_config()


def test_every_block_is_scoped_by_the_project_label_not_the_name() -> None:
    for name, block in _blocks().items():
        rules = _keep_rules(block)
        assert (_PROJECT_LABEL, 'ENV:OP_LOG_PROJECT') in rules, name
        assert all(label != '__meta_docker_container_name' for label, _ in rules), name


def test_compose_passes_the_project_to_alloy() -> None:
    compose = (REPO_ROOT / 'docker-compose.yml').read_text()
    alloy = compose[compose.index('\n  alloy:\n') :]
    alloy = alloy[: alloy.index('\n  # ====')]
    assert '- OP_LOG_PROJECT=${COMPOSE_PROJECT_NAME:-openprocessor}' in alloy


def test_rendered_compose_config_sets_the_alloy_project(tmp_path: Path) -> None:
    docker = shutil.which('docker') or ''
    if not docker:
        pytest.skip('docker CLI not installed')
    shutil.copy(REPO_ROOT / 'docker-compose.yml', tmp_path / 'docker-compose.yml')
    (tmp_path / '.env').write_text(
        'COMPOSE_PROJECT_NAME=opinst-test\nCOMPOSE_PROFILES=monitoring\n'
    )
    result = subprocess.run(
        [
            docker,
            'compose',
            '-p',
            'opinst-test',
            '--env-file',
            str(tmp_path / '.env'),
            '--project-directory',
            str(tmp_path),
            '-f',
            str(tmp_path / 'docker-compose.yml'),
            'config',
            '--format',
            'json',
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f'docker compose config unavailable: {result.stderr[:200]}')
    alloy = json.loads(result.stdout)['services']['alloy']
    assert alloy['environment']['OP_LOG_PROJECT'] == 'opinst-test'
    blocks = _blocks()
    env = {'OP_LOG_PROJECT': alloy['environment']['OP_LOG_PROJECT']}
    assert _kept(blocks['triton'], _container('opinst-test', 'triton-server'), env)
    assert not _kept(blocks['triton'], _container('openprocessor', 'triton-server'), env)


def test_prometheus_scrapes_each_worker_on_the_port_it_serves() -> None:
    import yaml

    jobs = {
        j['job_name']: j['static_configs'][0]['targets'][0]
        for j in yaml.safe_load((REPO_ROOT / 'monitoring' / 'prometheus.yml').read_text())[
            'scrape_configs'
        ]
    }
    compose = (REPO_ROOT / 'docker-compose.yml').read_text()
    expected = {
        'curation-detection-worker': (
            '4609',
            'OP_REGION_WORKER_METRICS_PORT',
            'scripts/curation/worker/runner.py',
        ),
        'curation-vlm-worker': (
            '4610',
            'OP_VLM_WORKER_METRICS_PORT',
            'scripts/curation/vlm_worker.py',
        ),
        'curation-auto-label-worker': (
            '4611',
            'OP_AUTO_LABEL_WORKER_METRICS_PORT',
            'scripts/curation/auto_label_worker.py',
        ),
    }
    for service, (port, env_var, source) in expected.items():
        assert jobs[service] == f'{service}:{port}'
        assert f'\n  {service}:\n' in compose
        text = (REPO_ROOT / source).read_text()
        assert env_var in text
        assert port in text
