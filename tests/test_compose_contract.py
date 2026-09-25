"""Pin the ``docker-compose.yml`` service contract.

The compose analogue of ``tests/curation/test_precommit_paths.py``:
convert silent compose drift into a loud test failure instead of a
3am "why won't the container start" surprise.

Covers:

(a) every service's ``command:`` that invokes a repo-relative Python
    script or ``-m`` module points at a file that actually exists;
(b) every curation worker service carries ``profiles: [curation]``, so
    the default ``docker compose up`` experience never starts it;
(c) no two services share a ``container_name`` or a host port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPO_ROOT / 'docker-compose.yml'

# The four curation worker services. Any
# service whose name starts with this prefix must also carry
# `profiles: [curation]` — see the loop in
# test_curation_services_carry_curation_profile for the general form.
_CURATION_WORKER_SERVICES = (
    'curation-detection-worker',
    'curation-vlm-worker',
    'curation-auto-label-worker',
    'curation-cluster-refresh',
)


def _load_compose() -> dict[str, Any]:
    with COMPOSE_PATH.open() as fh:
        return yaml.safe_load(fh)


def _services() -> dict[str, dict[str, Any]]:
    return _load_compose()['services']


def _repo_relative_command_targets(command: Any) -> list[str]:
    """Pull repo-relative Python script/module paths out of a `command:`.

    Recognizes the two forms this compose file's Python services use:
    ``python -m dotted.module.path [args...]`` and
    ``python path/to/script.py [args...]``. A bare list or a shell
    string form (`"cmd arg arg"`) are both handled. Anything else
    (uvicorn's ``src.main:app`` ASGI string, tritonserver/prometheus/
    grafana flags, binary entrypoints) is not a repo Python path
    invocation and yields nothing.
    """
    if command is None:
        return []
    tokens = command.split() if isinstance(command, str) else [str(t) for t in command]
    if not tokens:
        return []
    exe = Path(tokens[0]).name
    if not exe.startswith('python'):
        return []
    rest = tokens[1:]
    if not rest:
        return []
    if rest[0] == '-m' and len(rest) > 1:
        module = rest[1]
        return [module.replace('.', '/')]
    if rest[0].endswith('.py'):
        return [rest[0]]
    return []


def _module_or_script_exists(repo_relative: str) -> bool:
    """A ``-m`` target may resolve to ``<path>.py`` or a package's
    ``<path>/__main__.py``; a direct script target is checked as-is.
    """
    as_file = REPO_ROOT / f'{repo_relative}.py'
    as_package_main = REPO_ROOT / repo_relative / '__main__.py'
    as_literal = REPO_ROOT / repo_relative
    return as_file.is_file() or as_package_main.is_file() or as_literal.is_file()


def test_command_script_and_module_paths_exist_on_disk() -> None:
    services = _services()
    missing: list[str] = []
    checked = 0
    for name, spec in services.items():
        for target in _repo_relative_command_targets(spec.get('command')):
            checked += 1
            if not _module_or_script_exists(target):
                missing.append(f'{name}: {target!r}')
    assert checked > 0, 'expected at least one python/-m command in docker-compose.yml'
    assert not missing, 'command targets that do not exist on disk:\n' + '\n'.join(missing)


def test_curation_worker_services_carry_curation_profile() -> None:
    services = _services()
    missing_profile: list[str] = []
    for name in _CURATION_WORKER_SERVICES:
        assert name in services, f'expected curation worker service {name!r} in docker-compose.yml'
        profiles = services[name].get('profiles') or []
        if 'curation' not in profiles:
            missing_profile.append(name)
    assert not missing_profile, (
        f'curation worker services missing profiles: [curation]: {missing_profile}'
    )


def test_default_compose_up_service_set_is_unchanged() -> None:
    """Nothing carrying `profiles: [curation]` starts on a bare `up`."""
    services = _services()
    default_services = {name for name, spec in services.items() if not spec.get('profiles')}
    for name in _CURATION_WORKER_SERVICES:
        assert name not in default_services, (
            f'{name} has no profiles set — it would start on a bare `docker compose up`'
        )


def test_no_duplicate_container_names() -> None:
    services = _services()
    names = [spec['container_name'] for spec in services.values() if spec.get('container_name')]
    dupes = {n for n in names if names.count(n) > 1}
    assert not dupes, f'duplicate container_name values: {dupes}'


def _published_ports(spec: dict[str, Any]) -> list[str]:
    published: list[str] = []
    for entry in spec.get('ports') or []:
        if isinstance(entry, dict):
            published.append(str(entry.get('published')))
            continue
        # Short syntax: "HOST:CONTAINER" or "HOST:CONTAINER/proto" or a bare port.
        host_part = str(entry).split(':')[0] if ':' in str(entry) else str(entry)
        published.append(host_part)
    return published


def test_no_duplicate_host_ports() -> None:
    services = _services()
    all_ports: list[tuple[str, str]] = []
    for name, spec in services.items():
        all_ports.extend((name, port) for port in _published_ports(spec))
    seen: dict[str, str] = {}
    dupes: list[str] = []
    for name, port in all_ports:
        if port in seen and seen[port] != name:
            dupes.append(f'{port} used by both {seen[port]!r} and {name!r}')
        else:
            seen[port] = name
    assert not dupes, 'duplicate host ports:\n' + '\n'.join(dupes)


# =============================================================================
# Build identity -- OP_BUILD_SHA baked in at build time so
# code_versions.api_sha/trainer_sha reflect the actual built commit rather
# than a dev checkout's live `git rev-parse HEAD` fallback.
# =============================================================================

_BUILD_SHA_SERVICES = ('yolo-api', 'curation-trainer', 'curation-evaluator')

_BUILD_SHA_DOCKERFILES = (
    REPO_ROOT / 'Dockerfile',
    REPO_ROOT / 'docker' / 'trainer' / 'Dockerfile',
    REPO_ROOT / 'docker' / 'evaluator' / 'Dockerfile',
)


def test_build_sha_passed_as_a_build_arg_on_core_services() -> None:
    services = _services()
    missing: list[str] = []
    for name in _BUILD_SHA_SERVICES:
        assert name in services, f'expected service {name!r} in docker-compose.yml'
        args = (services[name].get('build') or {}).get('args') or {}
        if 'OP_BUILD_SHA' not in args:
            missing.append(name)
    assert not missing, f'services missing build.args.OP_BUILD_SHA: {missing}'


def test_build_sha_baked_into_the_final_stage_of_every_dockerfile() -> None:
    for path in _BUILD_SHA_DOCKERFILES:
        lines = path.read_text(encoding='utf-8').splitlines()
        from_indices = [i for i, line in enumerate(lines) if line.startswith('FROM ')]
        assert from_indices, f'{path}: no FROM instruction found'
        tail = '\n'.join(lines[from_indices[-1] :])
        assert 'ARG OP_BUILD_SHA' in tail, f'{path}: missing ARG OP_BUILD_SHA after final FROM'
        assert 'ENV OP_BUILD_SHA=${OP_BUILD_SHA}' in tail, f'{path}: missing ENV OP_BUILD_SHA'
        assert 'org.opencontainers.image.revision=${OP_BUILD_SHA}' in tail, (
            f'{path}: missing revision LABEL binding OP_BUILD_SHA'
        )


def test_evaluator_sees_exports_at_the_api_path() -> None:
    """Bake-off eval datasets are exports: the evaluator must read them at the
    path the API resolved (default ``OP_EXPORT_ROOT=./data/exports`` -> ``/app/data``)."""
    services = _services()
    api_mounts = services['yolo-api'].get('volumes') or []
    evaluator_mounts = services['curation-evaluator'].get('volumes') or []
    assert any(str(v).startswith('./data:/app/data') for v in api_mounts)
    assert './data:/app/data:ro' in evaluator_mounts


# =============================================================================
# S-2 — heartbeat-based worker healthchecks (worker_liveness.py) replace
# `pgrep -f <module>`, which can't see a deadlocked-but-alive event loop.
# =============================================================================

# name passed to `python -m src.services.curation.worker_liveness check <name>`
# for each service, matching the names each worker's heartbeat_loop()/
# write_heartbeat() call uses.
_WORKER_LIVENESS_NAMES = {
    'curation-detection-worker': 'detection_worker',
    'curation-vlm-worker': 'vlm_worker',
    'curation-auto-label-worker': 'auto_label_worker',
    'curation-cluster-refresh': 'cluster_refresh',
}


def test_curation_worker_healthchecks_use_liveness() -> None:
    services = _services()
    bad: list[str] = []
    for name, liveness_name in _WORKER_LIVENESS_NAMES.items():
        assert name in services, f'expected service {name!r} in docker-compose.yml'
        test = (services[name].get('healthcheck') or {}).get('test')
        test_str = ' '.join(test) if isinstance(test, list) else str(test or '')
        if 'worker_liveness' not in test_str or liveness_name not in test_str:
            bad.append(f'{name}: {test_str!r}')
    assert not bad, 'expected worker_liveness-based healthchecks:\n' + '\n'.join(bad)


# Workers whose depends_on already names yolo-api directly (main compose
# doesn't have every curation worker depend on yolo-api — e.g.
# curation-detection-worker depends on triton-server/opensearch, and
# curation-auto-label-worker only on opensearch, both by design, since
# they're triggered via files, not a direct HTTP call at startup).
_WORKERS_DEPENDING_ON_API = ('curation-vlm-worker', 'curation-cluster-refresh')


def test_curation_workers_depend_on_healthy_api() -> None:
    """S-7: workers that depend on `yolo-api` wait for it to report healthy,
    not merely started."""
    services = _services()
    bad: list[str] = []
    for name in _WORKERS_DEPENDING_ON_API:
        depends_on = services[name].get('depends_on')
        if not isinstance(depends_on, dict) or 'yolo-api' not in depends_on:
            bad.append(name)
            continue
        condition = (depends_on.get('yolo-api') or {}).get('condition')
        if condition != 'service_healthy':
            bad.append(name)
    assert not bad, f'expected depends_on.yolo-api.condition == service_healthy: {bad}'


def test_yolo_api_has_a_healthcheck() -> None:
    """A `service_healthy` dependency on yolo-api is meaningless without one."""
    services = _services()
    assert services['yolo-api'].get('healthcheck'), 'yolo-api needs a healthcheck'


def test_curation_mlflow_has_a_healthcheck() -> None:
    services = _services()
    assert services['curation-mlflow'].get('healthcheck'), 'curation-mlflow needs a healthcheck'
