"""Pin which services start by default, their profiles, command targets and healthchecks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compose._helpers import REPO_ROOT, _services


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

# the monitoring stack used to
# start on a bare `docker compose up -d` alongside the core services --
# a surprise on a shared host (alloy mounts /var/run/docker.sock and tails
# EVERY container on the host; dcgm-exporter reserves `count: all` GPUs).
_MONITORING_SERVICES = (
    'prometheus',
    'grafana',
    'loki',
    'alloy',
    'node-exporter',
    'dcgm-exporter',
    'opensearch-dashboards',
)


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


def test_monitoring_services_carry_monitoring_profile() -> None:
    services = _services()
    missing_profile: list[str] = []
    for name in _MONITORING_SERVICES:
        assert name in services, f'expected monitoring service {name!r} in docker-compose.yml'
        profiles = services[name].get('profiles') or []
        if 'monitoring' not in profiles:
            missing_profile.append(name)
    assert not missing_profile, (
        f'monitoring services missing profiles: [monitoring]: {missing_profile}'
    )


def test_monitoring_does_not_start_on_a_bare_compose_up() -> None:
    services = _services()
    default_services = {name for name, spec in services.items() if not spec.get('profiles')}
    for name in _MONITORING_SERVICES:
        assert name not in default_services, (
            f'{name} has no profiles set — it would start on a bare `docker compose up`, '
            'mounting docker.sock (alloy) or reserving all GPUs (dcgm-exporter) by surprise'
        )
    # opensearch itself (the vector DB, not opensearch-dashboards) IS core
    # and must still start by default.
    assert 'opensearch' in default_services


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


# Workers whose depends_on already names api directly (main compose
# doesn't have every curation worker depend on api — e.g.
# curation-detection-worker depends on triton-server/opensearch, and
# curation-auto-label-worker only on opensearch, both by design, since
# they're triggered via files, not a direct HTTP call at startup).
_WORKERS_DEPENDING_ON_API = ('curation-vlm-worker', 'curation-cluster-refresh')


def test_curation_workers_depend_on_healthy_api() -> None:
    """S-7: workers that depend on `api` wait for it to report healthy,
    not merely started."""
    services = _services()
    bad: list[str] = []
    for name in _WORKERS_DEPENDING_ON_API:
        depends_on = services[name].get('depends_on')
        if not isinstance(depends_on, dict) or 'api' not in depends_on:
            bad.append(name)
            continue
        condition = (depends_on.get('api') or {}).get('condition')
        if condition != 'service_healthy':
            bad.append(name)
    assert not bad, f'expected depends_on.api.condition == service_healthy: {bad}'


def test_api_waits_for_a_healthy_opensearch() -> None:
    """Found live: the API boots, fails to create the core kNN indexes
    because OpenSearch is not up yet, and never retries them -- so the first
    /ingest creates them with dynamic mappings."""
    depends_on = _services()['api'].get('depends_on')
    assert isinstance(depends_on, dict), depends_on
    assert (depends_on.get('opensearch') or {}).get('condition') == 'service_healthy'


def test_api_has_a_healthcheck() -> None:
    """A `service_healthy` dependency on api is meaningless without one."""
    services = _services()
    assert services['api'].get('healthcheck'), 'api needs a healthcheck'


def test_curation_mlflow_has_a_healthcheck() -> None:
    services = _services()
    assert services['curation-mlflow'].get('healthcheck'), 'curation-mlflow needs a healthcheck'
