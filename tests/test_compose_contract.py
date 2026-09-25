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


def test_auto_label_worker_caps_blas_threads() -> None:
    """LG-3: AHC/UMAP on the residual pool otherwise spawns one BLAS thread
    per host core (48 on the reference host) with nothing else configured.
    Cap them on curation-auto-label-worker so a big recluster doesn't starve
    the rest of a shared host."""
    env = _services()['curation-auto-label-worker'].get('environment') or []
    env_str = '\n'.join(str(e) for e in env)
    assert 'OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}' in env_str
    assert 'OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}' in env_str
    assert 'MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}' in env_str


def test_segmenter_hf_cache_mounted_at_appuser_home() -> None:
    """ST-2: the segmenter container runs as uid 1000 (``appuser``) with
    ``HF_HOME=/home/appuser/.cache/huggingface``. A cache bind at
    ``/root/.cache/huggingface`` silently misses (wrong user), so the ~3.3G
    of gated SAM3 weights land in the writable container layer and
    re-download on every recreate instead of hitting the nvme bind."""
    mounts = [str(v) for v in (_services()['segmenter'].get('volumes') or [])]
    assert any(m.endswith(':/home/appuser/.cache/huggingface') for m in mounts), mounts
    assert not any(m.endswith(':/root/.cache/huggingface') for m in mounts), mounts
