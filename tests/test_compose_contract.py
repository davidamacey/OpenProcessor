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

import re
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = REPO_ROOT / 'docker-compose.yml'
_NAME_LINE_RE = re.compile(r'^name:\s*(.+)$', re.MULTILINE)

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

# F-3 (fresh-start E2E findings 2026-09-25): the monitoring stack used to
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


def test_evaluator_gets_the_same_mlflow_tracking_url_as_the_trainer() -> None:
    """F-46 (fresh-start E2E findings 2026-09-25): the trainer gets
    MLFLOW_TRACKING_URI but the evaluator didn't, so bake-off runs always
    logged 'mlflow logging skipped: ... port=5000' against localhost
    instead of the curation-mlflow service."""
    services = _services()
    trainer_env = services['curation-trainer'].get('environment') or []
    evaluator_env = services['curation-evaluator'].get('environment') or []

    def _value(env: list[str], key: str) -> str | None:
        for entry in env:
            if entry.startswith(f'{key}='):
                return entry.split('=', 1)[1]
        return None

    trainer_url = _value(trainer_env, 'MLFLOW_TRACKING_URI')
    evaluator_url = _value(evaluator_env, 'MLFLOW_TRACKING_URI')
    assert trainer_url is not None
    assert evaluator_url == trainer_url


def test_api_evaluator_segmenter_gpu_ids_are_env_driven_not_hardcoded() -> None:
    """F-2 (fresh-start E2E findings 2026-09-25): yolo-api, curation-evaluator
    and segmenter used to hardcode device_ids: ['0'] regardless of
    TRITON_GPU_ID/VLM_GPU_ID, so a host whose free GPU wasn't 0 needed a
    compose edit to run them at all."""
    services = _services()
    expectations = {
        'yolo-api': 'API_GPU_ID',
        'curation-evaluator': 'EVALUATOR_GPU_ID',
        'segmenter': 'SEGMENTER_GPU_ID',
    }
    for service_name, env_var in expectations.items():
        devices = services[service_name]['deploy']['resources']['reservations']['devices']
        device_ids = devices[0]['device_ids']
        assert device_ids == [f'${{{env_var}:-0}}'], (
            f'{service_name}.deploy...device_ids should read {env_var}, got {device_ids}'
        )


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


def test_api_and_detection_worker_share_crop_cache() -> None:
    """ST-1: yolo-api and curation-detection-worker must mount the SAME
    named crop-cache volume at the SAME target, with OP_CROP_CACHE_DIR set
    to that target in both -- otherwise the worker's cache reads never see
    what the API's ingest path wrote."""
    services = _services()
    target = '/var/cache/openprocessor/crops'
    for name in ('yolo-api', 'curation-detection-worker'):
        mounts = [str(v) for v in (services[name].get('volumes') or [])]
        matching = [m for m in mounts if m.endswith(f':{target}')]
        assert matching, f'{name} has no crop-cache mount at {target}: {mounts}'
        volume_name = matching[0].split(':', 1)[0]
        assert volume_name == 'openprocessor-crop-cache', (name, matching[0])

        env = [str(e) for e in (services[name].get('environment') or [])]
        assert f'OP_CROP_CACHE_DIR={target}' in env, (name, env)

    top_level_volumes = _load_compose()['volumes']
    assert 'openprocessor-crop-cache' in top_level_volumes


def test_segmenter_hf_cache_mounted_at_appuser_home() -> None:
    """ST-2: the segmenter container runs as uid 1000 (``appuser``) with
    ``HF_HOME=/home/appuser/.cache/huggingface``. A cache bind at
    ``/root/.cache/huggingface`` silently misses (wrong user), so the ~3.3G
    of gated SAM3 weights land in the writable container layer and
    re-download on every recreate instead of hitting the nvme bind."""
    mounts = [str(v) for v in (_services()['segmenter'].get('volumes') or [])]
    assert any(m.endswith(':/home/appuser/.cache/huggingface') for m in mounts), mounts
    assert not any(m.endswith(':/root/.cache/huggingface') for m in mounts), mounts


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


# =============================================================================
# Fresh-start gaps batch B — compose/install portability (G-01/G-03/G-06/
# G-07/G-08/G-10/G-15/G-28).
# =============================================================================


def test_no_fixed_compose_project_name() -> None:
    """G-01: a hardcoded top-level `name:` forces every `docker compose`
    invocation onto the same project regardless of directory/-p/env,
    which is exactly how a fresh clone can end up operating on a
    different, already-running stack's containers and volumes."""
    raw = COMPOSE_PATH.read_text(encoding='utf-8')
    parsed = _load_compose()
    match = _NAME_LINE_RE.search(raw)
    assert match, 'expected a top-level `name:` key in docker-compose.yml'
    value = match.group(1).strip()
    assert '${COMPOSE_PROJECT_NAME' in value, (
        f'docker-compose.yml `name:` must be interpolated from COMPOSE_PROJECT_NAME, got: {value!r}'
    )
    assert 'services' in parsed  # sanity: file still parses as valid compose


def test_every_container_name_is_interpolated_from_project_name() -> None:
    """G-01: every container_name must move with COMPOSE_PROJECT_NAME, not
    just the ones someone remembered to update."""
    services = _services()
    bad = {
        name: spec['container_name']
        for name, spec in services.items()
        if spec.get('container_name')
        and '${COMPOSE_PROJECT_NAME' not in str(spec['container_name'])
    }
    assert not bad, f'container_name values not interpolated from COMPOSE_PROJECT_NAME: {bad}'


def test_no_hardcoded_published_host_ports() -> None:
    """G-03: every published host port must come from an env var (with a
    default matching today's value), not a bare literal -- otherwise
    env.template's port vars are decorative and a second stack on the same
    host collides on first `up`."""
    services = _services()
    bad: list[str] = []
    for name, spec in services.items():
        for entry in spec.get('ports') or []:
            if isinstance(entry, dict):
                published = entry.get('published')
                if published is not None and '${' not in str(published):
                    bad.append(f'{name}: {entry!r}')
                continue
            text = str(entry)
            host_part = text.split(':')[0] if ':' in text else text
            if host_part and '${' not in host_part:
                bad.append(f'{name}: {text!r}')
    assert not bad, 'published host ports not driven by an env var:\n' + '\n'.join(bad)


def test_source_root_mounted_at_the_same_path_on_api_and_detection_worker() -> None:
    """G-06: ingest resolves item paths against OP_SOURCE_ROOT on yolo-api;
    the detection worker re-reads the same items later. Without the same
    bind on both, every worker read fails with
    detection_failed/reason=image_unavailable."""
    services = _services()
    target = '/data/source'
    for name in ('yolo-api', 'curation-detection-worker', 'curation-auto-label-worker'):
        mounts = [str(v) for v in (services[name].get('volumes') or [])]
        matching = [m for m in mounts if m.endswith((f':{target}:ro', f':{target}'))]
        assert matching, f'{name} has no source-root mount at {target}: {mounts}'


def test_examples_mounted_on_api_and_detection_worker() -> None:
    """G-08: OP_REGION_PROFILE_PATH's worked example
    (examples/region_profiles/license_plate.json) only resolves inside the
    container if ./examples is actually mounted."""
    services = _services()
    target = '/app/examples'
    for name in ('yolo-api', 'curation-detection-worker', 'curation-auto-label-worker'):
        mounts = [str(v) for v in (services[name].get('volumes') or [])]
        assert any(m.endswith(f':{target}:ro') for m in mounts), (
            f'{name} has no ./examples mount at {target}: {mounts}'
        )


# F-33/F-34 (fresh-start E2E findings 2026-09-25): curation-auto-label-worker
# runs the same in-process pipeline (cascade_detect / profile_registry /
# source-image serving) as curation-detection-worker, not a thin HTTP client
# like curation-vlm-worker/curation-cluster-refresh -- it needs the same
# source-root/examples mounts or it fails lazily, on its first claimed job,
# with its idle healthcheck staying green the whole time. Unlike
# curation-detection-worker (which only ever produces unlabeled
# `*_proposal` detections), the auto-label worker's VLM stage additionally
# needs the class registry itself, so ./data is its own, narrower
# requirement -- see test_class_registry_data_mounted_where_needed below.
_IN_PROCESS_PIPELINE_SERVICES = (
    'yolo-api',
    'curation-detection-worker',
    'curation-auto-label-worker',
)
_HTTP_CLIENT_ONLY_CURATION_SERVICES = ('curation-vlm-worker', 'curation-cluster-refresh')


def test_class_registry_data_mounted_where_needed() -> None:
    """F-34b: OP_REGISTRY_PATH defaults to ./data/class_registry.json.
    Without ./data mounted, the registry loads empty and any class-aware
    stage (the VLM auto-label stage, training, promote, eval) raises
    'class_names or class_catalog must be supplied' or reads no classes
    instead of running. yolo-api and curation-auto-label-worker read the
    registry directly; curation-detection-worker deliberately does not
    (it only ever writes unlabeled `*_proposal` detections)."""
    services = _services()
    target = '/app/data'
    for name in ('yolo-api', 'curation-auto-label-worker'):
        mounts = [str(v) for v in (services[name].get('volumes') or [])]
        assert any(m.endswith((f':{target}', f':{target}:ro')) for m in mounts), (
            f'{name} has no ./data mount at {target}: {mounts}'
        )


def test_http_client_only_curation_services_stay_lightweight() -> None:
    """Documents *why* curation-vlm-worker/curation-cluster-refresh don't
    need the source-root/examples/data mounts above: they only ever talk
    to yolo-api over HTTP (verified via each entrypoint's own imports, not
    inferred from mounts), so adding a heavier mount surface there would be
    unnecessary attack/complexity surface, not a missing-mount bug."""
    services = _services()
    for name in _HTTP_CLIENT_ONLY_CURATION_SERVICES:
        mounts = [str(v) for v in (services[name].get('volumes') or [])]
        assert not any(m.endswith(':/data/source:ro') for m in mounts), (
            f'{name} now mounts /data/source -- update this test/comment if '
            f'that is intentional (it started calling cascade_detect directly?)'
        )


def test_pe_image_encoder_in_default_triton_load_list() -> None:
    """G-07: docs/CURATION.md calls pe_image_encoder required, not
    optional -- it must be in the default --load-model list, not left for
    every deployment to add by hand."""
    command = _services()['triton-server'].get('command') or []
    assert '--load-model=pe_image_encoder' in [str(c) for c in command]


def test_vlm_service_is_opt_in_with_a_pinned_image() -> None:
    """G-10: the in-compose VLM must not start on a bare `docker compose
    up` (profiles: [vlm]) and must not float on `:latest` (an untested
    vLLM bump can silently change chat-template/tool-call-parser
    behavior)."""
    services = _services()
    assert 'vlm' in services, 'expected an optional `vlm` service in docker-compose.yml'
    vlm = services['vlm']
    assert 'vlm' in (vlm.get('profiles') or []), 'vlm service must carry profiles: [vlm]'
    image = str(vlm.get('image', ''))
    assert ':latest' not in image, f'vlm service image must be pinned, not :latest: {image!r}'


def test_yolo_api_carries_op_api_network_alias() -> None:
    """G-28: Cropwright's default API_UPSTREAM is http://op-api:8000; this
    alias lets that default resolve without every deployer overriding it."""
    networks = _services()['yolo-api'].get('networks')
    assert isinstance(networks, dict), 'expected yolo-api networks: to carry aliases (dict form)'
    aliases = (networks.get('triton_net') or {}).get('aliases') or []
    assert 'op-api' in aliases, f'expected op-api in yolo-api triton_net aliases: {aliases}'


def test_gpu_arbiter_overlay_exists_and_mounts_docker_socket() -> None:
    """G-15: the docker socket must be opt-in (a separate overlay file),
    never a default mount on yolo-api."""
    overlay_path = REPO_ROOT / 'docker-compose.gpu-arbiter.yml'
    assert overlay_path.is_file(), 'expected docker-compose.gpu-arbiter.yml overlay'
    with overlay_path.open() as fh:
        overlay = yaml.safe_load(fh)
    mounts = [str(v) for v in (overlay['services']['yolo-api'].get('volumes') or [])]
    assert any('/var/run/docker.sock' in m for m in mounts), mounts
    # And the base compose must NOT already mount it (defeats the point of
    # an opt-in overlay).
    base_mounts = [str(v) for v in (_services()['yolo-api'].get('volumes') or [])]
    assert not any('docker.sock' in m for m in base_mounts), base_mounts


def test_triton_serves_partial_model_sets() -> None:
    """The minimal setup profile skips OCR, and setup continues past a failed
    export; with Triton's default exit-on-error a single missing engine in the
    --load-model list kills the server, so nothing is served at all."""
    cmd = [str(c) for c in _services()['triton-server']['command']]
    assert '--exit-on-error=false' in cmd, cmd
    assert '--strict-readiness=false' in cmd, cmd


def test_vlm_image_pinned_by_digest() -> None:
    image = str(_services()['vlm']['image'])
    assert '@sha256:' in image, image
