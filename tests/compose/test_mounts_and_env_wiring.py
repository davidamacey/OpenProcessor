"""Pin per-service mounts, environment wiring and Triton server flags."""

from __future__ import annotations

import re

from compose._helpers import REPO_ROOT, _load_compose, _load_yaml, _services


def test_evaluator_sees_exports_at_the_api_path() -> None:
    """Bake-off eval datasets are exports: the evaluator must read them at the
    path the API resolved (``./data/projects/<project>/exports`` -> ``/app/data``)."""
    services = _services()
    api_mounts = services['api'].get('volumes') or []
    evaluator_mounts = services['curation-evaluator'].get('volumes') or []
    assert any(str(v).startswith('./data:/app/data') for v in api_mounts)
    assert './data:/app/data:ro' in evaluator_mounts


def test_evaluator_gets_the_same_mlflow_tracking_url_as_the_trainer() -> None:
    """The trainer gets
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
    api_env = services['api'].get('environment') or []
    assert _value(api_env, 'MLFLOW_TRACKING_URI') == trainer_url, (
        'project delete cleans the MLflow experiment over REST from the API'
    )


def test_api_evaluator_segmenter_gpu_ids_are_env_driven_not_hardcoded() -> None:
    """api, curation-evaluator
    and segmenter used to hardcode device_ids: ['0'] regardless of
    TRITON_GPU_ID/VLM_GPU_ID, so a host whose free GPU wasn't 0 needed a
    compose edit to run them at all."""
    services = _services()
    expectations = {
        'api': 'API_GPU_ID',
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
    """ST-1: api and curation-detection-worker must mount the SAME
    named crop-cache volume at the SAME target, with OP_CROP_CACHE_DIR set
    to that target in both -- otherwise the worker's cache reads never see
    what the API's ingest path wrote."""
    services = _services()
    target = '/var/cache/openprocessor/crops'
    for name in ('api', 'curation-detection-worker'):
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


def test_triton_accepts_the_clients_idle_keepalive_pings() -> None:
    """Found live: clients ping every 30 s while idle; Triton refused (idle pings
    off, 5-minute minimum interval) and sent GOAWAY too_many_pings."""
    command = _services()['triton-server']['command']
    assert '--grpc-keepalive-permit-without-calls=true' in command
    interval = next(
        int(arg.split('=', 1)[1])
        for arg in command
        if arg.startswith('--grpc-http2-min-recv-ping-interval-without-data=')
    )
    assert interval <= 30_000


def test_triton_does_not_log_every_pinned_memory_allocation() -> None:
    """`--log-verbose=1` printed an allocation and a free line for every tensor of every
    request (12,038 lines in one 2,000-image window) and bought nothing in production."""
    command = _services()['triton-server']['command']
    assert '--log-verbose=0' in command
    assert not any(a.startswith('--log-verbose=') and a != '--log-verbose=0' for a in command)


def test_api_keeps_freed_pixel_buffers_instead_of_trimming_them() -> None:
    """Ingest frees tens of megabytes per image; glibc's default trims each thread arena after
    every call and the next allocation faults the pages back (0.109 against 0.096 CPU-s per
    image on 640 x 480 frames)."""
    env = dict(a.split('=', 1) for a in _services()['api']['environment'] if '=' in a)
    assert int(env['MALLOC_TRIM_THRESHOLD_']) >= 64 * 1024 * 1024
    assert int(env['MALLOC_TOP_PAD_']) > 0
    assert int(env['MALLOC_MMAP_THRESHOLD_']) >= 16 * 1024 * 1024


def test_source_root_mounted_at_the_same_path_on_api_and_detection_worker() -> None:
    """G-06: ingest resolves item paths against OP_SOURCE_ROOT on api;
    the detection worker re-reads the same items later. Without the same
    bind on both, every worker read fails with
    detection_failed/reason=image_unavailable."""
    services = _services()
    target = '/data/source'
    for name in ('api', 'curation-detection-worker', 'curation-auto-label-worker'):
        mounts = [str(v) for v in (services[name].get('volumes') or [])]
        matching = [m for m in mounts if m.endswith((f':{target}:ro', f':{target}'))]
        assert matching, f'{name} has no source-root mount at {target}: {mounts}'


def test_examples_reachable_on_api_and_detection_worker() -> None:
    """G-08: OP_REGION_PROFILE_PATH's worked example
    (examples/region_profiles/license_plate.json) must resolve inside the
    container at /app/examples.

    Installer plan §1: docker-compose.yml is now deploy-safe (no bind
    mount of ./examples, so an installed, non-checkout deployment doesn't
    get an empty host dir shadowing the image). The intent this test
    originally covered -- examples reachable at /app/examples -- is now
    met two ways instead of one base-compose mount:
    (a) the Dockerfile bakes examples/ into the image itself, so a plain
        `docker compose pull && up -d` deploy has it without any mount;
    (b) docker-compose.dev.yml (checkout hot-reload only) still bind-mounts
        ./examples read-only on top, so local edits are picked up without
        a rebuild.
    """
    dockerfile = (REPO_ROOT / 'Dockerfile').read_text()
    assert re.search(r'^COPY\s+--chown=\S+\s+examples/\s+\./examples/', dockerfile, re.MULTILINE), (
        'Dockerfile must COPY examples/ into the image -- '
        'docker-compose.yml no longer bind-mounts it'
    )

    dev_compose = _load_yaml(REPO_ROOT / 'docker-compose.dev.yml')
    dev_services = dev_compose['services']
    target = '/app/examples'
    for name in ('api', 'curation-detection-worker', 'curation-auto-label-worker'):
        mounts = [str(v) for v in (dev_services.get(name, {}).get('volumes') or [])]
        assert any(m.endswith(f':{target}:ro') for m in mounts), (
            f'{name} has no ./examples mount at {target} in docker-compose.dev.yml: {mounts}'
        )


# curation-auto-label-worker
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
    'api',
    'curation-detection-worker',
    'curation-auto-label-worker',
)
_HTTP_CLIENT_ONLY_CURATION_SERVICES = ('curation-vlm-worker', 'curation-cluster-refresh')


def test_class_registry_data_mounted_where_needed() -> None:
    """Each project's class registry lives under OP_PROJECTS_DATA_ROOT
    (default ./data/projects/<project>/class_registry.json).
    Without ./data mounted, the registry loads empty and any class-aware
    stage (the VLM auto-label stage, training, promote, eval) raises
    'class_names or class_catalog must be supplied' or reads no classes
    instead of running. api and curation-auto-label-worker read the
    registry directly; curation-detection-worker deliberately does not
    (it only ever writes unlabeled `*_proposal` detections)."""
    services = _services()
    target = '/app/data'
    for name in ('api', 'curation-auto-label-worker'):
        mounts = [str(v) for v in (services[name].get('volumes') or [])]
        assert any(m.endswith((f':{target}', f':{target}:ro')) for m in mounts), (
            f'{name} has no ./data mount at {target}: {mounts}'
        )


def test_http_client_only_curation_services_stay_lightweight() -> None:
    """Documents *why* curation-vlm-worker/curation-cluster-refresh don't
    need the source-root/examples/data mounts above: they only ever talk
    to api over HTTP (verified via each entrypoint's own imports, not
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


def test_triton_serves_partial_model_sets() -> None:
    """The minimal setup profile skips OCR, and setup continues past a failed
    export; with Triton's default exit-on-error a single missing engine in the
    --load-model list kills the server, so nothing is served at all."""
    cmd = [str(c) for c in _services()['triton-server']['command']]
    assert '--exit-on-error=false' in cmd, cmd
    assert '--strict-readiness=false' in cmd, cmd


def test_api_default_trainer_container_matches_the_trainer_service_name() -> None:
    """/train/preflight's trainer-reachability probe
    (GpuArbiterConfig.trainer_container, env OP_GPU_ARBITER_TRAINER_CONTAINER)
    used to default to unset -- reporting the misleading "no trainer
    container configured" even while curation-trainer was up and healthy,
    because nothing ever wired the env var to the trainer service's own
    container_name. api's environment must default
    OP_GPU_ARBITER_TRAINER_CONTAINER to that exact value (still
    overridable), so the probe finds it with no extra config on any
    deployment running the `training` profile."""
    services = _services()
    api_env = services['api']['environment']
    trainer_container_name = str(services['curation-trainer']['container_name'])

    env_map = {}
    for entry in api_env:
        key, _, value = str(entry).partition('=')
        env_map[key] = value

    assert 'OP_GPU_ARBITER_TRAINER_CONTAINER' in env_map, (
        'api must set a default OP_GPU_ARBITER_TRAINER_CONTAINER'
    )
    default_expr = env_map['OP_GPU_ARBITER_TRAINER_CONTAINER']
    # The env var's own default value (inside the outer ${VAR:-...}) must
    # be exactly the trainer service's container_name expression -- both
    # resolve identically off the same COMPOSE_PROJECT_NAME.
    assert trainer_container_name in default_expr, (
        f'OP_GPU_ARBITER_TRAINER_CONTAINER default {default_expr!r} does not match '
        f"curation-trainer's container_name {trainer_container_name!r}"
    )


def test_api_service_aggregates_metrics_across_workers() -> None:
    """uvicorn --workers N keeps one registry per process; without a shared
    multiprocess dir a /metrics scrape reflects a single worker."""
    spec = _services()['api']
    env = spec['environment']
    entries = dict(e.split('=', 1) for e in env) if isinstance(env, list) else env
    mp_dir = entries['PROMETHEUS_MULTIPROC_DIR']
    assert any(t.split(':')[0] == mp_dir for t in spec['tmpfs']), (
        'multiproc dir must be a tmpfs (wiped on start)'
    )
