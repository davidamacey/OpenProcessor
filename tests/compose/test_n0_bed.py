"""The N0 test bed of the native frame engine (``docker/frame-engine-native/``): the model server
beside both engines on one card, every container addressable by the name the hold scripts use, and
an image built from what the plan pins."""

from __future__ import annotations

import re
from typing import Any

import pytest

from compose._helpers import REPO_ROOT, _load_yaml


BED = REPO_ROOT / 'docker' / 'frame-engine-native'
COMPOSE = BED / 'compose.n0.yml'
DOCKERFILE = BED / 'Dockerfile'
PLAN = REPO_ROOT / 'docs' / 'design' / 'v070_rebuild' / '17_native_engine_plan.md'
BED_LIB = REPO_ROOT / 'scripts' / 'bench' / 'frame_engine_bed.sh'


def services() -> dict[str, Any]:
    out: dict[str, Any] = _load_yaml(COMPOSE)['services']
    return out


def command_flags() -> list[str]:
    return [str(a) for a in services()['triton-server']['command']]


def test_the_project_is_op_n0_never_the_default_one() -> None:
    assert _load_yaml(COMPOSE)['name'] == 'op-n0'


def test_container_names_are_the_ones_the_scripts_address() -> None:
    text = BED_LIB.read_text()
    defaults = dict(re.findall(r': "\$\{(\w+_CONTAINER):=([\w-]+)\}"', text))
    names = {key: svc['container_name'] for key, svc in services().items()}
    assert defaults == {
        'SERVER_CONTAINER': names['triton-server'],
        'NATIVE_CONTAINER': names['engine-native'],
        'PYTHON_CONTAINER': names['engine-python'],
    }


def test_published_ports_are_the_plans_and_only_on_loopback() -> None:
    published = {
        key: sorted(str(p) for p in svc.get('ports', [])) for key, svc in services().items()
    }
    assert published == {
        'triton-server': ['127.0.0.1:4620:8001'],
        'engine-native': ['127.0.0.1:4621:8001', '127.0.0.1:4623:8002'],
        'engine-python': ['127.0.0.1:4622:8001', '127.0.0.1:4624:8002'],
    }


def test_the_scripts_default_urls_are_the_published_ports() -> None:
    text = BED_LIB.read_text()
    urls = dict(re.findall(r': "\$\{(\w+_URL):=([\d.:]+)\}"', text))
    assert urls == {
        'SERVER_URL': '127.0.0.1:4620',
        'NATIVE_URL': '127.0.0.1:4621',
        'PYTHON_URL': '127.0.0.1:4622',
    }


def test_all_three_containers_reserve_the_one_card_by_device_id() -> None:
    for name, svc in services().items():
        (device,) = svc['deploy']['resources']['reservations']['devices']
        assert device['device_ids'] == ['${N0_GPU_ID:-2}'], name
        assert 'count' not in device, 'count and device_ids are exclusive'
        assert device['capabilities'] == ['gpu']


def test_both_engines_run_under_an_init_with_unlimited_memlock() -> None:
    """A region cannot be registered by a process that has the model server's process id."""
    for name in ('engine-native', 'engine-python'):
        assert services()[name]['init'] is True
        assert services()[name]['ulimits']['memlock'] == -1


def test_the_two_engines_never_unregister_each_others_region() -> None:
    env = {n: services()[n]['environment'] for n in ('engine-native', 'engine-python')}
    instances = {n: e['OP_FRAME_ENGINE_INSTANCE'] for n, e in env.items()}
    assert instances == {'engine-native': 'fe-native', 'engine-python': 'fe-python'}
    # the instance names differ, and only the native engine reads photos by key
    native_only = {
        'OP_FRAME_ENGINE_INSTANCE',
        'OP_FRAME_ENGINE_LANDING_ROOT',
        'OP_FRAME_ENGINE_SOURCE_ROOT',
    }
    shared = {k: v for k, v in env['engine-native'].items() if k not in native_only}
    assert shared == {
        k: v for k, v in env['engine-python'].items() if k != 'OP_FRAME_ENGINE_INSTANCE'
    }
    assert shared['OP_FRAME_ENGINE_TRITON_URL'] == 'triton-server:8001'
    assert 'OP_FRAME_ENGINE_SWITCH_INTERVAL_MS' not in shared, (
        'the native engine refuses to start with it'
    )


def test_the_engine_defaults_are_the_committed_ones() -> None:
    entries = _load_yaml(REPO_ROOT / 'docker-compose.yml')['services']['frame-engine'][
        'environment'
    ]
    main = {e.split('=', 1)[0]: e.split('=', 1)[1] for e in entries}
    env = services()['engine-native']['environment']
    for key in (
        'OP_FRAME_ENGINE_REGION_MIB',
        'OP_FRAME_ENGINE_WORKERS',
        'OP_FRAME_ENGINE_BUDGET_MIB',
    ):
        assert env[key] == re.search(r':-(\d+)\}', main[key]).group(1), key  # type: ignore[union-attr]


def test_the_model_server_takes_client_shared_memory_and_both_repositories() -> None:
    flags = command_flags()
    assert '--allow-client-shm=true' in flags
    assert [f for f in flags if f.startswith('--model-repository=')] == [
        '--model-repository=/models',
        '--model-repository=/stand_in',
    ]
    assert '--model-control-mode=explicit' in flags
    assert '--grpc-keepalive-permit-without-calls=true' in flags
    assert '--grpc-http2-min-recv-ping-interval-without-data=20000' in flags


def test_every_model_the_plans_name_is_loaded() -> None:
    from scripts.bench import frame_engine_stand_ins as si

    loaded = {f.split('=', 1)[1] for f in command_flags() if f.startswith('--load-model=')}
    real = {'yolov11_small_trt_end2end', 'pe_image_encoder', 'pe_crop_encoder'}
    assert loaded == real | set(si.null_names()) | set(si.capture_names())


def test_the_stand_ins_and_the_capture_directory_are_mounted_where_the_configs_say() -> None:
    from scripts.bench import frame_engine_stand_ins as si

    # `${VARIABLE:-default}:/target[:ro]`: the target follows the closing brace
    volumes = [str(v) for v in services()['triton-server']['volumes']]
    mounts = {m.group(1): v for v in volumes if (m := re.search(r'\}:(/[\w/]+)(?::ro)?$', v))}
    assert set(mounts) == {'/models', si.SERVER_STAND_IN_DIR, si.SERVER_CAPTURE_DIR}
    assert mounts['/models'].endswith(':/models:ro')
    assert mounts['/capture'].endswith(':/capture')


def test_images_are_local_only_tags_and_built_from_the_repository_root() -> None:
    for name, svc in services().items():
        assert svc['image'].split(':-')[1].rstrip('}').startswith('op-n0/'), name
        assert not svc['image'].rstrip('}').endswith(':latest')
        assert svc['build']['context'] == '../..'
    assert (
        services()['engine-native']['build']['dockerfile']
        == 'docker/frame-engine-native/Dockerfile'
    )
    assert services()['engine-native']['build']['additional_contexts'] == {
        'contracts_json': '../../contracts/json'
    }


# ----------------------------------------------------------------------------- the image


def stage_froms() -> list[str]:
    return re.findall(r'^FROM (\S+)', DOCKERFILE.read_text(), re.M)


def test_the_image_has_the_plans_three_stages_on_the_plans_pinned_bases() -> None:
    plan = PLAN.read_text()
    froms = stage_froms()
    assert len(froms) == 3
    for base in froms:
        assert base in plan, f'{base} is not the base the plan pins'


def test_the_toolchain_is_the_one_the_workspace_pins() -> None:
    channel = re.search(
        r'channel = "([\d.]+)"', (REPO_ROOT / 'native' / 'rust-toolchain.toml').read_text()
    )
    assert channel
    text = DOCKERFILE.read_text()
    assert f'FROM rust:{channel.group(1)}-slim-trixie@sha256:' in text
    assert f'RUSTUP_TOOLCHAIN={channel.group(1)}' in text


def test_libjpeg_turbo_is_the_plans_version_and_checked_by_digest() -> None:
    text = DOCKERFILE.read_text()
    assert 'ARG LIBJPEG_TURBO_VERSION=3.1.2' in text
    assert re.search(r'ARG LIBJPEG_TURBO_SHA256=[0-9a-f]{64}\n', text)
    assert 'sha256sum -c' in text
    assert '-DREQUIRE_SIMD=1' in text
    assert 'TURBOJPEG_SOURCE=explicit' in text


def test_the_cuda_libraries_come_from_the_python_engines_pins_not_a_second_copy() -> None:
    text = DOCKERFILE.read_text()
    assert 'COPY docker/frame-engine/requirements.txt' in text
    assert not re.search(r'nvidia-\w+(-\w+)?==\d', text), (
        'a version spelled out here would be a second copy'
    )
    pins = [
        ln
        for ln in (REPO_ROOT / 'docker/frame-engine/requirements.txt').read_text().splitlines()
        if ln.startswith('nvidia-')
    ]
    assert len(pins) == 3


def test_the_runtime_is_non_root_with_a_health_check_and_the_engine_as_entry() -> None:
    final = DOCKERFILE.read_text().split('\nFROM ')[-1]
    assert re.search(r'^USER engine$', final, re.M)
    assert 'useradd -r -g engine -G video' in final
    assert 'HEALTHCHECK' in final
    assert 'CMD ["op-frame-engine", "health"]' in final
    assert 'ENTRYPOINT ["op-frame-engine"]' in final


@pytest.mark.parametrize('rel', ['native', 'docker/frame-engine/requirements.txt'])
def test_what_the_image_copies_from_the_context_exists_and_is_not_ignored(rel: str) -> None:
    from tests.installer.test_dockerignore_context import is_ignored, load_rules

    assert (REPO_ROOT / rel).exists()
    assert not is_ignored(load_rules('.'), f'{rel}/Cargo.toml' if rel == 'native' else rel)


def test_only_the_native_engine_mounts_the_landing_zone_and_the_source_folder_and_only_to_read() -> (
    None
):
    native, python = services()['engine-native'], services()['engine-python']
    mounts = {':'.join(str(v).rsplit(':', 2)[1:]) for v in native['volumes']}
    assert mounts == {'/data/landing:ro', '/data/source:ro'}
    env = native['environment']
    assert (env['OP_FRAME_ENGINE_LANDING_ROOT'], env['OP_FRAME_ENGINE_SOURCE_ROOT']) == (
        '/data/landing', '/data/source')  # fmt: skip
    assert 'volumes' not in python
    assert not any('ROOT' in k for k in python['environment'])
