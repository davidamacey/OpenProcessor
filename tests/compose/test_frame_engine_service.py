"""The frame-engine service: the shape its CUDA shared-memory transport and its recovery need."""

from __future__ import annotations

from compose._helpers import REPO_ROOT, _load_yaml, _services


def _engine() -> dict:
    return _services()['frame-engine']


def test_the_engine_is_in_the_core_stack_and_comes_back_by_itself() -> None:
    engine = _engine()
    assert 'profiles' not in engine
    assert engine['restart'] == 'always'


def test_the_engine_runs_under_an_init_so_it_is_never_process_one() -> None:
    """Registering a CUDA shared-memory region fails when the registering process has the model
    server's process id; both are 1 without an init (spike E0, results/ns_tests.jsonl)."""
    assert _engine()['init'] is True


def test_the_engine_shares_tritons_card_unless_told_otherwise() -> None:
    devices = _engine()['deploy']['resources']['reservations']['devices']
    assert devices[0]['device_ids'] == ['${FRAME_ENGINE_GPU_ID:-${TRITON_GPU_ID:-0}}']


def test_triton_accepts_client_shared_memory() -> None:
    command = [str(a) for a in _services()['triton-server']['command']]
    assert '--allow-client-shm=true' in command


def test_the_engine_starts_after_triton_and_checks_readiness() -> None:
    engine = _engine()
    assert 'triton-server' in engine['depends_on']
    probe = ' '.join(str(a) for a in engine['healthcheck']['test'])
    assert 'is_server_ready' in probe
    env = {e.split('=', 1)[0]: e.split('=', 1)[1] for e in engine['environment']}
    assert env['OP_FRAME_ENGINE_TRITON_URL'] == '${OP_FRAME_ENGINE_TRITON_URL:-triton-server:8001}'
    assert env['OP_FRAME_ENGINE_CROP_MAX_SIDE'] == '${OP_CROP_MAX_SIDE:-0}'


def test_the_dev_overlay_builds_the_engine_from_the_repo_root() -> None:
    dev = _load_yaml(REPO_ROOT / 'docker-compose.dev.yml')['services']['frame-engine']
    assert dev['build'] == {
        'context': '.',
        'dockerfile': 'docker/frame-engine/Dockerfile',
        'args': {'OP_BUILD_SHA': '${OP_BUILD_SHA:-}'},
    }
