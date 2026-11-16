"""The API port is served by the Rust front (service `api`, `op-server api`); the Python API runs
behind it (service `api-py`, internal port 4613, not published) and every request the front does
not own is forwarded there (plan 18a, slice R1a)."""

from __future__ import annotations

import re
from typing import Any

from compose._helpers import REPO_ROOT, _load_yaml, _services


def _env(name: str) -> dict[str, str]:
    entries = _services()[name]['environment']
    return dict(e.split('=', 1) for e in entries if '=' in e)


def test_the_front_runs_the_server_binary_in_its_api_role() -> None:
    front = _services()['api']
    assert front['image'].startswith('${OP_SERVER_IMAGE:-')
    assert front['image'].endswith('openprocessor-server:${OP_IMAGE_TAG:-0.6.0}}')
    assert front['command'] == ['api']


def test_only_the_front_publishes_the_api_port() -> None:
    services = _services()
    assert services['api']['ports'] == ['${OP_BIND_ADDRESS:-127.0.0.1}:${API_PORT:-4603}:8000']
    assert not services['api-py'].get('ports')


def test_the_front_forwards_to_the_python_api_on_its_internal_port() -> None:
    services = _services()
    assert _env('api')['OP_PY_API_URL'] == 'http://api-py:4613'
    command = [str(a) for a in services['api-py']['command']]
    assert '--port=4613' in command
    assert '--port=8000' not in command
    probe = services['api-py']['healthcheck']['test'][-1]
    assert 'localhost:4613/health' in probe


def test_the_front_starts_after_the_python_api_is_healthy() -> None:
    depends_on = _services()['api']['depends_on']
    assert depends_on == {'api-py': {'condition': 'service_healthy'}}


def test_the_front_is_healthy_when_the_python_api_answers_through_it() -> None:
    check = _services()['api']['healthcheck']
    assert check['test'] == ['CMD', 'op-server', 'health', '--path', '/health']


def test_every_caller_inside_the_network_reaches_the_front_on_the_old_address() -> None:
    services = _services()
    aliases = services['api']['networks']['triton_net']['aliases']
    assert 'op-api' in aliases
    assert 'networks' in services['api-py']
    assert 'aliases' not in str(services['api-py']['networks'])
    assert 'OP_API_LOOPBACK_URL=http://api:8000' in services['jobs-worker']['environment']


def test_the_front_gets_the_whole_env_file_so_a_retired_variable_in_it_is_refused() -> None:
    env_file = _services()['api']['env_file']
    assert any(entry.get('path') == './.env' for entry in env_file)


def test_the_front_runs_without_privileges_and_without_a_writable_filesystem() -> None:
    front = _services()['api']
    assert 'no-new-privileges:true' in front['security_opt']
    assert front['cap_drop'] == ['ALL']
    assert front['read_only'] is True
    assert not front.get('privileged')
    assert not front.get('volumes')


def test_the_front_gets_as_long_to_stop_as_uvicorn_gave_requests() -> None:
    assert _services()['api']['stop_grace_period'] == '35s'


def test_the_dev_overlay_builds_the_front_from_its_own_dockerfile() -> None:
    dev = _load_yaml(REPO_ROOT / 'docker-compose.dev.yml')['services']
    build = dev['api']['build']
    assert build['dockerfile'] == 'docker/op-server/Dockerfile'
    assert build['additional_contexts'] == {'contracts_json': './contracts/json'}
    assert dev['api-py']['build']['target'] == 'api'


def _stages(text: str) -> list[str]:
    return re.findall(r'^FROM\s+(\S+)(?:\s+AS\s+\S+)?', text, flags=re.M)


def test_the_server_image_is_multi_stage_slim_non_root_and_carries_exiftool() -> None:
    dockerfile = (REPO_ROOT / 'docker' / 'op-server' / 'Dockerfile').read_text()
    stages = _stages(dockerfile)
    assert len(stages) == 2, stages
    assert all('@sha256:' in stage for stage in stages), 'every base is pinned by digest'
    assert all('slim' in stage.split(':', 1)[1] for stage in stages), 'slim bases'
    runtime = dockerfile.split('AS build', 1)[1].split('FROM ', 1)[1]
    assert re.search(r'^USER\s+(?!root)\S+', runtime, flags=re.M)
    assert 'libimage-exiftool-perl' in runtime
    assert 'ENTRYPOINT ["op-server"]' in runtime
    assert 'HEALTHCHECK' in runtime
    assert 'cargo build --release --locked' in dockerfile


def test_the_image_key_table_builds_the_server_image_from_its_dockerfile() -> None:
    table = (REPO_ROOT / 'scripts' / 'lib' / 'image_keys.sh').read_text()
    row = '"server|OP_SERVER_IMAGE|build|docker/op-server/Dockerfile|openprocessor-server|"'
    assert row in table


def test_the_python_api_keeps_its_gpu_and_replica_settings() -> None:
    spec: dict[str, Any] = _services()['api-py']
    assert spec['deploy']['replicas'] == '${OP_API_REPLICAS:-1}'
    device = spec['deploy']['resources']['reservations']['devices'][0]
    assert device['device_ids'] == ['${API_GPU_ID:-0}']


def test_the_s3_tryout_overlay_configures_the_container_that_reads_s3() -> None:
    """boto3 runs in the Python API; the front reads none of OP_S3_*."""
    overlay = _load_yaml(REPO_ROOT / 'docker/test/compose.s3.yml')['services']
    readers = [name for name, spec in overlay.items() if 'environment' in spec]
    assert readers == ['api-py']
    assert any('OP_S3_ENDPOINT_URL' in entry for entry in overlay['api-py']['environment'])
    assert set(overlay) - {'minio'} <= set(_services())
