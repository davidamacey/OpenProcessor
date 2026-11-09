"""6B-7 / S-16: the triton-server card comes from the env, never a fixed name.

Docs: https://docs.docker.com/reference/compose-file/deploy/#devices (device_ids),
https://docs.docker.com/reference/compose-file/services/#container_name.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import yaml

from compose._helpers import REPO_ROOT, render


def _device_ids(env: dict[str, str]) -> list[str]:
    devices = render(env)['triton-server']['deploy']['resources']['reservations']['devices']
    return devices[0]['device_ids']


def test_one_card_render_defaults_to_card_zero() -> None:
    assert _device_ids({}) == ['0']


def test_the_card_comes_from_triton_gpu_id() -> None:
    assert _device_ids({'TRITON_GPU_ID': '2'}) == ['2']


def test_a_card_list_is_one_device_entry() -> None:
    assert _device_ids({'TRITON_GPU_ID': '0,2'}) == ['0,2']


def test_triton_server_has_no_container_name() -> None:
    assert 'container_name' not in render({})['triton-server']


# --- the installer-generated per-card overlay (scripts/lib/gpu.sh) -----------------------------


class _TagIgnoringLoader(yaml.SafeLoader):
    """Reads Compose's `!override` tag as the plain value it wraps."""


_TagIgnoringLoader.add_multi_constructor(
    '!', lambda loader, _suffix, node: loader.construct_sequence(node)
)

TOPOLOGY = Path(__file__).parent / 'fixtures' / 'topo_two_numa.txt'


def _generate(ids: str, topology: Path = TOPOLOGY) -> subprocess.CompletedProcess[str]:
    script = f'source scripts/lib/gpu.sh; gpu_triton_cards_overlay "$1" "$(cat {topology})"'
    return subprocess.run(
        ['bash', '-c', script, 'x', ids],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_the_generated_overlay_has_one_pinned_server_per_card() -> None:
    done = _generate('2,0')
    assert done.returncode == 0, done.stderr
    services = yaml.load(done.stdout, Loader=_TagIgnoringLoader)['services']
    assert set(services) == {'triton-server', 'triton-server-2'}
    primary, second = services['triton-server'], services['triton-server-2']
    assert primary['cpuset'] == '0-11,24-35'
    assert second['cpuset'] == '12-23,36-47'
    for service, card in ((primary, '0'), (second, '2')):
        assert service['deploy']['resources']['reservations']['devices'][0]['device_ids'] == [card]
    assert second['extends'] == {'file': 'docker-compose.yml', 'service': 'triton-server'}
    assert second['ports'] == []
    assert 'ports' not in primary


def test_the_generated_overlay_renders_with_the_base_file(tmp_path: Path) -> None:
    done = _generate('0,2')
    assert done.returncode == 0, done.stderr
    shutil.copy(REPO_ROOT / 'docker-compose.yml', tmp_path / 'docker-compose.yml')
    (tmp_path / 'docker-compose.triton-cards.yml').write_text(done.stdout)
    rendered = subprocess.run(
        [
            'docker',
            'compose',
            '-p',
            'op-6b-check',
            '-f',
            str(tmp_path / 'docker-compose.yml'),
            '-f',
            str(tmp_path / 'docker-compose.triton-cards.yml'),
            'config',
            '--format',
            'json',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env={'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/tmp'},
    )
    assert rendered.returncode == 0, rendered.stderr
    services = json.loads(rendered.stdout)['services']
    second = services['triton-server-2']
    assert second['cpuset'] == '12-23,36-47'
    assert 'ports' not in second
    assert second['command'] == services['triton-server']['command']
    assert second['deploy']['resources']['reservations']['devices'][0]['device_ids'] == ['2']


def test_one_card_a_repeat_or_an_unknown_card_makes_no_overlay() -> None:
    for ids in ('0', '0,0', '0,7', '0,x'):
        done = _generate(ids)
        assert done.returncode == 1, ids
        assert done.stdout == ''
        assert done.stderr


def test_a_matrix_without_cpu_affinity_makes_no_overlay(tmp_path: Path) -> None:
    bare = tmp_path / 'topo.txt'
    bare.write_text('\tGPU0\tGPU1\nGPU0\t X \tPHB\nGPU1\tPHB\t X \n')
    done = _generate('0,1', bare)
    assert done.returncode == 1
    assert done.stdout == ''
