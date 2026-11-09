"""6B-7 / S-16: the triton-server card comes from the env, never a fixed name.

Docs: https://docs.docker.com/reference/compose-file/deploy/#devices (device_ids),
https://docs.docker.com/reference/compose-file/services/#container_name.
"""

from __future__ import annotations

from compose._helpers import render


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
