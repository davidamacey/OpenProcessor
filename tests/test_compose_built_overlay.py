"""The built-image overlay replaces each service's volume list (Compose cannot delete one list item),
so every mount it does not repeat is lost. It may drop only the checkout's code mounts, which the
image already contains. A dropped fake script or data mount leaves the container crashing on start
(the trainer lost ``fake_onnx.py`` and every run stayed queued)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


TEST_DIR = Path(__file__).resolve().parents[1] / 'docker' / 'test'
BAKED_INTO_IMAGE = {'../../src', '../../scripts', '../../VERSION'}


class _Loader(yaml.SafeLoader):
    pass


def _tag(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return None


_Loader.add_constructor('!override', _tag)
_Loader.add_constructor('!reset', _tag)


def _load(name: str) -> dict:
    return yaml.load((TEST_DIR / name).read_text(encoding='utf-8'), Loader=_Loader)[
        'services'
    ]  # SafeLoader subclass


_BASE = _load('compose.yml')
_BUILT = {n: s for n, s in _load('compose.built.yml').items() if 'volumes' in s}


def test_overlay_overrides_some_volumes() -> None:
    assert _BUILT


@pytest.mark.parametrize('service', sorted(_BUILT))
def test_overlay_drops_only_baked_in_code_mounts(service: str) -> None:
    kept = set(_BUILT[service]['volumes'])
    lost = [
        m
        for m in _BASE[service]['volumes']
        if m not in kept and m.split(':')[0] not in BAKED_INTO_IMAGE
    ]
    assert not lost, f'{service}: compose.built.yml drops mounts the service needs: {lost}'
    assert not [m for m in kept if m not in _BASE[service]['volumes']], (
        f'{service}: overlay adds mounts'
    )
