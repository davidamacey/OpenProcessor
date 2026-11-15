"""The CUDA source each frame-engine module hands to the compiler is the recorded text, byte for byte.

The digests were taken from the ``_SRC`` strings the Python modules held before the sources moved to
``native/frame_engine/kernels/``. Any drift in a kernel changes what NVRTC compiles and so every
output built on it; a change on purpose is made by recording the new digest in the same commit.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest


KERNELS = Path(__file__).resolve().parents[2] / 'native' / 'frame_engine' / 'kernels'

#: kernel module -> (file, SHA-256 of the exact text given to the compiler, its length in bytes)
RECORDED = {
    'frame_engine.prep_kernels': (
        'prep.cu',
        '5f4867c6af5bc69f283d08638216f9e4886c58b8a08fe3a09b2cea7f19579376',
        3653,
    ),
    'frame_engine.colour': (
        'colour.cu',
        '12ebb05386b1256d83bcfb8b257e859601fe7fa6a846c4ee11cb9d78b0bcc162',
        423,
    ),
    'frame_engine.faces.kernels': (
        'faces_warp.cu',
        'c905e3f28ad3cebbe1fd455aa5ab02a373952015a24daddc494dfc0040f9fc5d',
        2012,
    ),
    'frame_engine.ocr.kernels': (
        'ocr_warp.cu',
        '34503d9c7effb53b06bf0e56aa5bb3e73c83cf24b506c34b63860cf19aac365a',
        2349,
    ),
}


@pytest.fixture
def load(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import a kernel module against a stand-in ``cupy`` (importing it compiles nothing)."""
    monkeypatch.setitem(sys.modules, 'cupy', types.ModuleType('cupy'))

    def _load(name: str) -> Any:
        monkeypatch.delitem(sys.modules, name, raising=False)
        return importlib.import_module(name)

    return _load


@pytest.mark.parametrize('module', sorted(RECORDED))
def test_the_kernel_file_is_the_recorded_text(module: str) -> None:
    name, digest, size = RECORDED[module]
    text = (KERNELS / name).read_bytes()
    assert (hashlib.sha256(text).hexdigest(), len(text)) == (digest, size)


@pytest.mark.parametrize('module', sorted(RECORDED))
def test_the_source_given_to_the_compiler_is_the_file_and_the_recorded_text(
    module: str, load: Any
) -> None:
    name, digest, size = RECORDED[module]
    text = load(module)._SRC.encode()
    assert text == (KERNELS / name).read_bytes()
    assert (hashlib.sha256(text).hexdigest(), len(text)) == (digest, size)


def test_the_image_and_the_dev_mount_put_the_kernels_where_the_modules_read_them() -> None:
    root = KERNELS.parents[2]
    where = '/app/native/frame_engine/kernels'
    dockerfile = (root / 'docker' / 'frame-engine' / 'Dockerfile').read_text()
    assert f'COPY native/frame_engine/kernels {where}\n' in dockerfile
    dev = (root / 'docker-compose.dev.yml').read_text()
    assert f'./native/frame_engine/kernels:{where}:ro' in dev
    assert root / 'native' / 'frame_engine' / 'kernels' == KERNELS
