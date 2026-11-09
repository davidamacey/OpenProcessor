"""The Triton release and the Python bases are named in several files; they must agree.

TensorRT plans are bound to the TensorRT version of the Triton image that serves them, and
``tensorrt-cu13`` in the export image must equal it (``test_requirements_contract.py``). These
tests tie the rest: the Triton release tag in every file that names it, and the interpreter
minor each Dockerfile builds on against the paths it hard-codes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

_TRITON_TAG = r'nvcr\.io/nvidia/tritonserver:(\d{2}\.\d{2})-py3'


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding='utf-8')


def _one(pattern: str, rel: str) -> str:
    (found,) = re.findall(pattern, _read(rel), re.M)
    return found


def test_the_triton_release_is_one_tag_in_every_file_that_names_it() -> None:
    tags = {
        'Dockerfile.triton': _one(
            rf'^FROM {_TRITON_TAG}@sha256:[0-9a-f]{{64}}$', 'Dockerfile.triton'
        ),
        'docker-compose.yml (triton-sdk)': _one(
            rf'image: {_TRITON_TAG}-sdk$', 'docker-compose.yml'
        ),
        'scripts/bench/suite_phases.py': _one(
            rf"^SDK_IMAGE = '{_TRITON_TAG}-sdk'$", 'scripts/bench/suite_phases.py'
        ),
        'docker/hardened/triton/Dockerfile': _one(
            r'^ARG TRITON_TAG=(\d{2}\.\d{2})-py3$', 'docker/hardened/triton/Dockerfile'
        ),
    }
    assert len(set(tags.values())) == 1, tags


def test_the_triton_dockerfile_header_names_the_release_it_builds_on() -> None:
    release = _one(rf'^FROM {_TRITON_TAG}@', 'Dockerfile.triton')
    header = _read('Dockerfile.triton').split('\nFROM ', 1)[0]
    assert f'Triton {release}' in header


_PYTHON_FROM = re.compile(r'^FROM python:(\d+\.\d+)-slim-trixie@(sha256:[0-9a-f]{64})', re.M)


@pytest.mark.parametrize('rel', ['Dockerfile', 'docker/trainer/Dockerfile'])
def test_every_stage_of_an_image_uses_one_python_base_and_its_paths(rel: str) -> None:
    text = _read(rel)
    bases = _PYTHON_FROM.findall(text)
    assert bases, rel
    assert len(set(bases)) == 1, bases
    minor = bases[0][0]
    stated = set(re.findall(r'python(\d+\.\d+)(?=/|\s|$)', re.sub(r'\bpython:\S+', '', text)))
    assert stated <= {minor}, (rel, minor, stated)


def test_the_api_image_stays_below_the_python_that_faiss_gpu_supports() -> None:
    """faiss-gpu-cu12 declares ``Requires-Python <3.14`` and ships no CUDA 13 build
    (docs/design/v070_rebuild/01_package_pins.md section 2), and the API image installs it."""
    runtime = _read('requirements.txt')
    assert re.search(r'^faiss-gpu-cu12==', runtime, re.M), (
        'faiss left the API image: drop this test'
    )
    major, minor = (int(p) for p in _PYTHON_FROM.findall(_read('Dockerfile'))[0][0].split('.'))
    assert (major, minor) < (3, 14)
