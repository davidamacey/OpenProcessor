"""The API image's dependency set must stay resolvable.

perception_models' own requirements exact-pin timm==1.0.15 (and ~30 research
packages), which conflicts with open-clip-torch>=3.2 (timm>=1.0.17, the first
release with MobileCLIP2-S2). It is installed with --no-deps in the Dockerfile
instead; listing it in requirements.txt makes `docker compose build` fail with
ResolutionImpossible.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines() -> list[str]:
    lines = (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    return [ln.split('#', 1)[0].strip() for ln in lines if ln.split('#', 1)[0].strip()]


def test_perception_models_not_resolved_with_its_own_pins() -> None:
    assert not [ln for ln in _requirement_lines() if ln.startswith('perception_models')]


def test_dockerfile_installs_pinned_perception_models_without_deps() -> None:
    dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    match = re.search(
        r'--no-deps\s*\\?\s*"perception_models @ git\+\S+@([0-9a-f]{40})"', dockerfile
    )
    assert match, 'perception_models must be installed --no-deps at a pinned commit'


def test_pe_encoder_runtime_deps_declared() -> None:
    names = {re.split(r'[<>=\[ ;@]', ln, maxsplit=1)[0].lower() for ln in _requirement_lines()}
    assert {'einops', 'regex', 'timm', 'ftfy', 'huggingface_hub'} <= names


def test_the_image_pins_fastapi_as_the_tests_do() -> None:
    """The committed OpenAPI contract and the route warm-up's test (which reads FastAPI's
    included-router internals) are checked against the test pin; an unpinned image could
    serve another FastAPI where both differ unnoticed."""
    test_lines = (ROOT / 'requirements-test.txt').read_text(encoding='utf-8').splitlines()
    (tested,) = [ln.split('#', 1)[0].strip() for ln in test_lines if ln.startswith('fastapi')]
    assert tested in _requirement_lines()


_EXPORT_ONLY = (
    'tensorrt',
    'nvidia-modelopt',
    'onnxsim',
    'onnxslim',
    'onnx-graphsurgeon',
    'onnxscript',
)


def _pinned_trt(path: str) -> str:
    text = (ROOT / path).read_text(encoding='utf-8')
    (pin,) = re.findall(r'^tensorrt-cu13==(\S+)', text, re.M)
    return pin.split('#')[0]


def test_runtime_requirements_carry_no_export_toolchain() -> None:
    """The slim runtime image (API + workers) must not install the engine-build toolchain;
    it lives in requirements-export.txt and ships only in the export image."""
    names = {re.split(r'[<>=\[ ;@]', ln, maxsplit=1)[0].lower() for ln in _requirement_lines()}
    leaked = sorted(n for n in names if n.startswith(_EXPORT_ONLY))
    assert leaked == []
    assert {'onnx', 'onnxruntime-gpu', 'cupy-cuda12x'} <= names


def _ultralytics_specs() -> dict[str, str]:
    specs = {}
    for path in ('requirements.txt', 'requirements-test.txt'):
        for ln in (ROOT / path).read_text(encoding='utf-8').splitlines():
            spec = ln.split('#', 1)[0].strip()
            if spec.startswith('ultralytics'):
                specs[path] = spec
    with (ROOT / 'pyproject.toml').open('rb') as f:
        deps = tomllib.load(f)['project']['dependencies']
    specs['pyproject.toml'] = next(d for d in deps if d.startswith('ultralytics'))
    return specs


def test_ultralytics_is_one_exact_version_everywhere() -> None:
    """A range lets each image build resolve a different release, and the export head
    changed inside 8.4.x without notice (#447). The isolated YOLO11 export venv
    (requirements-export-y11.txt) has its own exact pin until the single-version move."""
    specs = _ultralytics_specs()
    assert set(specs) == {'requirements.txt', 'requirements-test.txt', 'pyproject.toml'}
    assert all(re.fullmatch(r'ultralytics==\d+\.\d+\.\d+', s) for s in specs.values()), specs
    assert len(set(specs.values())) == 1, specs


def test_tensorrt_pin_matches_the_triton_image_everywhere() -> None:
    triton = (ROOT / 'Dockerfile.triton').read_text(encoding='utf-8')
    (version,) = re.findall(r'^ARG TRT_VERSION=(\S+)', triton, re.M)
    assert _pinned_trt('requirements-export.txt') == version
    assert _pinned_trt('requirements-export-y11.txt') == version
