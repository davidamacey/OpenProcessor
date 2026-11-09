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


# ---------------------------------------------------------------------------
# Exact pins, identical across manifests (ENV-11; docs/design/v070_rebuild/01_package_pins.md
# section 5). pip-style requirement lines are parsed by hand: `packaging` is not a test dep
# and only name, extras and the `==` version matter here.
# ---------------------------------------------------------------------------

_REQ_RE = re.compile(
    r'^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[(?P<extras>[^\]]*)\])?\s*(?P<spec>.*)$'
)

# Packages the pin series has not reached yet (part B of the series). SHRINK-ONLY: part B
# deletes each name in the commit that pins it, and its last commit leaves this empty. A
# name here that no manifest declares loosely any more is rejected below, so a pinned
# package cannot stay on the list.
_UNPINNED_UNTIL_PART_B = frozenset(
    {
        'onnx',
        'onnxruntime',
        'onnxruntime-gpu',
        'cupy-cuda12x',
        'faiss-cpu',
        'faiss-gpu-cu12',
    }
)

# One distribution family, one version: the extra or build differs, the release must not.
_VERSION_FAMILIES = {
    'onnxruntime-gpu': 'onnxruntime',
    'opencv-python-headless': 'opencv-python',
}
# CPU and CUDA builds of faiss are versioned independently and never share a venv.
_NO_SHARED_VERSION = frozenset({'faiss-cpu', 'faiss-gpu-cu12'})


def _parse(line: str) -> tuple[str, str] | None:
    text = line.split('#', 1)[0].strip()
    if not text or text.startswith('-'):
        return None
    match = _REQ_RE.match(text.split(';', 1)[0].strip())
    assert match, text
    name = match['name'].lower().replace('_', '-')
    return name, match['spec'].replace(' ', '')


def _file_specs(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for ln in (ROOT / path).read_text(encoding='utf-8').splitlines():
        if parsed := _parse(ln):
            out[parsed[0]] = parsed[1]
    return out


def _pyproject_specs(group: str | None = None) -> dict[str, str]:
    with (ROOT / 'pyproject.toml').open('rb') as f:
        project = tomllib.load(f)['project']
    deps = project['dependencies'] if group is None else project['optional-dependencies'][group]
    return dict(p for d in deps if (p := _parse(d)))


def _runtime_manifests() -> dict[str, dict[str, str]]:
    return {
        'requirements.txt': _file_specs('requirements.txt'),
        'pyproject.toml': _pyproject_specs(),
        'requirements-test.txt': _file_specs('requirements-test.txt'),
    }


def _family(name: str) -> str:
    return _VERSION_FAMILIES.get(name, name)


def test_exact_pins_are_identical_across_runtime_manifests() -> None:
    versions: dict[str, dict[str, str]] = {}
    for manifest, specs in _runtime_manifests().items():
        for name, spec in specs.items():
            if spec.startswith('==') and name not in _NO_SHARED_VERSION:
                versions.setdefault(_family(name), {})[manifest] = spec
    differing = {n: v for n, v in versions.items() if len(set(v.values())) > 1}
    assert not differing, differing


def test_dev_tools_match_between_pyproject_and_the_test_requirements() -> None:
    dev = _pyproject_specs('dev')
    tests = _file_specs('requirements-test.txt')
    differing = {n: (s, tests[n]) for n, s in dev.items() if n in tests and s != tests[n]}
    assert not differing, differing


def test_ruff_pin_equals_the_pre_commit_rev() -> None:
    config = (ROOT / '.pre-commit-config.yaml').read_text(encoding='utf-8')
    (rev,) = re.findall(r'ruff-pre-commit\s*\n\s*rev:\s*v(\S+)', config)
    assert _pyproject_specs('dev')['ruff'] == f'=={rev}'
    assert _file_specs('requirements-test.txt')['ruff'] == f'=={rev}'


def _loose_runtime_lines() -> set[str]:
    """Names with a floor, a range or no version in any pinned manifest."""
    loose = set()
    groups = [*_runtime_manifests().values(), _pyproject_specs('dev')]
    for specs in groups:
        loose |= {n for n, s in specs.items() if not re.fullmatch(r'==[^,<>=!~]+', s)}
    return loose


def test_no_floor_and_no_unpinned_runtime_line_outside_part_b() -> None:
    unexpected = sorted(_loose_runtime_lines() - _UNPINNED_UNTIL_PART_B)
    assert not unexpected, f'pin these with == (owner rule, 01_package_pins.md): {unexpected}'


def test_the_part_b_allowlist_only_shrinks() -> None:
    stale = sorted(_UNPINNED_UNTIL_PART_B - _loose_runtime_lines())
    assert not stale, f'already pinned everywhere, delete from the allowlist: {stale}'


def test_pydantic_is_declared_explicitly_in_every_runtime_manifest() -> None:
    """fastapi only constrains pydantic loosely; the OpenAPI contract depends on the exact one."""
    for manifest, specs in _runtime_manifests().items():
        assert specs.get('pydantic', '').startswith('=='), manifest
