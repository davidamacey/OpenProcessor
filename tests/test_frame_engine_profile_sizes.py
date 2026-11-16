"""The frame engine's sizes per installer profile.

The engine runs on Triton's card in every profile. A profile whose card has room for the GPU
decode gives it its working sizes (the region every model input is a slice of, the CuPy pool cap,
the frame budget); one without room sets ``ingest_gpu_decode`` false and runs the engine at the
smallest sizes it starts with and one decode lane and one worker, where it is never called.
``setup-openprocessor.sh`` mirrors the values (``_frame_engine_sizes_for_profile``) and
``openprocessor gpu plan`` sizes the engine from them (``scripts/lib/cli_gpu.sh``).

The minimal profile's models take its whole 8 GB budget, so the idle engine (a core service) runs
about 0.35 GB over it; docs-site/docs/deployment/gpu-sizing.mdx states that allowance.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from frame_engine.settings import EngineSettings


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'codegen'))
import frame_engine_lane_budget as lane_budget


ROOT = Path(__file__).resolve().parent.parent
PROFILES = ROOT / 'config_templates' / 'profiles'
NAMES = ('minimal', 'small', 'standard', 'full')
KEYS = (
    'frame_engine_region_mib',
    'frame_engine_pool_mib',
    'frame_engine_budget_mib',
    'frame_engine_lanes',
    'frame_engine_workers',
)
SETTINGS = ('region_mib', 'pool_mib', 'budget_mib', 'lanes', 'workers')
#: The free memory each profile's own models need (setup-openprocessor.sh _triton_need_gb).
MODELS_MIB = {'minimal': 8192, 'small': 10240, 'standard': 12288, 'full': 17408}
WORKING = (1024, 4096, 2048, 16, 16)
SMALLEST = (16, 64, 64, 1, 1)
#: The documented overrun of the minimal profile's model budget by the idle engine (gpu-sizing).
MINIMAL_ALLOWANCE_MIB = 360
SIZING_DOC = ROOT / 'docs-site' / 'docs' / 'deployment' / 'gpu-sizing.mdx'


def _profile(name: str) -> dict:
    return json.loads((PROFILES / f'{name}.json').read_text())


def _sizes(name: str) -> tuple[int, ...]:
    p = _profile(name)
    return tuple(p[key] for key in KEYS)


def _cli_gpu_mib(name: str) -> int:
    text = (ROOT / 'scripts' / 'lib' / 'cli_gpu.sh').read_text()
    m = re.search(rf'^{name}=(\d+)$', text, re.M)
    assert m
    return int(m.group(1))


def _engine_mib(sizes: tuple[int, ...]) -> int:
    """Working (the GPU decode on): context, region and pool cap."""
    region, pool = sizes[0], sizes[1]
    return _cli_gpu_mib('FRAME_ENGINE_CONTEXT_MIB') + region + pool


def _idle_engine_mib(sizes: tuple[int, ...]) -> int:
    """Never called (the GPU decode off): idle context, region and the frame budget it reserves."""
    region, budget = sizes[0], sizes[2]
    return _cli_gpu_mib('FRAME_ENGINE_IDLE_CONTEXT_MIB') + region + budget


def _resident_mib(name: str) -> int:
    sizes = _sizes(name)
    return _engine_mib(sizes) if _profile(name)['ingest_gpu_decode'] else _idle_engine_mib(sizes)


def _sh(script: str) -> str:
    done = subprocess.run(
        ['bash', '-c', f'OP_SOURCE_ONLY=1 source "{ROOT}/setup-openprocessor.sh"; {script}'],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return done.stdout.strip()


@pytest.mark.parametrize('name', NAMES)
def test_every_profile_sizes_the_engine_and_has_no_prep_instances(name: str) -> None:
    profile = _profile(name)
    assert 'gpu_image_prep_instances' not in profile
    assert _sizes(name) == (WORKING if profile['ingest_gpu_decode'] else SMALLEST)


@pytest.mark.parametrize('name', NAMES)
def test_the_engine_starts_with_the_profile_sizes(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in zip(KEYS, _sizes(name), strict=True):
        monkeypatch.setenv('OP_' + key.upper(), str(value))
    s = EngineSettings.from_env()
    assert tuple(getattr(s, field) for field in SETTINGS) == _sizes(name)


@pytest.mark.parametrize('name', NAMES)
def test_the_decode_is_on_exactly_where_the_working_engine_fits_the_smallest_card(
    name: str,
) -> None:
    floor = int(_sh(f'_profile_min_free_gb {name}')) * 1024
    fits = MODELS_MIB[name] + _engine_mib(WORKING) <= floor
    assert _profile(name)['ingest_gpu_decode'] is fits


@pytest.mark.parametrize('name', NAMES)
def test_the_engine_at_the_profile_sizes_fits_beside_the_models_on_the_profiles_smallest_card(
    name: str,
) -> None:
    """The least free memory the installer picks the profile for (``_profile_for_free_gb``;
    minimal: what its models need, ``_triton_need_gb``) holds the models and the engine, except
    the documented allowance on minimal."""
    floor_gb = {'minimal': int(_sh('_triton_need_gb minimal')), 'small': 12, 'standard': 16}
    floor_gb['full'] = int(_sh('_profile_min_free_gb full'))
    assert _sh(f'_profile_for_free_gb {floor_gb[name]}') == name
    allowance = MINIMAL_ALLOWANCE_MIB if name == 'minimal' else 0
    assert MODELS_MIB[name] + _resident_mib(name) <= floor_gb[name] * 1024 + allowance


def test_the_minimal_overrun_is_the_measured_idle_engine_and_is_documented() -> None:
    over = (
        MODELS_MIB['minimal']
        + _resident_mib('minimal')
        - int(_sh('_triton_need_gb minimal')) * 1024
    )
    assert 0 < over <= MINIMAL_ALLOWANCE_MIB
    text = ' '.join(SIZING_DOC.read_text().split())
    assert f'{over} MiB' in text
    assert 'about 0.35 GB over' in text


@pytest.mark.parametrize('name', NAMES)
def test_the_setup_mirror_equals_the_profile_file(name: str) -> None:
    assert tuple(int(v) for v in _sh(f'_frame_engine_sizes_for_profile {name}').split()) == (
        _sizes(name)
    )


@pytest.mark.parametrize('name', NAMES)
def test_each_profiles_lanes_are_what_the_lane_arithmetic_gives(name: str) -> None:
    """One lane where the decode is off; else what fits on the profile's smallest card beside its
    models (``frame_engine_lane_budget.profile_lanes``), at most the cap."""
    profile = _profile(name)
    floor_mib = int(_sh(f'_profile_min_free_gb {name}')) * 1024
    want = lane_budget.profile_lanes(
        profile['ingest_gpu_decode'],
        floor_mib,
        MODELS_MIB[name],
        profile['frame_engine_region_mib'],
        profile['frame_engine_pool_mib'],
    )
    assert profile['frame_engine_lanes'] == want


def test_the_full_profiles_memory_does_not_lower_its_lanes() -> None:
    free = (40 * 1024 - MODELS_MIB['full'] - lane_budget.CONTEXT_MIB - 1024) * lane_budget.MIB
    fit = lane_budget.lanes_that_fit(free, 4096 * lane_budget.MIB, lane_budget.SIZED_FOR_MEGAPIXELS)
    assert fit.fits
    assert fit.lanes >= lane_budget.PROFILE_LANE_CAP


def test_a_card_without_room_gets_the_minimum_and_says_so() -> None:
    fit = lane_budget.lanes_that_fit(10 * lane_budget.MIB, 4096 * lane_budget.MIB, 26.0)
    assert (fit.lanes, fit.fits) == (1, False)
