"""Engine settings: ``OP_FRAME_ENGINE_*`` read once; a malformed value refuses the start."""

from __future__ import annotations

import subprocess
import sys

import pytest

from frame_engine.settings import EngineSettings, SettingsError


def test_defaults_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [n for n in __import__('os').environ if n.startswith('OP_FRAME_ENGINE_')]:
        monkeypatch.delenv(name)
    s = EngineSettings.from_env()
    assert s == EngineSettings()
    assert (s.transport, s.region_mib, s.workers, s.budget_wait_seconds) == ('cuda', 1024, 16, 4.0)
    assert 4 * s.budget_wait_seconds <= 20  # one try and three API retries within the old wait
    # every model call ends well inside the request, and the request well inside the API's call
    assert (s.triton_timeout_seconds, s.request_deadline_seconds) == (15.0, 60.0)


def test_values_are_read_and_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_FRAME_ENGINE_TRANSPORT', 'body')
    monkeypatch.setenv('OP_FRAME_ENGINE_WORKERS', '8')
    monkeypatch.setenv('OP_FRAME_ENGINE_STAGE_SYNC', 'yes')
    monkeypatch.setenv('OP_FRAME_ENGINE_TRITON_URL', 'triton-gpu2:8001')
    monkeypatch.setenv('OP_FRAME_ENGINE_COLOUR_TABLES', '5')
    s = EngineSettings.from_env()
    assert (s.transport, s.workers, s.stage_sync, s.triton_url, s.colour_tables) == (
        'body', 8, True, 'triton-gpu2:8001', 5,
    )  # fmt: skip


@pytest.mark.parametrize(
    ('name', 'value'),
    [
        ('TRANSPORT', 'system'),
        ('WORKERS', '0'),
        ('WORKERS', 'many'),
        ('REGION_MIB', '8'),
        ('STAGE_SYNC', 'maybe'),
        ('INSTANCE', 'fe 0'),
        ('BUDGET_MIB', '8192'),  # above POOL_MIB
        ('TRITON_TIMEOUT_SECONDS', '90'),  # above REQUEST_DEADLINE_SECONDS
        ('REQUEST_DEADLINE_SECONDS', '0.5'),
        ('COLOUR_TABLES', '0'),
        ('COLOUR_TABLES', '65'),
    ],
)
def test_a_bad_value_refuses_the_start(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(f'OP_FRAME_ENGINE_{name}', value)
    with pytest.raises(SettingsError, match=f'OP_FRAME_ENGINE_{name}'):
        EngineSettings.from_env()


def test_the_wire_module_imports_without_the_api_config() -> None:
    """The engine image carries src/domain and, of src/utils, only the colour-profile rule
    (docker/frame-engine)."""
    code = (
        'import sys, src.domain.frame_engine_wire, src.domain.image_geometry,'
        ' frame_engine.front, frame_engine.transport, frame_engine.prep_geometry,'
        ' src.utils.icc_srgb, src.utils.icc_curves;'
        "bad = [m for m in sys.modules if m.startswith(('pydantic', 'fastapi'))"
        " or (m.startswith('src.') and not m.startswith('src.domain')"
        " and m not in ('src.utils', 'src.utils.icc_srgb', 'src.utils.icc_curves'))];"
        'print(bad); raise SystemExit(1 if bad else 0)'
    )
    run = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
