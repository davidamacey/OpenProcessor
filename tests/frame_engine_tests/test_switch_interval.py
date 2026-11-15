"""The interpreter's thread-switch interval (``OP_FRAME_ENGINE_SWITCH_INTERVAL_MS``): the gRPC loop
thread waits up to this long for the interpreter lock behind 32 busy worker threads."""

from __future__ import annotations

import sys

import pytest

from frame_engine.settings import EngineSettings, SettingsError


def test_the_default_is_a_tenth_of_a_millisecond(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_FRAME_ENGINE_SWITCH_INTERVAL_MS', raising=False)
    assert EngineSettings.from_env().switch_interval_ms == 0.1


@pytest.mark.parametrize('value', ['-1', '101', 'fast'])
def test_a_bad_value_refuses_the_start(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv('OP_FRAME_ENGINE_SWITCH_INTERVAL_MS', value)
    with pytest.raises(SettingsError, match='OP_FRAME_ENGINE_SWITCH_INTERVAL_MS'):
        EngineSettings.from_env()


def test_the_interval_is_applied_to_the_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    monkeypatch.setattr(sys, 'setswitchinterval', seen.append)
    EngineSettings().apply_to_interpreter()
    EngineSettings(switch_interval_ms=0.5).apply_to_interpreter()
    EngineSettings(switch_interval_ms=0).apply_to_interpreter()  # 0 leaves Python's default
    assert seen == [pytest.approx(0.0001), pytest.approx(0.0005)]
    monkeypatch.setenv('OP_FRAME_ENGINE_SWITCH_INTERVAL_MS', '2')
    assert EngineSettings.from_env().switch_interval_ms == 2.0
