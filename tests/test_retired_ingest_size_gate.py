"""``OP_INGEST_GPU_MIN_MEGAPIXELS`` is gone: the frame engine decodes every image, so the API
applies no size gate. A process that still has it set refuses to start instead of ignoring it."""

from __future__ import annotations

import pytest

from src.config.retired_env import RETIRED_ENV, reject_retired_env


NAME = 'OP_INGEST_GPU_MIN_MEGAPIXELS'


def test_the_size_gate_variable_is_retired_with_the_engines_own_setting_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert RETIRED_ENV[NAME].startswith('removed:')
    assert 'OP_FRAME_ENGINE_MIN_MEGAPIXELS' in RETIRED_ENV[NAME]
    monkeypatch.setenv(NAME, '3.0')
    with pytest.raises(RuntimeError, match=NAME):
        reject_retired_env()
