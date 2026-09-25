"""Tests for the shared TensorRT builder helpers in ``export/trt_utils.py``.

F-17 round 2 (fresh-start E2E findings 2026-09-25): ``export_paddleocr_rec.py``
used to build a raw ``NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` lookup
(removed in TRT 11) and wrote engines straight to the destination
``model.plan``, deleting the old one first -- a failed rebuild left the
model with no plan at all. ``create_explicit_network`` and
``atomic_write_plan`` are the shared fix; these tests fake ``tensorrt``
itself so they run without a GPU or the real package.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))


class _FakeEngine:
    """Stands in for a deserialized ``ICudaEngine`` -- truthy, no I/O."""


def _install_fake_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    explicit_batch_flag_exists: bool,
    deserializes_ok: bool,
):
    fake_runtime = MagicMock()
    fake_runtime.deserialize_cuda_engine.return_value = _FakeEngine() if deserializes_ok else None

    ns_kwargs = {}
    if explicit_batch_flag_exists:
        ns_kwargs['EXPLICIT_BATCH'] = 0

    fake_trt = types.SimpleNamespace(
        Logger=lambda _level=None: MagicMock(),
        init_libnvinfer_plugins=lambda _logger, _ns: None,
        Runtime=lambda _logger: fake_runtime,
        NetworkDefinitionCreationFlag=types.SimpleNamespace(**ns_kwargs),
    )
    fake_trt.Logger.ERROR = 0
    monkeypatch.setitem(sys.modules, 'tensorrt', fake_trt)
    monkeypatch.delitem(sys.modules, 'trt_utils', raising=False)
    return fake_trt


class TestCreateExplicitNetwork:
    def test_passes_the_flag_when_present_trt10(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=True)
        import trt_utils

        builder = MagicMock()
        trt_utils.create_explicit_network(builder)

        builder.create_network.assert_called_once_with(1)

    def test_no_attribute_error_when_flag_is_removed_trt11(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=False, deserializes_ok=True)
        import trt_utils

        builder = MagicMock()
        trt_utils.create_explicit_network(builder)

        builder.create_network.assert_called_once_with(0)


class TestAtomicWritePlan:
    def test_writes_a_valid_engine_and_returns_the_plan_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=True)
        import trt_utils

        plan_path = tmp_path / 'models' / 'some_model' / '1' / 'model.plan'
        result = trt_utils.atomic_write_plan(b'a-real-engine', plan_path)

        assert result == plan_path
        assert plan_path.read_bytes() == b'a-real-engine'
        # No leftover temp file next to it.
        assert sorted(p.name for p in plan_path.parent.iterdir()) == ['model.plan']

    def test_rejects_an_engine_that_fails_to_deserialize_and_keeps_the_old_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=False)
        import trt_utils

        plan_path = tmp_path / 'model.plan'
        plan_path.write_bytes(b'old-working-engine')

        with pytest.raises(ValueError, match='deserialize'):
            trt_utils.atomic_write_plan(b'garbage', plan_path)

        assert plan_path.read_bytes() == b'old-working-engine'

    def test_rejects_empty_bytes_without_touching_an_existing_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_tensorrt(monkeypatch, explicit_batch_flag_exists=True, deserializes_ok=True)
        import trt_utils

        plan_path = tmp_path / 'model.plan'
        plan_path.write_bytes(b'old-working-engine')

        with pytest.raises(ValueError, match='empty'):
            trt_utils.atomic_write_plan(b'', plan_path)

        assert plan_path.read_bytes() == b'old-working-engine'
