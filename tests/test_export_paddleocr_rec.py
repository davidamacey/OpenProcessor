"""Tests for the self-contained TensorRT build in
``export/export_paddleocr_rec.py``.

F-17 (fresh-start E2E findings 2026-09-25): this script runs inside
``yolo-api``, which has no docker CLI or docker.sock, and the old
``convert_to_tensorrt_via_trtexec`` shelled out to
``docker exec <TRITON_CONTAINER> trtexec ...`` -- always failing with
"triton-server container is not running" (the container is renamed
``${COMPOSE_PROJECT_NAME}-triton`` post-G-01, and yolo-api can't run
docker commands regardless). ``convert_to_tensorrt_via_python_api``
replaces it with the TensorRT Python API directly (same dependency
``export/export_models.py`` already uses in-process), so these tests
fake ``tensorrt`` itself -- no GPU, no real engine build, no docker.

Round 2 (F-17 regression + destructive rebuild, same findings doc):
(a) the fake ``tensorrt`` used here originally always exposed
``NetworkDefinitionCreationFlag.EXPLICIT_BATCH``, matching TRT <= 10 and
masking that the real image's TRT 11.1 has no such attribute --
``TestExplicitBatchAcrossTrtVersions`` covers both shapes. (b) the old
code unlinked ``plan_path`` before attempting a rebuild, so a failed
rebuild destroyed a working install; the tests below assert the
opposite of what they used to: a stale plan MUST survive a failed
build.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock


if TYPE_CHECKING:
    import pytest

EXPORT_DIR = Path(__file__).resolve().parents[1] / 'export'
if str(EXPORT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPORT_DIR))

import export_paddleocr_rec as rec_export  # noqa: E402


class _FakeParser:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.num_errors = 0 if ok else 1
        self.parsed_bytes: bytes | None = None

    def parse(self, data: bytes) -> bool:
        self.parsed_bytes = data
        return self.ok

    def get_error(self, _i: int) -> str:
        return 'fake parse error'


class _FakeEngine:
    """Stands in for a deserialized ``ICudaEngine`` -- truthy, no I/O."""


def _install_fake_tensorrt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    parse_ok: bool,
    engine: bytes | None,
    explicit_batch_flag_exists: bool = True,
    deserializes_ok: bool = True,
):
    """Inject a minimal fake ``tensorrt`` module into sys.modules covering
    exactly the surface convert_to_tensorrt_via_python_api /
    trt_utils.atomic_write_plan touch.

    ``explicit_batch_flag_exists=False`` simulates TRT 11, which removed
    ``NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` entirely (round-2
    regression: the old code did a bare attribute lookup and crashed).
    ``trt_utils`` is force-reimported so it re-binds to *this* fake module
    instead of a previous test's (trt_utils imports tensorrt lazily,
    per-call, precisely to avoid stale bindings across fakes).
    """
    fake_parser = _FakeParser(ok=parse_ok)

    fake_builder = MagicMock()
    fake_builder.create_builder_config.return_value = MagicMock()
    fake_builder.create_network.return_value = MagicMock()
    fake_builder.create_optimization_profile.return_value = MagicMock()
    fake_builder.build_serialized_network.return_value = engine

    fake_runtime = MagicMock()
    fake_runtime.deserialize_cuda_engine.return_value = _FakeEngine() if deserializes_ok else None

    ns_kwargs = {}
    if explicit_batch_flag_exists:
        ns_kwargs['EXPLICIT_BATCH'] = 0

    fake_trt = types.SimpleNamespace(
        Logger=lambda _level=None: MagicMock(),
        init_libnvinfer_plugins=lambda _logger, _ns: None,
        Builder=lambda _logger: fake_builder,
        Runtime=lambda _logger: fake_runtime,
        MemoryPoolType=types.SimpleNamespace(WORKSPACE=0),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(**ns_kwargs),
        OnnxParser=lambda _network, _logger: fake_parser,
    )
    fake_trt.Logger.INFO = 0
    fake_trt.Logger.ERROR = 0
    monkeypatch.setitem(sys.modules, 'tensorrt', fake_trt)
    monkeypatch.delitem(sys.modules, 'trt_utils', raising=False)
    # Never make a real network call in a unit test.
    monkeypatch.setattr(rec_export, 'unload_models_for_memory', lambda: None)
    return fake_builder, fake_parser


class TestExplicitBatchAcrossTrtVersions:
    """Regression coverage for the F-17 round-2 crash: a bare
    ``trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH`` lookup raises
    ``AttributeError`` on TRT 11, where the flag was removed entirely."""

    def test_builds_when_explicit_batch_flag_is_absent_trt11(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'ppocr_rec_v5_mobile.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        plan_path = tmp_path / 'models' / 'paddleocr_rec_trt' / '1' / 'model.plan'
        _install_fake_tensorrt(
            monkeypatch, parse_ok=True, engine=b'fake-engine', explicit_batch_flag_exists=False
        )

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result == plan_path
        assert plan_path.read_bytes() == b'fake-engine'

    def test_builds_when_explicit_batch_flag_is_present_trt10(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'ppocr_rec_v5_mobile.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        plan_path = tmp_path / 'models' / 'paddleocr_rec_trt' / '1' / 'model.plan'
        _install_fake_tensorrt(
            monkeypatch, parse_ok=True, engine=b'fake-engine', explicit_batch_flag_exists=True
        )

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result == plan_path
        assert plan_path.read_bytes() == b'fake-engine'


class TestConvertToTensorrtViaPythonApi:
    def test_builds_the_engine_with_no_docker_dependency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'ppocr_rec_v5_mobile.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        plan_path = tmp_path / 'models' / 'paddleocr_rec_trt' / '1' / 'model.plan'

        builder, parser = _install_fake_tensorrt(monkeypatch, parse_ok=True, engine=b'fake-engine')

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result == plan_path
        assert plan_path.read_bytes() == b'fake-engine'
        assert parser.parsed_bytes == b'fake-onnx-bytes'
        # The dynamic shape profile must cover the full advertised range.
        (call,) = builder.create_optimization_profile.return_value.set_shape.call_args_list
        _, kwargs = call
        assert kwargs['min'] == (
            rec_export.MIN_BATCH,
            3,
            rec_export.REC_HEIGHT,
            rec_export.MIN_WIDTH,
        )
        assert kwargs['max'] == (
            rec_export.MAX_BATCH,
            3,
            rec_export.REC_HEIGHT,
            rec_export.MAX_WIDTH,
        )

    def test_onnx_parse_failure_returns_none_not_an_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'bad.onnx'
        onnx_path.write_bytes(b'not-a-real-onnx-graph')
        plan_path = tmp_path / 'model.plan'
        _install_fake_tensorrt(monkeypatch, parse_ok=False, engine=b'unused')

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result is None
        assert not plan_path.exists()

    def test_engine_build_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        onnx_path = tmp_path / 'ok.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        plan_path = tmp_path / 'model.plan'
        _install_fake_tensorrt(monkeypatch, parse_ok=True, engine=None)

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result is None

    def test_stale_plan_survives_a_failed_rebuild(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """F-17(b) round 2: a leftover, WORKING plan from a previous run
        must survive a failed rebuild -- the old code unlinked it before
        even attempting the build, so a bad rerun destroyed a good
        install."""
        plan_path = tmp_path / 'model.plan'
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_bytes(b'working-engine-from-a-previous-run')
        onnx_path = tmp_path / 'ok.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        _install_fake_tensorrt(monkeypatch, parse_ok=True, engine=None)

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result is None
        assert plan_path.exists()
        assert plan_path.read_bytes() == b'working-engine-from-a-previous-run'

    def test_stale_plan_survives_an_engine_that_fails_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``build_serialized_network`` can return non-None bytes that
        still don't deserialize into a usable engine; that must not
        clobber a working plan either."""
        plan_path = tmp_path / 'model.plan'
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_bytes(b'working-engine-from-a-previous-run')
        onnx_path = tmp_path / 'ok.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        _install_fake_tensorrt(
            monkeypatch, parse_ok=True, engine=b'garbage-engine-bytes', deserializes_ok=False
        )

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result is None
        assert plan_path.read_bytes() == b'working-engine-from-a-previous-run'

    def test_successful_rebuild_replaces_a_stale_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan_path = tmp_path / 'model.plan'
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_bytes(b'old-engine')
        onnx_path = tmp_path / 'ok.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        _install_fake_tensorrt(monkeypatch, parse_ok=True, engine=b'new-engine')

        result = rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert result == plan_path
        assert plan_path.read_bytes() == b'new-engine'

    def test_no_leftover_temp_file_after_a_successful_build(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan_path = tmp_path / 'models' / 'paddleocr_rec_trt' / '1' / 'model.plan'
        onnx_path = tmp_path / 'ok.onnx'
        onnx_path.write_bytes(b'fake-onnx-bytes')
        _install_fake_tensorrt(monkeypatch, parse_ok=True, engine=b'fake-engine')

        rec_export.convert_to_tensorrt_via_python_api(onnx_path, plan_path)

        assert sorted(p.name for p in plan_path.parent.iterdir()) == ['model.plan']


def test_main_fails_loudly_when_tensorrt_conversion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F-17's fail-open half: main() used to always return 0, even when
    the whole point of the run (an installed model.plan) never happened."""
    monkeypatch.setattr(rec_export, 'check_onnx_exists', lambda: True)
    monkeypatch.setattr(rec_export, 'ONNX_PATH', tmp_path / 'ppocr_rec_v5_mobile.onnx')
    rec_export.ONNX_PATH.write_bytes(b'fake-onnx-bytes')
    monkeypatch.setattr(rec_export, 'verify_onnx_model', lambda _p: {'ok': True})
    monkeypatch.setattr(rec_export, 'test_onnx_inference', lambda _p: True)
    monkeypatch.setattr(rec_export, 'convert_to_tensorrt_via_python_api', lambda *_a, **_k: None)
    monkeypatch.setattr(sys, 'argv', ['export_paddleocr_rec.py'])

    assert rec_export.main() == 1
