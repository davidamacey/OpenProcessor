"""promote(): copy-before-delete, version numbering, cold-start flag and warm-up."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from curation.triton_promote._helpers import CLASS_MAP, _promoter
from src.services.training.promote_errors import ModelNameConflictError
from src.services.training.triton_promote import TritonPromoter


if TYPE_CHECKING:
    import httpx

    from src.services.training.job_models import TrainJobStatus


# =============================================================================
# test_promote_does_not_rmtree_before_copy
# =============================================================================


@pytest.mark.asyncio
async def test_promote_does_not_rmtree_before_copy(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the fix: `shutil.rmtree(model_dir)` ran unconditionally before
    the copy, gated only on `overwrite`. A successful promote over an
    existing model must never delete anything — old version 1 must survive
    untouched and a new version 2 must appear alongside it."""
    triton_name = 'op_smoke_v1'
    model_dir = scratch_models_dir / triton_name
    v1 = model_dir / '1'
    v1.mkdir(parents=True)
    (v1 / 'model.onnx').write_bytes(b'ORIGINAL-V1-WEIGHTS')
    (model_dir / 'config.pbtxt').write_text('# original config')

    rmtree_calls: list[Path] = []
    import shutil as shutil_mod

    real_rmtree = shutil_mod.rmtree

    def _tracking_rmtree(path: Any, *args: Any, **kwargs: Any) -> Any:
        rmtree_calls.append(Path(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr('src.services.training.triton_promote.shutil.rmtree', _tracking_rmtree)
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))

    promoter = _promoter(scratch_models_dir)
    result = await promoter.promote(
        status=fake_status,
        triton_name=triton_name,
        class_id_to_name=CLASS_MAP,
        overwrite=True,
    )

    assert rmtree_calls == [], f'rmtree must never run on a successful promote: {rmtree_calls}'
    assert (v1 / 'model.onnx').read_bytes() == b'ORIGINAL-V1-WEIGHTS', (
        'the old version must survive completely untouched'
    )
    assert result.version == '2'
    assert (model_dir / '2' / 'model.onnx').read_bytes() == b'ONNX-V1-BYTES'


@pytest.mark.asyncio
async def test_promote_fresh_model_uses_version_1(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brand-new model name (no existing versions) still gets version 1,
    same as before — the fix only changes the *existing model* path."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = _promoter(scratch_models_dir)
    result = await promoter.promote(
        status=fake_status,
        triton_name='op_smoke_v1',
        class_id_to_name=CLASS_MAP,
    )
    assert result.version == '1'
    assert (scratch_models_dir / 'op_smoke_v1' / '1' / 'model.onnx').is_file()


@pytest.mark.asyncio
async def test_promote_result_flags_cold_start(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PromoteResult must surface the TensorRT-accelerator cold-start
    caveat (final E2E run 2026-09-26: first post-promote inference took
    ~85s because the onnxruntime+TensorRT config.pbtxt JIT-builds the
    engine on first real inference, not on /load) so a caller doesn't
    mistake it for a hang."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = _promoter(scratch_models_dir)
    result = await promoter.promote(
        status=fake_status,
        triton_name='op_smoke_cold_start',
        class_id_to_name=CLASS_MAP,
    )
    assert result.cold_start_expected_on_first_inference is True


@pytest.mark.asyncio
async def test_promote_warm_up_runs_batch_1_and_max_batch_after_the_load_and_clears_the_cold_flag(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from src.services.training import triton_promote

    calls: list[str] = []
    batches: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith('/infer'):
            import json

            batches.append(json.loads(request.content)['inputs'][0]['shape'][0])
        return httpx.Response(200, json={})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        triton_promote.httpx,
        'AsyncClient',
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    promoter = _promoter(scratch_models_dir)
    result = await promoter.promote(
        status=fake_status, triton_name='op_warm_up_ok', class_id_to_name=CLASS_MAP
    )
    assert calls == [
        '/v2/repository/models/op_warm_up_ok/load',
        '/v2/models/op_warm_up_ok/infer',
        '/v2/models/op_warm_up_ok/infer',
    ]
    assert batches == [1, 8]  # batch 1 AND the config's max_batch_size
    assert result.cold_start_expected_on_first_inference is False


@pytest.mark.asyncio
async def test_promote_warm_up_failure_keeps_the_cold_flag_and_does_not_fail_the_promote(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    from src.services.training import triton_promote

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500 if request.url.path.endswith('/infer') else 200)

    real = httpx.AsyncClient
    monkeypatch.setattr(
        triton_promote.httpx,
        'AsyncClient',
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    promoter = _promoter(scratch_models_dir)
    result = await promoter.promote(
        status=fake_status, triton_name='op_warm_up_bad', class_id_to_name=CLASS_MAP
    )
    assert result.triton_loaded is True
    assert result.cold_start_expected_on_first_inference is True


@pytest.mark.asyncio
async def test_promote_without_overwrite_still_conflicts(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
) -> None:
    """The 409 conflict guard must still fire for an existing *served*
    version when overwrite=False — versioning doesn't silently disable it."""
    model_dir = scratch_models_dir / 'op_smoke_v1'
    (model_dir / '1').mkdir(parents=True)
    (model_dir / '1' / 'model.onnx').write_bytes(b'x')

    promoter = _promoter(scratch_models_dir)
    with pytest.raises(ModelNameConflictError):
        await promoter.promote(
            status=fake_status,
            triton_name='op_smoke_v1',
            class_id_to_name=CLASS_MAP,
            overwrite=False,
        )
