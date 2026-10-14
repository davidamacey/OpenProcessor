"""The HTTP timeout default and unload(): directory removal, refusal on a failed unload, endpoint used."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

from curation.triton_promote._helpers import _make_promoted_model_dir, _promoter
from src.services.training.promote_errors import ModelNotPromotedError, TritonUnloadError
from src.services.training.triton_promote import TritonPromoter, UnloadResult


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# _trigger_load timeout raised (P1-11 timeout bump, not a timeout-specific
# code branch — just confirms the constructor default changed).
# =============================================================================


def test_default_http_timeout_raised_above_30s() -> None:
    """Before the fix: 30s default, too short for a real TRT JIT-build."""
    promoter = TritonPromoter()
    assert promoter.http_timeout > 30.0


# =============================================================================
# unload() — follow-up gap 2 (Appendix D item 3, 2026-09-11)
#
# Mirrors promote()'s test style: a disposable tmp_path stands in for the
# Triton model repo, and _trigger_unload is mocked so no real Triton HTTP
# call ever happens. Never touches the real models/ directory.
# =============================================================================


@pytest.mark.asyncio
async def test_unload_happy_path_calls_triton_and_removes_directory(
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path: Triton confirms unload (mocked — no real HTTP call),
    then the model repo directory is actually removed from disk."""
    name = 'op_smoke_unload_v1'
    model_dir = _make_promoted_model_dir(scratch_models_dir, name)
    assert model_dir.is_dir()

    trigger = AsyncMock(return_value=True)
    monkeypatch.setattr(TritonPromoter, '_trigger_unload', trigger)

    promoter = _promoter(scratch_models_dir)
    result: UnloadResult = await promoter.unload(name)

    trigger.assert_awaited_once_with(name)
    assert result.triton_name == name
    assert result.triton_unloaded is True
    assert result.directory_removed is True
    assert not model_dir.exists(), 'model repo directory must be gone after a successful unload'


@pytest.mark.asyncio
async def test_unload_missing_model_dir_raises_not_promoted(scratch_models_dir: Path) -> None:
    promoter = _promoter(scratch_models_dir)
    with pytest.raises(ModelNotPromotedError):
        await promoter.unload('never_promoted_model')


@pytest.mark.asyncio
async def test_unload_refuses_to_delete_when_triton_unload_fails(
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before this method existed there was no code path at all; the
    invariant that matters going forward is that a failed/unreachable
    Triton unload must NEVER be followed by an rmtree — deleting files out
    from under a model Triton still thinks is loaded is exactly the kind
    of irreversible mistake P1-10 already burned once (promote's old
    rmtree-before-copy bug)."""
    name = 'op_smoke_unload_v1'
    model_dir = _make_promoted_model_dir(scratch_models_dir, name)

    monkeypatch.setattr(TritonPromoter, '_trigger_unload', AsyncMock(return_value=False))

    promoter = _promoter(scratch_models_dir)
    with pytest.raises(TritonUnloadError):
        await promoter.unload(name)

    assert model_dir.is_dir(), 'directory must survive a failed/unconfirmed Triton unload'
    assert (model_dir / '1' / 'model.onnx').read_bytes() == b'THROWAWAY-SMOKE-WEIGHTS'


@pytest.mark.asyncio
async def test_unload_posts_to_the_unload_endpoint_not_load(
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirms the real (unmocked) _trigger_unload hits
    `/v2/repository/models/<name>/unload`, not `/load` — a copy-paste from
    _trigger_load would silently re-load the model instead of releasing it."""
    name = 'op_smoke_unload_v1'
    _make_promoted_model_dir(scratch_models_dir, name)

    requested_urls: list[str] = []

    class _FakeResponse:
        status_code = 200
        text = ''

    class _FakeAsyncClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *_args: Any) -> None:
            return None

        async def post(self, url: str) -> _FakeResponse:
            requested_urls.append(url)
            return _FakeResponse()

    monkeypatch.setattr('src.services.training.triton_promote.httpx.AsyncClient', _FakeAsyncClient)

    promoter = _promoter(scratch_models_dir)
    await promoter.unload(name)

    assert len(requested_urls) == 1
    assert requested_urls[0].endswith(f'/v2/repository/models/{name}/unload')
