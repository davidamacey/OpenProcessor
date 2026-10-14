"""Model repo and URL resolution at construction time, and reload_promoted_models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import pytest

from curation.triton_promote._helpers import _make_promoted_model_dir, _promoter
from src.services.training.triton_promote import TritonPromoter
from src.services.training.triton_repo import (
    DEFAULT_TRITON_HTTP_URL,
    DEFAULT_TRITON_MODELS_DIR,
    resolve_triton_http_url,
    resolve_triton_models_dir,
)


if TYPE_CHECKING:
    from pathlib import Path


# =============================================================================
# OP_TRITON_MODEL_REPO / OP_TRITON_HTTP_URL resolved at construction
# time, not baked in as an import-time constant. A deployment overlay that
# mounts the Triton model repo somewhere other than /app/models (the
# private deployment mounts it at /models) previously had promote() write
# into a directory Triton never reads, with no error at all -- the write
# "succeeds" against the container's own writable layer.
# =============================================================================


def test_default_promoter_uses_app_models_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('OP_TRITON_MODEL_REPO', raising=False)
    monkeypatch.delenv('OP_TRITON_HTTP_URL', raising=False)
    monkeypatch.delenv('TRITON_HTTP_URL', raising=False)

    promoter = TritonPromoter()

    assert promoter.triton_models_dir == DEFAULT_TRITON_MODELS_DIR
    assert promoter.triton_http_url == DEFAULT_TRITON_HTTP_URL


def test_promoter_honors_op_triton_model_repo_env_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """This is the exact bug: a deployment sets OP_TRITON_MODEL_REPO=/models
    (matching where the repo is actually mounted), but a promoter built
    with no explicit `triton_models_dir` used to always resolve to the
    hard-coded /app/models constant, regardless of the env var."""
    override = tmp_path / 'models'
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', str(override))

    promoter = TritonPromoter()

    assert promoter.triton_models_dir == override
    assert promoter.triton_models_dir != DEFAULT_TRITON_MODELS_DIR


def test_promoter_honors_op_triton_http_url_env_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('OP_TRITON_HTTP_URL', 'http://custom-triton:9000')

    promoter = TritonPromoter()

    assert promoter.triton_http_url == 'http://custom-triton:9000'


def test_explicit_constructor_args_still_win_over_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('OP_TRITON_MODEL_REPO', '/should-not-be-used')
    explicit = tmp_path / 'explicit'

    promoter = TritonPromoter(triton_models_dir=explicit, triton_http_url='http://explicit:1')

    assert promoter.triton_models_dir == explicit
    assert promoter.triton_http_url == 'http://explicit:1'


def test_resolve_triton_http_url_falls_back_to_legacy_env_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """src/routers/curation/models.py previously read the unprefixed
    TRITON_HTTP_URL directly while the trainer and env.template document
    OP_TRITON_HTTP_URL -- the two names diverged. resolve_triton_http_url
    is now the one place both routers and the promoter go through."""
    monkeypatch.delenv('OP_TRITON_HTTP_URL', raising=False)
    monkeypatch.setenv('TRITON_HTTP_URL', 'http://legacy-name:8000')

    assert resolve_triton_http_url() == 'http://legacy-name:8000'


def test_resolve_triton_models_dir_matches_promoter_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('OP_TRITON_MODEL_REPO', raising=False)
    assert resolve_triton_models_dir() == DEFAULT_TRITON_MODELS_DIR


# =============================================================================
# reload_promoted_models
# =============================================================================


class _FakeIndexAndLoadClient:
    """Fake httpx.AsyncClient covering /v2/repository/index (GET-like POST)
    and /v2/repository/models/<name>/load, for reload_promoted_models."""

    def __init__(self, *, index_response: list[dict[str, Any]], load_ok: set[str]) -> None:
        self._index_response = index_response
        self._load_ok = load_ok
        self.load_calls: list[str] = []

    def __call__(self, *_args: Any, **_kwargs: Any) -> _FakeIndexAndLoadClient:
        return self

    async def __aenter__(self) -> _FakeIndexAndLoadClient:
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str) -> Any:
        class _Resp:
            def __init__(self, status_code: int, payload: Any) -> None:
                self.status_code = status_code
                self._payload = payload
                self.text = str(payload)

            def json(self) -> Any:
                return self._payload

        if url.endswith('/v2/repository/index'):
            return _Resp(200, self._index_response)
        name = url.rsplit('/', 2)[1]
        self.load_calls.append(name)
        return _Resp(200 if name in self._load_ok else 500, {})


@pytest.mark.asyncio
async def test_reload_promoted_models_skips_already_ready_models(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.training.triton_reload import reload_promoted_models

    _make_promoted_model_dir(scratch_models_dir, 'op_ready_v1')
    fake_client = _FakeIndexAndLoadClient(
        index_response=[{'name': 'op_ready_v1', 'state': 'READY'}], load_ok=set()
    )
    monkeypatch.setattr('src.services.training.triton_reload.httpx.AsyncClient', fake_client)

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result == {'status': 'ok', 'reloaded': [], 'failed': []}
    assert fake_client.load_calls == []


@pytest.mark.asyncio
async def test_reload_promoted_models_reloads_unavailable_promoted_models(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.training.triton_reload import reload_promoted_models

    _make_promoted_model_dir(scratch_models_dir, 'op_stranded_v1')
    fake_client = _FakeIndexAndLoadClient(
        index_response=[{'name': 'op_stranded_v1', 'state': 'UNAVAILABLE'}],
        load_ok={'op_stranded_v1'},
    )
    monkeypatch.setattr('src.services.training.triton_reload.httpx.AsyncClient', fake_client)

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result == {'status': 'ok', 'reloaded': ['op_stranded_v1'], 'failed': []}
    assert fake_client.load_calls == ['op_stranded_v1']


@pytest.mark.asyncio
async def test_reload_promoted_models_ignores_non_promoted_model_dirs(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model dir with no promote.json (e.g. a core pipeline model) is
    never a reload target, ready or not."""
    from src.services.training.triton_reload import reload_promoted_models

    core_dir = scratch_models_dir / 'core_model'
    (core_dir / '1').mkdir(parents=True)
    (core_dir / 'config.pbtxt').write_text('# core, no promote.json')

    fake_client = _FakeIndexAndLoadClient(
        index_response=[{'name': 'core_model', 'state': 'UNAVAILABLE'}], load_ok=set()
    )
    monkeypatch.setattr('src.services.training.triton_reload.httpx.AsyncClient', fake_client)

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result == {'status': 'ok', 'reloaded': [], 'failed': []}
    assert fake_client.load_calls == []


@pytest.mark.asyncio
async def test_reload_promoted_models_is_best_effort_on_unreachable_triton(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.training.triton_reload import reload_promoted_models

    _make_promoted_model_dir(scratch_models_dir, 'op_v1')

    class _RaisingClient:
        def __call__(self, *_a: Any, **_kw: Any) -> _RaisingClient:
            return self

        async def __aenter__(self) -> _RaisingClient:
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def post(self, _url: str) -> Any:
            raise httpx.ConnectError('unreachable')

    monkeypatch.setattr('src.services.training.triton_reload.httpx.AsyncClient', _RaisingClient())

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result['status'] == 'error'
    assert result['reloaded'] == []


@pytest.mark.asyncio
async def test_unloaded_model_is_never_resurrected_by_reload_promoted_models(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 3 (fresh-start E2E findings, round 2): unload must remove the
    model from reload_promoted_models' record, not just from Triton's
    live index -- otherwise the next reconcile tick (or a Triton
    restart) would silently reload something an operator deliberately
    unloaded. unload() rmtree's the whole model dir, which is exactly
    what strands promote.json for the discovery scan below."""
    from src.services.training.triton_reload import reload_promoted_models

    name = 'op_deliberately_unloaded_v1'
    model_dir = _make_promoted_model_dir(scratch_models_dir, name)
    promoter = _promoter(scratch_models_dir)

    monkeypatch.setattr(TritonPromoter, '_trigger_unload', AsyncMock(return_value=True))
    await promoter.unload(name)
    assert not model_dir.exists()

    # Even if Triton still (implausibly) reports it UNAVAILABLE in the
    # index, there's no promote.json left to discover -- no load call.
    fake_client = _FakeIndexAndLoadClient(
        index_response=[{'name': name, 'state': 'UNAVAILABLE'}], load_ok={name}
    )
    monkeypatch.setattr('src.services.training.triton_reload.httpx.AsyncClient', fake_client)

    result = await reload_promoted_models(promoter)

    assert result == {'status': 'ok', 'reloaded': [], 'failed': []}
    assert fake_client.load_calls == []
