"""Tests for src.services.training.triton_promote (Phase 9, P1-10).

TritonPromoter.promote() had zero coverage before this phase — everything
here runs against a disposable tmp_path standing in for the Triton model
repo (never the real ``models/`` directory) and mocks ``_trigger_load`` so
no real Triton HTTP call is made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from src.services.training.jobs import TrainJobStatus
from src.services.training.triton_promote import (
    DEFAULT_TRITON_HTTP_URL,
    DEFAULT_TRITON_MODELS_DIR,
    ModelNameConflictError,
    ModelNotPromotedError,
    PromoteResult,
    TritonPromoter,
    TritonUnloadError,
    UnloadResult,
    resolve_triton_http_url,
    resolve_triton_models_dir,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fake_checkpoint(tmp_path: Path) -> Path:
    """A fake run dir with best.pt + best.onnx, like the trainer produces."""
    run_dir = tmp_path / 'jobs' / 'job-abc'
    run_dir.mkdir(parents=True)
    (run_dir / 'best.pt').write_bytes(b'not-a-real-checkpoint')
    (run_dir / 'best.onnx').write_bytes(b'ONNX-V1-BYTES')
    return run_dir / 'best.pt'


@pytest.fixture
def fake_status(fake_checkpoint: Path) -> TrainJobStatus:
    return TrainJobStatus(
        job_id='job-abc',
        state='finished',
        checkpoint_path=str(fake_checkpoint),
        eval={'map50': 0.9, 'per_class': []},
    )


@pytest.fixture
def scratch_models_dir(tmp_path: Path) -> Path:
    """Disposable stand-in for the Triton model repo — never the real one."""
    d = tmp_path / 'scratch_triton_models'
    d.mkdir()
    return d


def _promoter(models_dir: Path) -> TritonPromoter:
    return TritonPromoter(triton_models_dir=models_dir, triton_http_url='http://unused:8000')


CLASS_MAP = {0: 'pickup', 1: 'sedan'}


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


# =============================================================================
# test_promote_restores_previous_version_on_failure
# =============================================================================


@pytest.mark.asyncio
async def test_promote_restores_previous_version_on_failure(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the fix: a failed copy happened AFTER the existing model dir
    was already rmtree'd — the model dir would simply be gone. Now: the
    previously-serving version must remain fully intact and servable, and
    the half-written new version must not linger to confuse Triton."""
    triton_name = 'op_smoke_v1'
    model_dir = scratch_models_dir / triton_name
    v1 = model_dir / '1'
    v1.mkdir(parents=True)
    (v1 / 'model.onnx').write_bytes(b'ORIGINAL-V1-WEIGHTS')
    (model_dir / 'config.pbtxt').write_text('# original config')

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        msg = 'simulated disk failure mid-copy'
        raise OSError(msg)

    monkeypatch.setattr('src.services.training.triton_promote.shutil.copy2', _boom)
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))

    promoter = _promoter(scratch_models_dir)
    with pytest.raises(OSError, match='simulated disk failure'):
        await promoter.promote(
            status=fake_status,
            triton_name=triton_name,
            class_id_to_name=CLASS_MAP,
            overwrite=True,
        )

    # The previously-serving version was never touched — nothing to
    # "restore" because it was never at risk.
    assert (v1 / 'model.onnx').read_bytes() == b'ORIGINAL-V1-WEIGHTS'
    assert (model_dir / 'config.pbtxt').read_text() == '# original config'
    # The half-written new version must be cleaned up, not left behind.
    assert not (model_dir / '2').exists()


@pytest.mark.asyncio
async def test_promote_config_write_failure_also_rolls_back_new_version(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure writing config.pbtxt (after a successful ONNX copy) must
    also roll back the new version dir, not leave a weights-only half-model."""
    triton_name = 'op_smoke_v1'
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))

    def _boom(*_args: Any, **_kwargs: Any) -> str:
        msg = 'simulated config render failure'
        raise RuntimeError(msg)

    # triton_promote.py does `from ...yolo26_triton_config import render_config`,
    # binding the name into its own module namespace — patch it there, not on
    # the source module, or the already-imported reference won't see it.
    monkeypatch.setattr('src.services.training.triton_promote.render_config', _boom)

    promoter = _promoter(scratch_models_dir)
    with pytest.raises(RuntimeError, match='simulated config render failure'):
        await promoter.promote(
            status=fake_status,
            triton_name=triton_name,
            class_id_to_name=CLASS_MAP,
        )
    assert not (scratch_models_dir / triton_name / '1').exists()


# =============================================================================
# test_promote_writes_job_id_backpointer_and_version
# =============================================================================


@pytest.mark.asyncio
async def test_promote_writes_job_id_backpointer_and_version(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the fix: neither promote.json nor any on-disk lineage existed
    at all — given only a serving model dir, there was no way to trace it
    back to the run that produced it."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = _promoter(scratch_models_dir)
    triton_name = 'op_smoke_v1'

    result: PromoteResult = await promoter.promote(
        status=fake_status,
        triton_name=triton_name,
        class_id_to_name=CLASS_MAP,
    )

    backpointer_path = scratch_models_dir / triton_name / 'promote.json'
    assert backpointer_path.is_file()
    backpointer = json.loads(backpointer_path.read_text())
    assert backpointer['job_id'] == 'job-abc'
    assert backpointer['triton_name'] == triton_name
    assert backpointer['version'] == '1'
    assert 'promoted_at' in backpointer
    assert result.version == '1'


@pytest.mark.asyncio
async def test_promote_backpointer_written_even_when_triton_load_times_out(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The back-pointer records what's on disk, not whether Triton has
    picked it up yet — it must exist even on a fail-soft load outcome."""
    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=False))
    promoter = _promoter(scratch_models_dir)
    triton_name = 'op_smoke_v1'

    result = await promoter.promote(
        status=fake_status,
        triton_name=triton_name,
        class_id_to_name=CLASS_MAP,
    )
    assert result.triton_loaded is False
    assert (scratch_models_dir / triton_name / 'promote.json').is_file()


# =============================================================================
# test_class_remap_missing_fails_loudly
# =============================================================================


def test_class_remap_corrupt_weights_dir_file_raises_loudly(tmp_path: Path) -> None:
    """P2-7: a *present but corrupt* class_remap.json used to silently fall
    back to "full-class run" — now it's a loud, typed failure instead of a
    quiet None, since a mislabeled labels.txt is a serving-correctness bug."""
    from src.services.training.triton_promote import ClassRemapUnreadableError, resolve_class_remap

    run_dir = tmp_path / 'job-with-bad-remap'
    run_dir.mkdir()
    (run_dir / 'class_remap.json').write_text('{not valid json')
    checkpoint_path = run_dir / 'best.pt'

    with pytest.raises(ClassRemapUnreadableError):
        resolve_class_remap(
            job_id='job-with-bad-remap', checkpoint_path=checkpoint_path, manifest=None
        )


def test_class_remap_genuinely_absent_is_not_an_error(tmp_path: Path) -> None:
    """A full-class run legitimately has no class_remap anywhere — that
    resolves to the 'none' source, not an error. The caller (promote_run)
    is responsible for refusing to serve that for an actual subset run."""
    from src.services.training.triton_promote import resolve_class_remap

    run_dir = tmp_path / 'job-full-class'
    run_dir.mkdir()
    checkpoint_path = run_dir / 'best.pt'

    result = resolve_class_remap(
        job_id='job-full-class', checkpoint_path=checkpoint_path, manifest=None
    )

    assert result.source == 'none'
    assert result.mapping == {}


def test_class_remap_real_payload_shape_round_trips(tmp_path: Path) -> None:
    """subset_dataset.py's REAL payload shape (original_to_new/new_to_original/
    single_cls/names/include_classes) must parse correctly — this is the P2-7
    bug: the old parser only understood a flat {orig: new} dict or a
    {'mapping': {...}} wrapper and silently produced an empty result here."""
    from src.services.training.triton_promote import resolve_class_remap

    run_dir = tmp_path / 'job-subset'
    run_dir.mkdir()
    payload = {
        'original_to_new': {'12': 0, '81': 1},
        'new_to_original': {'0': 12, '1': 81},
        'single_cls': False,
        'names': ['pickup', 'license_plate'],
        'include_classes': [12, 81],
    }
    (run_dir / 'class_remap.json').write_text(json.dumps(payload))
    checkpoint_path = run_dir / 'best.pt'

    result = resolve_class_remap(
        job_id='job-subset', checkpoint_path=checkpoint_path, manifest=None
    )

    assert result.source == 'weights_dir'
    assert result.mapping == {12: 0, 81: 1}
    assert result.names == ['pickup', 'license_plate']
    assert result.single_cls is False


def test_class_remap_single_cls_payload(tmp_path: Path) -> None:
    from src.services.training.triton_promote import build_class_id_to_name, resolve_class_remap

    run_dir = tmp_path / 'job-single-cls'
    run_dir.mkdir()
    payload = {
        'original_to_new': {'12': 0, '81': 0},
        'new_to_original': {},
        'single_cls': True,
        'names': ['object'],
        'include_classes': [12, 81],
    }
    (run_dir / 'class_remap.json').write_text(json.dumps(payload))
    checkpoint_path = run_dir / 'best.pt'

    result = resolve_class_remap(
        job_id='job-single-cls', checkpoint_path=checkpoint_path, manifest=None
    )
    assert result.single_cls is True

    labels = build_class_id_to_name(remap=result, full_registry={12: 'pickup', 81: 'license_plate'})
    assert labels == {0: 'object'}


def test_class_remap_manifest_lineage_preferred_over_weights_dir(tmp_path: Path) -> None:
    """Every already-completed run has lineage.class_remap in its manifest,
    captured before the trainer's /tmp cleanup — this must resolve correctly
    even when there's no weights-dir file at all (older runs, pre-fix)."""
    from src.services.training.triton_promote import resolve_class_remap

    run_dir = tmp_path / 'job-old-run'
    run_dir.mkdir()
    checkpoint_path = run_dir / 'best.pt'  # no class_remap.json next to it
    manifest = {
        'lineage': {
            'class_remap': {
                'original_to_new': {'3': 0},
                'single_cls': False,
                'names': ['sedan'],
                'include_classes': [3],
            }
        }
    }

    result = resolve_class_remap(
        job_id='job-old-run', checkpoint_path=checkpoint_path, manifest=manifest
    )
    assert result.source == 'manifest'
    assert result.mapping == {3: 0}


@pytest.mark.asyncio
async def test_promote_copies_class_remap_into_model_dir(
    fake_status: TrainJobStatus,
    scratch_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P2-7 step 6: a resolved remap must land in the served model dir too
    (class_remap.json alongside config.pbtxt/labels.txt), and promote.json
    must carry a class_remap provenance block."""
    from src.services.training.triton_promote import ClassRemapResult

    monkeypatch.setattr(TritonPromoter, '_trigger_load', AsyncMock(return_value=True))
    promoter = _promoter(scratch_models_dir)
    remap = ClassRemapResult(
        mapping={12: 0, 81: 1},
        names=['pickup', 'license_plate'],
        single_cls=False,
        include_classes=[12, 81],
        source='manifest',
    )
    result = await promoter.promote(
        status=fake_status,
        triton_name='op_smoke_remap',
        class_id_to_name={0: 'pickup', 1: 'license_plate'},
        class_remap=remap,
    )
    assert result.class_remap_source == 'manifest'
    model_dir = scratch_models_dir / 'op_smoke_remap'
    remap_dest = json.loads((model_dir / 'class_remap.json').read_text())
    assert remap_dest['original_to_new'] == {'12': 0, '81': 1}
    assert remap_dest['source'] == 'manifest'
    promote_json = json.loads((model_dir / 'promote.json').read_text())
    assert promote_json['class_remap']['source'] == 'manifest'
    assert promote_json['class_remap']['n_classes'] == 2


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


def _make_promoted_model_dir(scratch_models_dir: Path, name: str = 'op_smoke_unload_v1') -> Path:
    model_dir = scratch_models_dir / name
    v1 = model_dir / '1'
    v1.mkdir(parents=True)
    (v1 / 'model.onnx').write_bytes(b'THROWAWAY-SMOKE-WEIGHTS')
    (model_dir / 'config.pbtxt').write_text('# throwaway smoke config')
    (model_dir / 'promote.json').write_text('{"job_id": "job-abc"}')
    return model_dir


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
    from src.services.training.triton_promote import reload_promoted_models

    _make_promoted_model_dir(scratch_models_dir, 'op_ready_v1')
    fake_client = _FakeIndexAndLoadClient(
        index_response=[{'name': 'op_ready_v1', 'state': 'READY'}], load_ok=set()
    )
    monkeypatch.setattr('src.services.training.triton_promote.httpx.AsyncClient', fake_client)

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result == {'status': 'ok', 'reloaded': [], 'failed': []}
    assert fake_client.load_calls == []


@pytest.mark.asyncio
async def test_reload_promoted_models_reloads_unavailable_promoted_models(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.training.triton_promote import reload_promoted_models

    _make_promoted_model_dir(scratch_models_dir, 'op_stranded_v1')
    fake_client = _FakeIndexAndLoadClient(
        index_response=[{'name': 'op_stranded_v1', 'state': 'UNAVAILABLE'}],
        load_ok={'op_stranded_v1'},
    )
    monkeypatch.setattr('src.services.training.triton_promote.httpx.AsyncClient', fake_client)

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result == {'status': 'ok', 'reloaded': ['op_stranded_v1'], 'failed': []}
    assert fake_client.load_calls == ['op_stranded_v1']


@pytest.mark.asyncio
async def test_reload_promoted_models_ignores_non_promoted_model_dirs(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model dir with no promote.json (e.g. a core pipeline model) is
    never a reload target, ready or not."""
    from src.services.training.triton_promote import reload_promoted_models

    core_dir = scratch_models_dir / 'core_model'
    (core_dir / '1').mkdir(parents=True)
    (core_dir / 'config.pbtxt').write_text('# core, no promote.json')

    fake_client = _FakeIndexAndLoadClient(
        index_response=[{'name': 'core_model', 'state': 'UNAVAILABLE'}], load_ok=set()
    )
    monkeypatch.setattr('src.services.training.triton_promote.httpx.AsyncClient', fake_client)

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result == {'status': 'ok', 'reloaded': [], 'failed': []}
    assert fake_client.load_calls == []


@pytest.mark.asyncio
async def test_reload_promoted_models_is_best_effort_on_unreachable_triton(
    scratch_models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.services.training.triton_promote import reload_promoted_models

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

    monkeypatch.setattr('src.services.training.triton_promote.httpx.AsyncClient', _RaisingClient())

    result = await reload_promoted_models(_promoter(scratch_models_dir))

    assert result['status'] == 'error'
    assert result['reloaded'] == []
