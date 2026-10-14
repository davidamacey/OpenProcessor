"""A job started with the prompt pack omitted runs the pack it echoes, and the worker's IVF
self-trigger resolves to a pipeline function that accepts the worker's call shape."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from curation.activation.constants import PREFIX, pack_body
from curation.test_auto_label_selection import _run_kwargs, job_dir, labeler_spy, packs  # noqa: F401 - fixtures
from curation.test_pipeline import _FakeClassEntry, _FakeOpenSearch, _FakeRegistry
from src.services.curation.item_filter import ItemFilter


SCRIPTS_DIR = Path(__file__).resolve().parents[3] / 'scripts' / 'curation'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


pytestmark = pytest.mark.usefixtures('vlm_env')


def test_ivf_pipeline_path_accepts_worker_call_shape() -> None:
    """auto_label_worker._run_one always calls
    ``pipeline_fn(opensearch=..., progress=..., **args)``
    (auto_label_worker.py:257). The IVF self-trigger's ``pipeline`` path
    must resolve to a callable whose signature actually binds that shape
    with the IVF trigger's own args. Pointing ``_IVF_PIPELINE_PATH`` at the
    public route wrapper ``pipeline_auto_label`` (which has no ``progress``
    param) must make this fail.
    """
    import auto_label_worker as w

    fn = w._resolve_pipeline_fn(w._IVF_PIPELINE_PATH)
    args = w._ivf_retrain_trigger()['args']
    # Must not raise TypeError.
    inspect.signature(fn).bind(opensearch=None, progress=object(), **args)


def test_ivf_pipeline_path_is_the_internal_implementation() -> None:
    """Belt-and-suspenders: the resolved callable must be the real
    ``_run_auto_label`` internal, not the public route wrapper -- the
    wrapper forces ``prompt_pack_resolved=False``/``prompt_pack_
    revision=None``, silently re-resolving (and potentially re-pinning)
    every IVF retrain even if its signature happened to accept
    ``progress`` in the future."""
    import auto_label_worker as w

    from src.routers.curation.pipeline import _run_auto_label

    fn = w._resolve_pipeline_fn(w._IVF_PIPELINE_PATH)
    assert fn is _run_auto_label


def test_start_job_pipeline_path_also_binds_worker_call_shape() -> None:
    """Every pipeline path the worker can be asked to resolve -- not just
    the IVF constant -- must accept the worker's call shape."""
    import auto_label_worker as w

    from src.routers.curation.pipeline import _run_auto_label
    from src.services.curation.autolabel.job import _pipeline_import_path

    path = _pipeline_import_path(_run_auto_label)
    fn = w._resolve_pipeline_fn(path)
    inspect.signature(fn).bind(
        opensearch=None, progress=object(), train_clusters=False, run_vlm=False
    )


# =============================================================================
# with prompt_pack omitted and a legacy settings-doc default, the job
# must run the SAME pack it echoes in summary/args.
# =============================================================================


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_omitted_stale_settings_doc_pack_is_ignored_and_echo_matches_labeler(
    labeler_spy: list[Any],  # noqa: F811 - pytest fixture param shadows the import
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A leftover settings-doc key says 'food_v2'; nothing is activated in
    the config store (env/file default is 'pallet_v1'). Omitted
    ``prompt_pack``: the stale key is ignored (the activation record is the
    only source of truth) and the job echoes exactly what it ran."""
    from src.routers.curation import pipeline

    monkeypatch.setattr(
        'src.clients.curation_opensearch.settings_doc.get_curation_settings',
        AsyncMock(return_value={'defaults': {'prompt_pack': 'food_v2'}}),
    )
    summary = await pipeline._run_auto_label(
        opensearch=_FakeOpenSearch({3: ['pallet-1']}), **_run_kwargs()
    )
    ran = [inst._pack.name for inst in labeler_spy]
    assert summary['prompt_pack'] == 'pallet_v1'
    assert ran == [summary['prompt_pack']]


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_omitted_store_active_pack_still_uses_none_none(
    labeler_spy: list[Any],  # noqa: F811 - pytest fixture param shadows the import
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the omitted default DOES come from the config store's own
    activation (not a legacy settings doc / env / file default), the TOCTOU
    protection (resolve against ``(None, None)``, not the echoed name) must
    still apply."""
    from src.routers.curation.pipeline_params import omitted_pack_is_store_active
    from src.services.config_store.store import ConfigSnapshot, get_config_store

    store = get_config_store()
    original = store.current
    try:
        store.current = ConfigSnapshot(
            config_revision=original.config_revision,
            packs={},
            profiles={},
            active_pack=('food_v2', 1),
            active_profile=None,
            active_pack_body=None,
            loaded_at=original.loaded_at,
        )
        assert omitted_pack_is_store_active('food_v2') is True
        assert omitted_pack_is_store_active('pallet_v1') is False
    finally:
        store.current = original


def test_omitted_pack_is_store_active_false_when_never_activated() -> None:
    """No activation in the store at all (env/file default only) ->
    never treated as store-active, so the echo == run guarantee holds
    for every env/file-default deployment."""
    from src.routers.curation.pipeline_params import omitted_pack_is_store_active
    from src.services.config_store.store import get_config_store

    store = get_config_store()
    assert store.current.active_pack in (None, 'off')
    assert omitted_pack_is_store_active('pallet_v1') is False


@pytest.mark.usefixtures('packs', 'job_dir')
@pytest.mark.asyncio
async def test_start_omitted_stale_settings_doc_pack_is_ignored_not_flagged_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same guard as
    ``test_omitted_stale_settings_doc_pack_is_ignored_and_echo_matches_labeler``,
    but for ``/start``'s own gating (a separate call site from the
    direct-call branch): nothing is activated in the config store, the
    legacy settings doc names 'food_v2'. ``/start`` must NOT set
    ``prompt_pack_omitted=True`` here -- that flag exists only for the
    store-active TOCTOU case -- else the job would
    later resolve the labeler against ``(None, None)`` (the env/file
    default) while echoing 'food_v2'."""
    from src.routers.curation import pipeline_start
    from src.services.curation.autolabel import job

    captured: dict[str, Any] = {}

    def _fake_start(_fn: object, kwargs: dict[str, Any]) -> dict[str, Any]:
        captured.update(kwargs)
        return {'args': {k: v for k, v in kwargs.items() if k != 'opensearch'}}

    monkeypatch.setattr(job, 'start_job', _fake_start)
    monkeypatch.setattr(
        'src.clients.curation_opensearch.settings_doc.get_curation_settings',
        AsyncMock(return_value={'defaults': {'prompt_pack': 'food_v2'}}),
    )
    await pipeline_start.pipeline_auto_label_start(opensearch=object(), item_filter=ItemFilter())
    assert captured['prompt_pack'] == 'pallet_v1'
    assert captured['prompt_pack_omitted'] is False


@pytest.mark.asyncio
async def test_start_then_active_switch_real_job_serves_pinned_or_new(
    app_client,
    labeler_spy: list[Any],  # noqa: F811 - pytest fixture param shadows the import
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drives the production call site of ``labeler_resolution_args`` inside
    ``_run_auto_label`` through the real job path: build the trigger args
    exactly as ``/start`` does (pack omitted), then activate a different
    pack, then run ``_run_auto_label(**args, run_vlm=True)`` for real and
    assert the ACTUAL labeler instance the job obtained never serves
    'my_pack''s un-activated draft.

    Calling ``_get_vlm_labeler(prompt_pack, prompt_pack_revision)`` directly
    instead of going through ``labeler_resolution_args`` must turn this red.
    """
    from src.routers.curation import pipeline

    c = app_client
    assert c.post(PREFIX, json={'name': 'my_pack', 'body': pack_body('A')}).status_code == 201
    assert c.post(f'{PREFIX}/my_pack/activate', json={'expected_active': None}).status_code == 200
    # An un-activated draft on my_pack (r2): the labeler must never silently serve
    # it once 'other' becomes active. Without it the test would be a no-op, since the
    # pinned revision-1 body is byte-identical whichever code path resolves it.
    assert (
        c.put(
            f'{PREFIX}/my_pack', json={'expected_revision': 1, 'body': pack_body('B DRAFT')}
        ).status_code
        == 200
    )
    assert c.post(PREFIX, json={'name': 'other', 'body': pack_body('OTHER')}).status_code == 201

    captured: dict[str, Any] = {}

    def _fake_start(_fn: object, kwargs: dict[str, Any]) -> dict[str, Any]:
        captured.update(kwargs)
        return {'job_id': 'x'}

    from src.services.curation.autolabel import job as auto_label_job

    monkeypatch.setattr(auto_label_job, 'start_job', _fake_start)
    r = c.post('/curation/projects/default/pipeline/auto_label/start')
    assert r.status_code == 200, r.text
    assert captured['prompt_pack_omitted'] is True

    switch = c.post(
        f'{PREFIX}/other/activate', json={'expected_active': {'name': 'my_pack', 'revision': 1}}
    )
    assert switch.status_code == 200, switch.text

    args = {k: v for k, v in captured.items() if k != 'opensearch'}
    args['train_clusters'] = False
    args['run_vlm'] = True
    monkeypatch.setattr(
        'src.routers.curation.get_class_registry',
        lambda: _FakeRegistry([_FakeClassEntry(3, 'wooden_pallet')]),
    )
    # `ensure_fresh` (prompt_pack_resolved=True) is a no-op here -- the
    # config-store snapshot was just written < 1s ago by the activate
    # calls above -- so this items-search-only client (distinct from
    # `c.fake_os`, which only models the configs index) never needs to
    # answer a config-store read.
    await pipeline._run_auto_label(opensearch=_FakeOpenSearch({3: ['pallet-1']}), **args)

    ran = [inst._pack.class_system for inst in labeler_spy]
    assert ran
    assert ran[0] in ('A', 'OTHER')
    assert ran[0] != 'B DRAFT'


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_direct_call_omitted_labeler_matches_active(
    labeler_spy: list[Any],  # noqa: F811 - pytest fixture param shadows the import
) -> None:
    """Same guard for the direct-call branch (never through ``/start``):
    a real ``_run_auto_label`` call with ``prompt_pack`` omitted must
    use the currently-active pack for the labeler."""
    from src.routers.curation import pipeline

    summary = await pipeline._run_auto_label(
        opensearch=_FakeOpenSearch({3: ['pallet-1']}), **_run_kwargs()
    )
    assert [inst._pack.name for inst in labeler_spy] == [summary['prompt_pack']]


@pytest.mark.usefixtures('packs')
@pytest.mark.asyncio
async def test_direct_call_branch_computes_omitted_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct unit guard for the ``if not prompt_pack_resolved:`` branch
    in ``_run_auto_label``: deleting its ``prompt_pack_omitted = not isinstance(...)``
    line (so ``prompt_pack_omitted`` silently stays at the function's
    ``False`` default for every direct call) must be observable --
    assert ``omitted_pack_is_store_active`` is actually invoked for an
    omitted-pack direct call, which only happens when that line ran."""
    from src.routers.curation import pipeline

    calls: list[Any] = []
    real = pipeline.omitted_pack_is_store_active

    def _spy(resolved_name: Any) -> bool:
        calls.append(resolved_name)
        return real(resolved_name)

    monkeypatch.setattr(pipeline, 'omitted_pack_is_store_active', _spy)
    await pipeline._run_auto_label(
        opensearch=_FakeOpenSearch({}), **_run_kwargs(train_clusters=False, run_vlm=False)
    )
    assert calls == ['pallet_v1']


def test_labeler_resolution_args_uses_none_when_omitted():
    """Direct unit test of `labeler_resolution_args`, which `_run_auto_label` otherwise
    only exercises indirectly (through `run_vlm=True` driving the full pipeline). It
    fails if the labeler is resolved from the echoed `(prompt_pack, prompt_pack_revision)`
    instead of `(None, None)` when the pack was omitted."""
    from src.routers.curation.pipeline_params import labeler_resolution_args

    assert labeler_resolution_args('my_pack', 1, prompt_pack_omitted=True) == (None, None)
    assert labeler_resolution_args('my_pack', 1, prompt_pack_omitted=False) == ('my_pack', 1)
    assert labeler_resolution_args(None, None, prompt_pack_omitted=False) == (None, None)
