"""Tests for the generic terminal-status region requeue tool
(:mod:`src.services.curation.region_requeue` and
``scripts/curation/requeue_regions.py``).

Backed by :class:`curation.query_fakes.QueryFakeOpenSearch`, which evaluates
the selection query, aggregations, ``search_after`` paging and the OCC bulk
write for real. The requeued items are also checked against the detection
worker's own pending query, so "requeued" means "the worker will pick it up".
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch, matches
from src.config import CurationConfig, RegionStatus
from src.config.region_fields import RegionFields, get_region_fields
from src.services.curation.region_requeue import (
    NONE_BUCKET,
    RequeueSelection,
    apply_requeue,
    requeue_breakdown,
)


if TYPE_CHECKING:
    from types import ModuleType


ITEMS = 'test_items'
CFG = CurationConfig(items_index=ITEMS)
F = get_region_fields()
FAILED = RegionStatus.DETECTION_FAILED
REJECTED = RegionStatus.VERIFY_REJECTED


def _item(doc_id: str, status: str, **region: Any) -> dict[str, Any]:
    return {
        'crop_id': doc_id,
        'image_path': f'{doc_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.9, 0.9],
        'class_name': 'widget',
        F.status: status,
        **{getattr(F, k): v for k, v in region.items()},
    }


def _corpus() -> dict[str, dict[str, Any]]:
    box = [0.2, 0.2, 0.4, 0.3]
    docs = [
        _item('f1', FAILED, detector='det_a', rejection_reason='aspect', detector_chain=['x'],
              bbox_norm=box, score=0.4, embedding=[0.1, 0.2], cluster_id=3),
        _item('f2', FAILED, detector='det_a', rejection_reason='tiny', detector_chain=['x']),
        _item('f3', FAILED, detector='det_b', rejection_reason='aspect', detector_chain=['x']),
        _item('f4', FAILED),  # pre-provenance: no detector, reason or chain
        _item('f5', FAILED, detector='det_a', rejection_reason='aspect', validated=True),
        _item('r1', REJECTED, detector='det_a', detector_chain=['x'], bbox_norm=box),
        _item('r2', REJECTED, detector='det_b'),
        _item('r3', REJECTED, detector='det_b', bbox_norm=box,
              status_legacy=RegionStatus.NO_REGION_VISIBLE.value),
        _item('d1', RegionStatus.DETECTED, detector='det_a', bbox_norm=box),
        _item('v1', RegionStatus.NO_REGION_VISIBLE, validated=True),
    ]  # fmt: skip
    return {d['crop_id']: d for d in docs}


def _fake() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch({ITEMS: _corpus()})


def _statuses(fake: QueryFakeOpenSearch) -> dict[str, str]:
    return {i: d[F.status] for i, d in fake.docs(ITEMS).items()}


def _changed(fake: QueryFakeOpenSearch) -> set[str]:
    original = _corpus()
    return {i for i, d in fake.docs(ITEMS).items() if d != original[i]}


@pytest.mark.asyncio
async def test_breakdown_groups_by_detector_then_reason():
    report = await requeue_breakdown(_fake(), RequeueSelection(FAILED), config=CFG)

    assert report['total'] == 4  # f5 is human-validated -> never selected
    by_det = {d['detector']: d for d in report['by_detector']}
    assert by_det['det_a']['count'] == 2
    assert {r['reason']: r['count'] for r in by_det['det_a']['reasons']} == {
        'aspect': 1,
        'tiny': 1,
    }
    assert by_det['det_b']['count'] == 1
    assert by_det[NONE_BUCKET]['reasons'] == [{'reason': NONE_BUCKET, 'count': 1}]


@pytest.mark.asyncio
async def test_apply_requeues_only_the_filtered_cohort():
    fake = _fake()
    sel = RequeueSelection(FAILED, detectors=('det_a',), reasons=('aspect',))
    totals = await apply_requeue(fake, sel, config=CFG)

    assert totals == {'updated': 1, 'skipped': 0, 'errors': 0}
    assert _changed(fake) == {'f1'}
    f1 = fake.docs(ITEMS)['f1']
    assert f1[F.status] == RegionStatus.PENDING_DETECTION
    assert f1[F.status_legacy] == FAILED
    assert f1[F.rejection_reason] is None
    assert f1[F.bbox_norm] == [0.2, 0.2, 0.4, 0.3], 'box kept without --clear-detection'
    assert fake.indices.refreshed == [ITEMS]


@pytest.mark.asyncio
async def test_none_bucket_selects_rows_without_a_detector():
    fake = _fake()
    await apply_requeue(fake, RequeueSelection(FAILED, detectors=(NONE_BUCKET,)), config=CFG)
    assert _changed(fake) == {'f4'}


@pytest.mark.asyncio
async def test_clear_detection_resets_every_detection_field():
    fake = _fake()
    sel = RequeueSelection(FAILED, detectors=('det_a',), reasons=('aspect',))
    await apply_requeue(fake, sel, clear_detection=True, config=CFG)

    f1 = fake.docs(ITEMS)['f1']
    for field in (F.bbox_norm, F.score, F.detector, F.detector_chain, F.embedding, F.cluster_id):
        assert f1[field] is None, field
    assert f1['bbox_norm'] == [0.1, 0.1, 0.9, 0.9], 'item bbox is not a region field'


@pytest.mark.asyncio
async def test_pending_verification_target_needs_an_existing_box():
    fake = _fake()
    sel = RequeueSelection(REJECTED, target=RegionStatus.PENDING_VERIFICATION)
    await apply_requeue(fake, sel, config=CFG)

    assert _changed(fake) == {'r1', 'r3'}
    assert fake.docs(ITEMS)['r1'][F.status] == RegionStatus.PENDING_VERIFICATION
    # an earlier requeue's audit trail is preserved, not overwritten
    assert fake.docs(ITEMS)['r3'][F.status_legacy] == RegionStatus.NO_REGION_VISIBLE


@pytest.mark.asyncio
async def test_missing_provenance_filter():
    fake = _fake()
    await apply_requeue(fake, RequeueSelection(REJECTED, missing_provenance=True), config=CFG)
    assert _changed(fake) == {'r2', 'r3'}


@pytest.mark.asyncio
async def test_requeue_is_idempotent_and_never_touches_human_verdicts():
    fake = _fake()
    first = await apply_requeue(fake, RequeueSelection(FAILED), config=CFG)
    second = await apply_requeue(fake, RequeueSelection(FAILED), config=CFG)
    assert first['updated'] == 4
    assert second['updated'] == 0
    assert (await requeue_breakdown(fake, RequeueSelection(FAILED), config=CFG))['total'] == 0
    assert _statuses(fake)['f5'] == FAILED
    assert 'f5' not in _changed(fake)


@pytest.mark.asyncio
async def test_max_docs_caps_across_pages():
    fake = _fake()
    totals = await apply_requeue(
        fake, RequeueSelection(FAILED), config=CFG, page_size=1, max_docs=3
    )
    assert totals['updated'] == 3
    assert (await requeue_breakdown(fake, RequeueSelection(FAILED), config=CFG))['total'] == 1


@pytest.mark.asyncio
async def test_requeued_items_enter_the_worker_queue():
    from scripts.curation.worker.cascade import _build_pending_query

    fake = _fake()
    worker_query = _build_pending_query()
    assert not matches(fake.docs(ITEMS)['f2'], worker_query)
    await apply_requeue(fake, RequeueSelection(FAILED), config=CFG)
    queued = {i for i, d in fake.docs(ITEMS).items() if matches(d, worker_query)}
    assert queued == {'f1', 'f2', 'f3', 'f4'}


@pytest.mark.asyncio
async def test_custom_region_field_names_are_honoured():
    custom = RegionFields(
        status='roi_state',
        rejection_reason='roi_why',
        detector='roi_by',
        status_legacy='roi_state_prev',
        validated='roi_ok',
    )
    doc = {'crop_id': 'c1', 'roi_state': FAILED.value, 'roi_by': 'det_a', 'roi_why': 'x'}
    fake = QueryFakeOpenSearch({ITEMS: {'c1': copy.deepcopy(doc)}})

    report = await requeue_breakdown(fake, RequeueSelection(FAILED), config=CFG, fields=custom)
    assert report['by_detector'][0]['detector'] == 'det_a'
    await apply_requeue(fake, RequeueSelection(FAILED), config=CFG, fields=custom)
    c1 = fake.docs(ITEMS)['c1']
    assert (c1['roi_state'], c1['roi_state_prev'], c1['roi_why']) == (
        RegionStatus.PENDING_DETECTION,
        FAILED,
        None,
    )


def test_selection_rejects_non_failure_statuses():
    with pytest.raises(ValueError, match='not requeueable'):
        RequeueSelection(RegionStatus.DETECTED)
    with pytest.raises(ValueError, match='not requeueable'):
        RequeueSelection(RegionStatus.FALSE_POSITIVE)
    with pytest.raises(ValueError, match='not a pending status'):
        RequeueSelection(FAILED, target=RegionStatus.DETECTED)


@pytest.mark.asyncio
async def test_clear_detection_refuses_pending_verification_target():
    sel = RequeueSelection(REJECTED, target=RegionStatus.PENDING_VERIFICATION)
    with pytest.raises(ValueError, match='pending_verification'):
        await apply_requeue(_fake(), sel, clear_detection=True, config=CFG)


# --------------------------------------------------------------------------- CLI


def _load_script() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / 'scripts' / 'curation' / 'requeue_regions.py'
    spec = importlib.util.spec_from_file_location('curation_requeue_regions_cli_test', path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Closable:
    def __init__(self, inner: QueryFakeOpenSearch) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def close(self) -> None:
        return None


def _run_cli(monkeypatch, argv: list[str], fake: QueryFakeOpenSearch) -> int:
    mod = _load_script()
    monkeypatch.setattr(mod, 'AsyncOpenSearch', lambda **_kw: _Closable(fake))
    monkeypatch.setattr(mod, 'get_curation_config', lambda: CFG)
    monkeypatch.setattr(sys, 'argv', ['requeue_regions.py', *argv])
    return mod.main()


def test_cli_defaults_to_dry_run(monkeypatch, capsys):
    fake = _fake()
    assert _run_cli(monkeypatch, ['--status', 'detection_failed'], fake) == 0
    out = capsys.readouterr().out
    assert '4 regions selected' in out
    assert 'detector=det_a' in out
    assert 'reason=aspect' in out
    assert _changed(fake) == set()


def test_cli_apply_with_filters(monkeypatch):
    fake = _fake()
    argv = ['--status', 'detection_failed', '--detector', 'det_b', '--apply']
    assert _run_cli(monkeypatch, argv, fake) == 0
    assert _changed(fake) == {'f3'}


def test_cli_rejects_clear_detection_with_pending_verification(monkeypatch):
    argv = ['--status', 'verify_rejected', '--to', 'pending_verification', '--clear-detection']
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, argv, _fake())


def test_cli_status_choices_exclude_human_and_success_states(monkeypatch):
    for status in ('detected', 'false_positive'):
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, ['--status', status], _fake())
