"""Tests for the generic terminal-status region requeue tool
(:mod:`src.services.curation.region_requeue` and
``scripts/curation/requeue_regions.py``).

W8c: the corpus is ``region_boxes``-shaped (the worker's only write
target since W8b) rather than the deleted single-scalar fields
(``bbox_norm`` / item-level ``detector`` / ``rejection_reason`` /
``embedding`` / ``cluster_id``) the pre-W8c version of this test used --
those are never populated by a fresh W8 write, so a query against them
would silently select nothing for the pipeline this branch ships.

Backed by :class:`curation.query_fakes.QueryFakeOpenSearch`, which evaluates
the selection query (incl. ``nested`` queries/aggs over ``region_boxes``),
``search_after`` paging and the OCC bulk write for real. The requeued items
are also checked against the detection worker's own pending query, so
"requeued" means "the worker will pick it up".
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
from src.config.region_rejection import REJECT_REASON_HUMAN
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


def _box(
    box_id: str = 'b1',
    *,
    state: str = 'rejected',
    detector: str | None = 'det_a',
    reason: str | None = 'aspect',
    source: str = 'detector',
    bbox: tuple[float, float, float, float] = (0.2, 0.2, 0.4, 0.3),
    score: float = 0.4,
) -> dict[str, Any]:
    return {
        'box_id': box_id,
        'bbox_norm': list(bbox),
        'state': state,
        'score': score,
        'detector': detector,
        'detector_version': '1' if detector else None,
        'source': source,
        'rejection_reason': reason,
    }


def _item(
    doc_id: str,
    status: str,
    *,
    boxes: tuple[dict[str, Any], ...] = (),
    detector_chain: list[str] | None = None,
    validated: bool | None = None,
    status_legacy: str | None = None,
) -> dict[str, Any]:
    boxes = tuple(boxes)
    doc: dict[str, Any] = {
        'crop_id': doc_id,
        'image_path': f'{doc_id}.jpg',
        'bbox_norm': [0.1, 0.1, 0.9, 0.9],
        'class_name': 'widget',
        F.status: status,
        F.boxes: list(boxes),
        F.count: sum(1 for b in boxes if b['state'] == 'accepted'),
        F.rejected_count: sum(1 for b in boxes if b['state'] == 'rejected'),
    }
    if detector_chain is not None:
        doc[F.detector_chain] = detector_chain
    if validated is not None:
        doc[F.validated] = validated
    if status_legacy is not None:
        doc[F.status_legacy] = status_legacy
    return doc


def _corpus() -> dict[str, dict[str, Any]]:
    box = (0.2, 0.2, 0.4, 0.3)
    docs = [
        _item(
            'f1', FAILED,
            boxes=(_box('b1', detector='det_a', reason='aspect', bbox=box, score=0.4),),
            detector_chain=['x'],
        ),
        _item(
            'f2', FAILED,
            boxes=(_box('b1', detector='det_a', reason='tiny'),),
            detector_chain=['x'],
        ),
        _item(
            'f3', FAILED,
            boxes=(_box('b1', detector='det_b', reason='aspect'),),
            detector_chain=['x'],
        ),
        _item('f4', FAILED),  # pre-provenance: no box at all
        _item(
            'f5', FAILED,
            boxes=(_box('b1', detector='det_a', reason='aspect'),),
            validated=True,
        ),
        # f6: a box exists but its detector was never recorded on it (older
        # partially-provenanced data) -- distinct from f4 (no box at all),
        # which a nested aggregation can never bucket (see the breakdown
        # test below).
        _item(
            'f6', FAILED,
            boxes=(_box('b1', detector=None, reason=None),),
        ),
        _item(
            'r1', REJECTED,
            boxes=(_box('b1', detector='det_a', reason=None, bbox=box),),
            detector_chain=['x'],
        ),
        # r2: rejected, with no chain and no box recorded -- a rejection
        # that happened before a box was ever chosen.
        _item('r2', REJECTED),
        _item(
            'r3', REJECTED,
            boxes=(_box('b1', detector='det_b', reason=None, bbox=box),),
            status_legacy=RegionStatus.NO_REGION_VISIBLE.value,
        ),
        _item(
            'd1', RegionStatus.DETECTED,
            boxes=(_box('b1', state='accepted', detector='det_a', reason=None, bbox=box),),
        ),
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


def _box_ids(doc: dict[str, Any]) -> list[str]:
    return [b['box_id'] for b in doc.get(F.boxes) or []]


@pytest.mark.asyncio
async def test_breakdown_groups_by_detector_then_reason():
    report = await requeue_breakdown(_fake(), RequeueSelection(FAILED), config=CFG)

    # f5 is human-validated -> never selected. The total (5: f1-f4, f6)
    # counts ITEMS via the outer query; the nested by-detector breakdown
    # below counts BOXES and can never bucket f4 (no box at all -- a
    # nested agg has no element to bucket it under, not even "(none)"),
    # so its bucket counts sum to 4, one less than `total`.
    assert report['total'] == 5
    by_det = {d['detector']: d for d in report['by_detector']}
    assert by_det['det_a']['count'] == 2
    assert {r['reason']: r['count'] for r in by_det['det_a']['reasons']} == {
        'aspect': 1,
        'tiny': 1,
    }
    assert by_det['det_b']['count'] == 1
    assert by_det[NONE_BUCKET]['count'] == 1  # f6: a box exists, no detector on it
    assert by_det[NONE_BUCKET]['reasons'] == [{'reason': NONE_BUCKET, 'count': 1}]
    # W8c nit fix: f4 (no box at all) is invisible to the nested breakdown
    # above, but IS counted here -- `total - no_box` (5 - 1 = 4) is
    # exactly the item count with at least one box, restoring the
    # pre-nested-query "(none)" bucket's information.
    assert report['no_box'] == 1


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
    assert _box_ids(f1) == ['b1'], 'box kept without --clear-detection'
    assert f1[F.boxes][0]['bbox_norm'] == [0.2, 0.2, 0.4, 0.3]
    assert fake.indices.refreshed == [ITEMS]


@pytest.mark.asyncio
async def test_none_bucket_selects_rows_without_a_detector():
    fake = _fake()
    await apply_requeue(fake, RequeueSelection(FAILED, detectors=(NONE_BUCKET,)), config=CFG)
    # f4: no box at all. f6: a box exists but carries no detector value.
    # Both are "no value recorded" for this filter.
    assert _changed(fake) == {'f4', 'f6'}


@pytest.mark.asyncio
async def test_clear_detection_drops_machine_boxes_but_keeps_a_human_ones():
    fake = _fake()
    fake.docs(ITEMS)['f1'][F.boxes].append(
        _box('b2', state='rejected', detector='human', source='human', reason='human_reject')
    )
    fake.docs(ITEMS)['f1'][F.rejected_count] = 2
    sel = RequeueSelection(FAILED, detectors=('det_a',), reasons=('aspect',))
    await apply_requeue(fake, sel, clear_detection=True, config=CFG)

    f1 = fake.docs(ITEMS)['f1']
    assert _box_ids(f1) == ['b2'], 'the machine box is dropped, the human one survives'
    assert f1[F.boxes][0]['source'] == 'human'
    assert f1['bbox_norm'] == [0.1, 0.1, 0.9, 0.9], 'item bbox is not a region field'


@pytest.mark.asyncio
async def test_clear_detection_prunes_the_embedding_of_the_dropped_box():
    from src.services.curation.region_box_embeddings import entry_for
    from src.services.curation.region_boxes import read_boxes

    fake = _fake()
    doc = fake.docs(ITEMS)['f1']
    doc[F.box_embeddings] = [entry_for(b, [1.0]) for b in read_boxes(doc, F)]
    assert doc[F.box_embeddings]

    await apply_requeue(
        fake,
        RequeueSelection(FAILED, detectors=('det_a',), reasons=('aspect',)),
        clear_detection=True,
        config=CFG,
    )

    f1 = fake.docs(ITEMS)['f1']
    assert _box_ids(f1) == []
    assert f1[F.box_embeddings] == []


@pytest.mark.asyncio
async def test_clear_detection_protects_a_human_reject_action_on_a_machine_box():
    """W8c M3 fix: `source == 'human'` alone only protects a box a human
    CREATED. A human's per-box REJECT action on a MACHINE-created box
    (`PATCH .../regions/{box_id}`, `POST /regions/batch_box_state`) never
    changes `source`/`detector` -- the only trace is the rejection reason
    those routes now also stamp. This must survive `clear_detection` too,
    not just a human-created box."""
    fake = _fake()
    fake.docs(ITEMS)['f1'][F.boxes].append(
        # source/detector still the MACHINE's -- only the reason marks
        # this as a human verdict, exactly what patch_crop_region_box /
        # batch_set_region_box_state now write.
        _box(
            'b2',
            state='rejected',
            detector='det_a',
            source='detector',
            reason=REJECT_REASON_HUMAN,
        )
    )
    fake.docs(ITEMS)['f1'][F.rejected_count] = 2
    sel = RequeueSelection(FAILED, detectors=('det_a',), reasons=('aspect',))
    await apply_requeue(fake, sel, clear_detection=True, config=CFG)

    f1 = fake.docs(ITEMS)['f1']
    assert _box_ids(f1) == ['b2'], 'the human-rejected machine box must survive clear_detection'
    assert f1[F.boxes][0]['rejection_reason'] == REJECT_REASON_HUMAN


@pytest.mark.asyncio
async def test_clear_detection_is_a_noop_when_there_is_no_box_to_drop():
    fake = _fake()
    await apply_requeue(fake, RequeueSelection(FAILED), clear_detection=True, config=CFG)
    f4 = fake.docs(ITEMS)['f4']
    assert _box_ids(f4) == []
    assert f4[F.status] == RegionStatus.PENDING_DETECTION


@pytest.mark.asyncio
async def test_pending_verification_target_needs_an_existing_box():
    fake = _fake()
    sel = RequeueSelection(REJECTED, target=RegionStatus.PENDING_VERIFICATION)
    await apply_requeue(fake, sel, config=CFG)

    assert _changed(fake) == {'r1', 'r3'}
    r1 = fake.docs(ITEMS)['r1']
    assert r1[F.status] == RegionStatus.PENDING_VERIFICATION
    # an earlier requeue's audit trail is preserved, not overwritten
    assert fake.docs(ITEMS)['r3'][F.status_legacy] == RegionStatus.NO_REGION_VISIBLE
    # W8c M2 fix: the item's own box(es) were always `rejected` (a
    # REQUEUEABLE_STATUSES item can never already carry a `proposed` box
    # -- `derive_status` would report `pending_verification` instead).
    # This requeue must re-propose them, or the worker's Path 1 has
    # nothing to re-verify and silently runs a fresh detection instead.
    box = r1[F.boxes][0]
    assert box['state'] == 'proposed'
    assert box['rejection_reason'] is None
    assert box['bbox_norm'] == [0.2, 0.2, 0.4, 0.3]  # geometry untouched


@pytest.mark.asyncio
async def test_pending_verification_skips_item_whose_only_box_is_human_owned():
    """W8c M2 fix: a human's rejected verdict is a verdict, not a failure
    to re-propose -- an item with nothing else to re-verify must not move
    to `pending_verification` (that would silently fall through to a
    fresh detection pass instead, per the review's exact finding)."""
    fake = _fake()
    fake.docs(ITEMS)['r1'][F.boxes] = [
        _box('b1', detector='human', source='human', reason=REJECT_REASON_HUMAN)
    ]
    # Snapshot AFTER the human-owned-box mutation above -- `_changed()`'s
    # own baseline (`_corpus()`) predates it, so comparing against that
    # would always show r1 as "changed" regardless of what `apply_requeue`
    # does.
    before_r1 = copy.deepcopy(fake.docs(ITEMS)['r1'])
    sel = RequeueSelection(REJECTED, target=RegionStatus.PENDING_VERIFICATION)
    totals = await apply_requeue(fake, sel, config=CFG)

    # r1 is excluded (nothing to re-verify); r3 (non-human) still moves.
    assert totals['updated'] == 1
    r1 = fake.docs(ITEMS)['r1']
    assert r1 == before_r1, 'an item with nothing to re-verify must be left untouched'
    assert r1[F.status] == REJECTED  # untouched -- verify_rejected status stays
    assert r1[F.boxes][0]['state'] == 'rejected'


@pytest.mark.asyncio
async def test_pending_verification_preserves_a_human_rejected_sibling():
    """A mixed item (one non-human rejected box, one human-rejected box):
    only the non-human box is re-proposed; the human's verdict survives
    untouched."""
    fake = _fake()
    fake.docs(ITEMS)['r1'][F.boxes].append(
        _box('b2', detector='human', source='human', reason=REJECT_REASON_HUMAN)
    )
    sel = RequeueSelection(REJECTED, target=RegionStatus.PENDING_VERIFICATION)
    await apply_requeue(fake, sel, config=CFG)

    r1 = fake.docs(ITEMS)['r1']
    boxes_by_id = {b['box_id']: b for b in r1[F.boxes]}
    assert boxes_by_id['b1']['state'] == 'proposed'
    assert boxes_by_id['b2']['state'] == 'rejected'
    assert boxes_by_id['b2']['rejection_reason'] == REJECT_REASON_HUMAN
    assert r1[F.status] == RegionStatus.PENDING_VERIFICATION


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
    assert first['updated'] == 5  # f1-f4, f6
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
    assert (await requeue_breakdown(fake, RequeueSelection(FAILED), config=CFG))['total'] == 2


@pytest.mark.asyncio
async def test_requeued_items_enter_the_worker_queue():
    from scripts.curation.worker.cascade import _build_pending_query

    fake = _fake()
    worker_query = _build_pending_query()
    assert not matches(fake.docs(ITEMS)['f2'], worker_query)
    await apply_requeue(fake, RequeueSelection(FAILED), config=CFG)
    queued = {i for i, d in fake.docs(ITEMS).items() if matches(d, worker_query)}
    assert queued == {'f1', 'f2', 'f3', 'f4', 'f6'}


@pytest.mark.asyncio
async def test_custom_region_field_names_are_honoured():
    custom = RegionFields(
        status='roi_state',
        rejection_reason='roi_why',
        boxes='roi_boxes',
        count='roi_count',
        rejected_count='roi_rejected_count',
        status_legacy='roi_state_prev',
        validated='roi_ok',
    )
    doc = {
        'crop_id': 'c1',
        'roi_state': FAILED.value,
        'roi_boxes': [_box('b1', detector='det_a', reason='x')],
        'roi_rejected_count': 1,
    }
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


def _bound_items() -> str:
    from src.config import get_curation_config

    return get_curation_config().items_index


def _cli_fake(corpus: dict[str, dict[str, Any]] | None = None) -> QueryFakeOpenSearch:
    """The CLI resolves its index from the bound project, not from ``CFG``."""
    return QueryFakeOpenSearch({_bound_items(): corpus if corpus is not None else _corpus()})


def _cli_changed(fake: QueryFakeOpenSearch, original: dict[str, dict[str, Any]]) -> set[str]:
    return {i for i, d in fake.docs(_bound_items()).items() if d != original[i]}


def _run_cli(monkeypatch, argv: list[str], fake: QueryFakeOpenSearch) -> int:
    mod = _load_script()
    monkeypatch.setattr(mod, 'make_script_opensearch', lambda *_a, **_kw: _Closable(fake))
    monkeypatch.setattr(sys, 'argv', ['requeue_regions.py', *argv])
    return mod.main()


def test_cli_defaults_to_dry_run(monkeypatch, capsys):
    fake = _cli_fake()
    assert _run_cli(monkeypatch, ['--status', 'detection_failed'], fake) == 0
    out = capsys.readouterr().out
    # W8c nit fix: "items selected" (not "regions" -- the count is items,
    # never boxes, and the per-detector/reason lines below it are boxes;
    # conflating the two units in one label is exactly what confused the
    # review). The selection includes the one human-validated item f5,
    # which the lock rule skips and the report names.
    assert '6 items selected' in out
    assert 'detector=det_a' in out
    assert 'reason=aspect' in out
    assert '(no box at all)' in out  # f4
    assert '(locked, skipped)' in out
    assert _cli_changed(fake, _corpus()) == set()


def test_cli_apply_with_filters(monkeypatch):
    fake = _cli_fake()
    argv = ['--status', 'detection_failed', '--detector', 'det_b', '--apply']
    assert _run_cli(monkeypatch, argv, fake) == 0
    assert _cli_changed(fake, _corpus()) == {'f3'}


def test_cli_pending_verification_reverifies_and_keeps_the_box(monkeypatch):
    """Was ``--clear-detection`` refused with ``--to pending_verification``:
    the flag is gone (a full re-detect always clears unlocked machine
    boxes), and a re-verify never clears one -- r1's rejected box is
    re-proposed in place."""
    fake = _cli_fake()
    argv = ['--status', 'verify_rejected', '--to', 'pending_verification', '--apply']
    assert _run_cli(monkeypatch, argv, fake) == 0
    r1 = fake.docs(_bound_items())['r1']
    assert r1[F.status] == RegionStatus.PENDING_VERIFICATION
    assert [(b['box_id'], b['state']) for b in r1[F.boxes]] == [('b1', 'proposed')]
    with pytest.raises(SystemExit):
        _run_cli(
            monkeypatch,
            ['--status', 'verify_rejected', '--to', 'pending_verification', '--clear-detection'],
            _cli_fake(),
        )


def test_cli_status_choices_exclude_human_and_success_states(monkeypatch):
    for status in ('detected', 'false_positive'):
        with pytest.raises(SystemExit):
            _run_cli(monkeypatch, ['--status', status], _fake())


# ------------------------------------------------------------- unseeded backfill


def _unseeded_corpus() -> dict[str, dict[str, Any]]:
    """Items ingested before ingest seeded a region status, plus two that
    must never be touched: one already queued, one human-validated."""
    docs: dict[str, dict[str, Any]] = {
        f'u{i}': {'crop_id': f'u{i}', 'image_path': f'u{i}.jpg', 'bbox_norm': [0.1, 0.1, 0.9, 0.9]}
        for i in range(3)
    }
    docs['q1'] = _item('q1', RegionStatus.PENDING_DETECTION)
    docs['h1'] = {
        'crop_id': 'h1',
        'image_path': 'h1.jpg',
        'bbox_norm': [0.1, 0.1, 0.9, 0.9],
        F.validated: True,
    }
    return docs


@pytest.mark.asyncio
async def test_unseeded_selection_breakdown_counts_items_without_status():
    fake = QueryFakeOpenSearch({ITEMS: _unseeded_corpus()})
    report = await requeue_breakdown(fake, RequeueSelection(None), config=CFG)
    assert report['total'] == 3
    assert report['status'] == NONE_BUCKET
    assert report['no_box'] == 3  # none of u0-u2 carry a region_boxes element


@pytest.mark.asyncio
async def test_unseeded_backfill_seeds_pending_detection_into_worker_queue():
    from scripts.curation.worker.cascade import _build_pending_query

    fake = QueryFakeOpenSearch({ITEMS: _unseeded_corpus()})
    before = copy.deepcopy(fake.docs(ITEMS))
    totals = await apply_requeue(fake, RequeueSelection(None), config=CFG)

    assert totals == {'updated': 3, 'skipped': 0, 'errors': 0}
    after = fake.docs(ITEMS)
    assert {i for i, d in after.items() if d != before[i]} == {'u0', 'u1', 'u2'}
    worker_query = _build_pending_query()
    for i in ('u0', 'u1', 'u2'):
        assert after[i][F.status] == RegionStatus.PENDING_DETECTION
        # No prior status existed, so there is nothing to stash or clear.
        assert F.status_legacy not in after[i]
        assert F.rejection_reason not in after[i]
        assert matches(after[i], worker_query)

    again = await apply_requeue(fake, RequeueSelection(None), config=CFG)
    assert again['updated'] == 0


def test_cli_missing_status_dry_run_then_apply(monkeypatch, capsys):
    fake = _cli_fake(_unseeded_corpus())
    assert _run_cli(monkeypatch, ['--missing-status'], fake) == 0
    out = capsys.readouterr().out
    # u0-u2 plus the human-validated h1, which the lock rule skips.
    assert '4 items selected' in out
    assert '(locked, skipped)' in out
    assert F.status not in fake.docs(_bound_items())['u0']

    assert _run_cli(monkeypatch, ['--missing-status', '--apply'], fake) == 0
    assert fake.docs(_bound_items())['u0'][F.status] == RegionStatus.PENDING_DETECTION
    assert F.status not in fake.docs(_bound_items())['h1']


def test_cli_requires_exactly_one_of_status_or_missing_status(monkeypatch):
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, [], _fake())
    with pytest.raises(SystemExit):
        _run_cli(monkeypatch, ['--status', 'detection_failed', '--missing-status'], _fake())
