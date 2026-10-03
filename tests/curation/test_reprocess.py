"""Unified reprocess (W10.13): targets, the region / vlm / embed scopes and
the lock rule. ``detect`` is in ``test_reprocess_detect.py``; the routes,
job, CLI and busy wiring in ``test_reprocess_entrypoints.py``."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from curation.reprocess_fixtures import (
    IMPORT,
    F,
    FakePE,
    FakeTriton,
    RegionStatus,
    box,
    docs,
    images_index,
    item,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.services.curation.region_box_embeddings import current_vectors
from src.services.curation.region_boxes import read_boxes
from src.services.curation.reprocess import apply_reprocess, plan_reprocess
from src.services.curation.reprocess_models import (
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)
from src.services.curation.reprocess_targets import ReprocessTargetsError, validate_targets


if TYPE_CHECKING:
    from pathlib import Path

FAILED = RegionStatus.DETECTION_FAILED.value


def _req(
    *,
    scopes: list[str],
    filt: ReprocessFilter | None = None,
    crops: list[str] | None = None,
    images: list[str] | None = None,
    dry_run: bool = True,
    mode: str = 'redetect',
) -> ReprocessRequest:
    return ReprocessRequest(
        targets=ReprocessTargets(filter=filt, crop_ids=crops, image_ids=images),
        scopes=scopes,  # type: ignore[arg-type]
        region_mode=mode,  # type: ignore[arg-type]
        dry_run=dry_run,
    )


def _result(resp: Any, scope: str) -> Any:
    return next(r for r in resp.scopes if r.scope == scope)


# ------------------------------------------------------------------ targets


@pytest.mark.parametrize(
    'targets',
    [
        ReprocessTargets(),
        ReprocessTargets(crop_ids=['a'], image_ids=['b']),
        ReprocessTargets(crop_ids=[]),
        ReprocessTargets(image_ids=['x'] * 5001),
        ReprocessTargets(filter=ReprocessFilter()),
        ReprocessTargets(crop_ids=['a'], filter=ReprocessFilter(source='s')),
    ],
)
def test_malformed_targets_are_rejected(targets: ReprocessTargets) -> None:
    with pytest.raises(ReprocessTargetsError):
        validate_targets(targets)


def test_request_models_forbid_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        ReprocessRequest.model_validate(
            {'targets': {'crop_ids': ['a']}, 'scopes': ['region'], 'clear_detection': True}
        )
    with pytest.raises(ValidationError):
        ReprocessFilter.model_validate({'status': 'detected'})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'filt',
    [
        ReprocessFilter(),  # empty
        ReprocessFilter(source='x'),  # no status / profile selector for the region scope
        ReprocessFilter(region_status=['detected']),  # not requeueable
        ReprocessFilter(region_status=['bogus']),
        ReprocessFilter(include_detected=True, region_status=[FAILED]),  # no profile selector
        ReprocessFilter(missing_status=True, region_status=[FAILED]),
    ],
)
async def test_region_scope_refuses_a_filter_that_selects_nothing_specific(
    filt: ReprocessFilter,
) -> None:
    with pytest.raises(ReprocessTargetsError):
        await plan_reprocess(make_fake([]), _req(scopes=['region'], filt=filt))


@pytest.mark.asyncio
async def test_detected_cannot_be_reverified() -> None:
    filt = ReprocessFilter(profile_not='p', include_detected=True)
    with pytest.raises(ReprocessTargetsError, match='detected'):
        await plan_reprocess(make_fake([]), _req(scopes=['region'], filt=filt, mode='reverify'))


# ------------------------------------------------------------------- region


def _region_corpus() -> list[dict[str, Any]]:
    imported = box('b2', state='accepted', source=IMPORT, detector=IMPORT, reason=None)
    return [
        # machine-only failure: redetect removes its box
        item('m1', FAILED, boxes=(box('b1'),)),
        # imported box + a machine box: only the machine box goes
        item('i1', FAILED, boxes=(imported, box('b1', bbox=(0.5, 0.5, 0.7, 0.7)))),
        # a validated (human or import) region set: locked, never selected
        item(
            'v1',
            FAILED,
            boxes=(imported,),
            validated=True,
            verifier=IMPORT,
        ),
        item('h1', RegionStatus.NO_REGION_VISIBLE.value, validated=True, verifier='human'),
        item('ok', RegionStatus.DETECTED.value, boxes=(box('b1', state='accepted'),)),
    ]


@pytest.mark.asyncio
async def test_region_dry_run_counts_selected_and_locked_skipped() -> None:
    fake = make_fake(_region_corpus())
    filt = ReprocessFilter(region_status=[FAILED])
    before = copy.deepcopy(docs(fake))
    resp = await apply_reprocess(fake, _req(scopes=['region'], filt=filt))
    region = _result(resp, 'region')
    assert resp.dry_run is True
    assert (region.selected, region.locked_skipped, region.queued) == (3, 1, 0)
    assert {(r.detector, r.reason): r.count for r in region.breakdown} == {
        ('det_a', 'aspect'): 2,
        (IMPORT, '(none)'): 1,
    }
    assert docs(fake) == before


@pytest.mark.asyncio
async def test_region_redetect_removes_machine_boxes_and_keeps_locked_ones_byte_identical() -> None:
    fake = make_fake(_region_corpus())
    before = copy.deepcopy(docs(fake))
    resp = await apply_reprocess(
        fake, _req(scopes=['region'], filt=ReprocessFilter(region_status=[FAILED]), dry_run=False)
    )
    assert _result(resp, 'region').queued == 2
    after = docs(fake)
    assert after['m1'][F.status] == RegionStatus.PENDING_DETECTION
    assert after['m1'][F.boxes] == []
    assert after['m1'][F.status_legacy] == FAILED
    # imported box survives untouched, the machine sibling is gone
    assert after['i1'][F.status] == RegionStatus.PENDING_DETECTION
    # (the write normalizes the stored element; compare the boxes themselves)
    assert read_boxes(after['i1']) == [read_boxes(before['i1'])[0]]
    # locked sets and other statuses are byte-identical
    for cid in ('v1', 'h1', 'ok'):
        assert after[cid] == before[cid]


@pytest.mark.asyncio
async def test_region_reverify_reproposes_unlocked_rejected_boxes_only() -> None:
    rejected = RegionStatus.VERIFY_REJECTED.value
    imported_rejected = box('b2', state='rejected', source=IMPORT, detector=IMPORT, reason=None)
    fake = make_fake(
        [
            item('m1', rejected, boxes=(box('b1'),)),
            item('only_locked', rejected, boxes=(imported_rejected,)),
        ]
    )
    before = copy.deepcopy(docs(fake))
    resp = await apply_reprocess(
        fake,
        _req(
            scopes=['region'],
            filt=ReprocessFilter(region_status=[rejected]),
            mode='reverify',
            dry_run=False,
        ),
    )
    assert _result(resp, 'region').queued == 1
    after = docs(fake)
    assert after['m1'][F.status] == RegionStatus.PENDING_VERIFICATION
    assert [(b['box_id'], b['state']) for b in after['m1'][F.boxes]] == [('b1', 'proposed')]
    assert after['only_locked'] == before['only_locked']


@pytest.mark.asyncio
async def test_region_explicit_crop_and_image_targets_any_status_and_locked_counted() -> None:
    fake = make_fake(_region_corpus())
    for cid, image in (('ok', 'img-1'),):
        docs(fake)[cid]['image_id'] = image
    crops = await apply_reprocess(
        fake, _req(scopes=['region'], crops=['ok', 'v1', 'missing'], dry_run=False)
    )
    region = _result(crops, 'region')
    assert (region.selected, region.locked_skipped, region.not_found, region.queued) == (2, 1, 1, 1)
    # the detected, unlocked item was redetected on request: a per-item
    # button is an explicit action, not a status-gated sweep
    assert docs(fake)['ok'][F.status] == RegionStatus.PENDING_DETECTION
    assert docs(fake)['ok'][F.boxes] == []


@pytest.mark.asyncio
async def test_region_profile_stale_filter_selects_detected_items_from_other_profiles() -> None:
    fake = make_fake(
        [
            item('old', RegionStatus.DETECTED.value, boxes=(box('b1', state='accepted'),),
                 **{F.profile: 'wheel', F.profile_revision: 1}),
            item('cur', RegionStatus.DETECTED.value, boxes=(box('b1', state='accepted'),),
                 **{F.profile: 'wheel', F.profile_revision: 2}),
            item('other', RegionStatus.DETECTED.value, boxes=(box('b1', state='accepted'),),
                 **{F.profile: 'plate', F.profile_revision: 9}),
            item('lock', RegionStatus.DETECTED.value, validated=True, verifier='human',
                 **{F.profile: 'plate', F.profile_revision: 9}),
        ]
    )  # fmt: skip
    filt = ReprocessFilter(profile_not='wheel', profile_revision_below=2, include_detected=True)
    plan = await plan_reprocess(fake, _req(scopes=['region'], filt=filt))
    region = plan.results[0]
    assert (region.selected, region.locked_skipped) == (3, 1)
    await apply_reprocess(fake, _req(scopes=['region'], filt=filt, dry_run=False))
    after = docs(fake)
    assert after['old'][F.status] == RegionStatus.PENDING_DETECTION
    assert after['old'][F.boxes] == []
    assert after['other'][F.status] == RegionStatus.PENDING_DETECTION
    assert after['cur'][F.status] == RegionStatus.DETECTED
    assert after['lock'][F.status] == RegionStatus.DETECTED


@pytest.mark.asyncio
async def test_detection_failed_retry_selects_exactly_what_the_old_requeue_did() -> None:
    """``filter.region_status=['detection_failed']`` is the retry path:
    the same cohort ``requeue_regions.py --status detection_failed`` moved
    (test_region_requeue pins that cohort for the engine itself)."""
    fake = make_fake(_region_corpus())
    await apply_reprocess(
        fake, _req(scopes=['region'], filt=ReprocessFilter(region_status=[FAILED]), dry_run=False)
    )
    moved = {
        cid for cid, d in docs(fake).items() if d[F.status] == RegionStatus.PENDING_DETECTION.value
    }
    assert moved == {'m1', 'i1'}


# ---------------------------------------------------------------------- vlm


def _vlm_history(prior_source: str = 'unlabeled_proposal') -> list[dict[str, Any]]:
    return [
        {
            'class_id': None,
            'class_name': None,
            'class_source': prior_source,
            'label_source': prior_source,
            'class_validated': False,
            'cluster_id': None,
            'writer': 'vlm_label_batch',
            'restorable': True,
            'at': 't0',
        }
    ]


def _vlm_corpus() -> list[dict[str, Any]]:
    attempted = {
        'vlm_class_attempted_at': 't1',
        'vlm_class_empty_reason': None,
        'vlm_raw_class': 'widgetish',
    }
    return [
        item('v1', class_source='vlm', class_id=1, class_name='widget',
             history=_vlm_history(), **attempted),
        item('imp', class_source='external_label', class_id=1, class_name='widget',
             class_validated=True, history=_vlm_history()),
        item('hum', class_source='human', class_id=1, class_name='widget', class_validated=True),
        item('hold', class_source='vlm', class_id=1, class_name='widget', test_holdout=True,
             history=_vlm_history()),
        item('nosnap', class_source='vlm', class_id=1, class_name='widget'),
        item('plain', class_source='unlabeled_proposal'),
    ]  # fmt: skip


@pytest.mark.asyncio
async def test_vlm_restores_the_pre_vlm_snapshot_and_skips_locked_items() -> None:
    fake = make_fake(_vlm_corpus())
    before = copy.deepcopy(docs(fake))
    crops = ['v1', 'imp', 'hum', 'hold', 'nosnap', 'plain']
    plan = await plan_reprocess(fake, _req(scopes=['vlm'], crops=crops))
    vlm = plan.results[0]
    assert (vlm.selected, vlm.locked_skipped, vlm.detail['eligible']) == (6, 3, 1)

    resp = await apply_reprocess(fake, _req(scopes=['vlm'], crops=crops, dry_run=False))
    assert _result(resp, 'vlm').queued == 1
    after = docs(fake)
    assert after['v1']['class_source'] == 'unlabeled_proposal'
    assert after['v1'].get('class_id') is None
    assert after['v1']['vlm_class_attempted_at'] is None
    assert after['v1']['vlm_raw_class'] is None
    assert after['v1']['class_id_history'][-1]['writer'] == 'reprocess:vlm'
    for cid in ('imp', 'hum', 'hold', 'nosnap', 'plain'):
        assert after[cid] == before[cid]


@pytest.mark.asyncio
async def test_vlm_filter_target_counts_by_query_and_applies_to_the_same_cohort() -> None:
    fake = make_fake(_vlm_corpus())
    filt = ReprocessFilter(class_id=1)
    plan = await plan_reprocess(fake, _req(scopes=['vlm'], filt=filt))
    vlm = plan.results[0]
    # class 1: v1, imp, hum, hold, nosnap are class 1; imp/hum/hold are locked
    assert (vlm.selected, vlm.locked_skipped) == (5, 3)
    assert vlm.detail['eligible'] == 2  # v1 and nosnap carry a VLM class source
    resp = await apply_reprocess(fake, _req(scopes=['vlm'], filt=filt, dry_run=False))
    assert _result(resp, 'vlm').queued == 1  # nosnap has no snapshot: nothing to restore
    assert docs(fake)['v1']['class_source'] == 'unlabeled_proposal'


# -------------------------------------------------------------------- embed


@pytest.mark.asyncio
async def test_embed_rewrites_vectors_only_and_includes_locked_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    accepted = box('b1', state='accepted', detector='det_a', reason=None)
    locked = item(
        'locked', RegionStatus.DETECTED.value, image_path=str(path), boxes=(accepted,),
        class_source='human', class_id=1, class_name='widget', class_validated=True,
        pe_embedding=[9.0, 9.0, 9.0],
        **{F.box_embeddings: [
            {'box_id': 'b1', 'bbox_norm': accepted['bbox_norm'], 'embedding': [9.0, 9.0, 9.0]}
        ]},
    )  # fmt: skip
    plain = item('plain', image_path=str(path), bbox_norm=(0.2, 0.2, 0.6, 0.6))
    fake = make_fake([locked, plain], [{'image_id': 'img-1', 'image_path': str(path)}])
    before = copy.deepcopy(docs(fake))
    image_before = copy.deepcopy(fake.docs(images_index())['img-1'])
    pe = FakePE()
    service = make_service(fake, FakeTriton([]), pe)

    resp = await apply_reprocess(
        fake, _req(scopes=['embed'], crops=['locked', 'plain'], dry_run=False),
        service_factory=_factory(service),
    )  # fmt: skip
    embed = _result(resp, 'embed')
    assert (embed.selected, embed.locked_skipped, embed.queued, embed.failed) == (1, 0, 1, 0)
    after = docs(fake)
    for cid in ('locked', 'plain'):
        changed = {k for k in after[cid] if after[cid][k] != before[cid].get(k)}
        assert changed <= {'pe_embedding', 'embedding_state', F.box_embeddings}, (cid, changed)
        assert after[cid]['pe_embedding'] == [0.0, 0.0, 1.0]
    assert current_vectors(after['locked']) == {'b1': pytest.approx([0.0, 0.0, 1.0])}
    assert F.box_embeddings not in after['plain']  # no accepted/false-positive box to embed
    image_after = fake.docs(images_index())['img-1']
    assert image_after['pe_embedding'] == [0.0, 1.0, 0.0]
    assert {k for k in image_after if image_after[k] != image_before.get(k)} == {'pe_embedding'}
    assert pe.frame_calls == 1


@pytest.mark.asyncio
async def test_embed_is_the_retry_path_for_a_failed_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    failed = item(
        'failed', image_path=str(path), bbox_norm=(0.2, 0.2, 0.6, 0.6), embedding_state='failed'
    )
    fake = make_fake([failed], [{'image_id': 'img-1', 'image_path': str(path)}])
    service = make_service(fake, FakeTriton([]), FakePE())

    await apply_reprocess(
        fake, _req(scopes=['embed'], crops=['failed'], dry_run=False),
        service_factory=_factory(service),
    )  # fmt: skip

    doc = docs(fake)['failed']
    assert doc['pe_embedding'] == [0.0, 0.0, 1.0]
    assert doc['embedding_state'] == 'embedded'


@pytest.mark.asyncio
async def test_embed_never_reads_an_unservable_stored_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servable_root(tmp_path / 'allowed', monkeypatch)
    outside = tmp_path / 'outside.jpg'
    outside.write_bytes(jpeg_bytes())
    fake = make_fake(
        [item('c1', image_path=str(outside))], [{'image_id': 'img-1', 'image_path': str(outside)}]
    )
    pe = FakePE()
    resp = await apply_reprocess(
        fake, _req(scopes=['embed'], crops=['c1'], dry_run=False),
        service_factory=_factory(make_service(fake, FakeTriton([]), pe)),
    )  # fmt: skip
    assert _result(resp, 'embed').failed == 1
    assert (pe.crop_calls, pe.frame_calls) == (0, 0)


def _factory(service: Any) -> Any:
    async def build() -> Any:
        return service

    return build


@pytest.mark.asyncio
async def test_region_missing_status_counts_only_items_the_active_profile_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An item with no region status is queued only when its class is one of
    the active profile's parent classes (ingest would never have seeded the
    others, so the worker would never pick them up)."""
    from types import SimpleNamespace

    monkeypatch.setattr(
        'src.services.curation.reprocess_region.get_active_region_profile',
        lambda: SimpleNamespace(parent_classes=('car',)),
    )
    corpus = [
        item('c1', None, class_id=1, class_name='car'),
        item('c2', None, class_id=1, class_name='CAR'),
        item('p1', None, class_id=2, class_name='person'),
    ]
    fake = make_fake(corpus)
    filt = ReprocessFilter(missing_status=True)
    resp = await apply_reprocess(fake, _req(scopes=['region'], filt=filt, dry_run=False))
    region = _result(resp, 'region')
    assert (region.selected, region.queued) == (2, 2)
    after = docs(fake)
    assert after['p1'].get(F.status) is None
    assert after['c1'][F.status] == RegionStatus.PENDING_DETECTION.value
