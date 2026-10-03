"""The ``embed`` scope as embed-missing: item-unit selection, only the missing,
cluster placement of a newly embedded item, dry run equal to the applied run."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from curation.reprocess_fixtures import (
    FakePE,
    FakeTriton,
    docs,
    images_index,
    item,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.services.curation import reprocess_embed, reprocess_job
from src.services.curation.ingest_index import PARKED_CLUSTER_ID
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_models import (
    EmbedOptions,
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)
from src.services.curation.reprocess_targets import ReprocessTargetsError, validate_targets


if TYPE_CHECKING:
    from pathlib import Path

BOX = (0.1, 0.1, 0.5, 0.5)


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, FakePE, Any]:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    p = str(path)
    items = [
        item(
            'car', image_path=p, bbox_norm=BOX, proposal_name='car', embedding_state='not_selected'
        ),
        item(
            'dog', image_path=p, bbox_norm=BOX, proposal_name='dog', embedding_state='not_selected'
        ),
        item(
            'have',
            image_path=p,
            bbox_norm=BOX,
            proposal_name='car',
            embedding_state='embedded',
            pe_embedding=[9.0, 9.0, 9.0],
        ),
        item('bad', image_path=p, bbox_norm=BOX, proposal_name='dog', embedding_state='failed'),
        item(
            'human',
            image_path=p,
            bbox_norm=BOX,
            class_source='human',
            class_id=1,
            class_name='widget',
            class_validated=True,
            cluster_id=1,
            embedding_state='deferred',
        ),
    ]
    fake = make_fake(
        items, [{'image_id': 'img-1', 'image_path': p, 'pe_embedding': [0.5, 0.5, 0.5]}]
    )
    pe = FakePE()
    return fake, pe, make_service(fake, FakeTriton([]), pe)


def _req(filt: ReprocessFilter | None = None, *, crops: list[str] | None = None, **kw: Any) -> Any:
    only_missing = kw.pop('only_missing', True)
    return ReprocessRequest(
        targets=ReprocessTargets(filter=filt, crop_ids=crops, **kw),
        scopes=['embed'],
        embed=EmbedOptions(only_missing=only_missing),
        dry_run=kw.pop('dry_run', False),
    )


def _factory(service: Any) -> Any:
    async def build() -> Any:
        return service

    return build


@pytest.mark.asyncio
async def test_a_class_filter_embeds_only_that_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, pe, service = _world(tmp_path, monkeypatch)
    resp = await apply_reprocess(
        fake,
        _req(ReprocessFilter(class_names=['car'], embedding_state=['not_selected'])),
        service_factory=_factory(service),
    )
    assert pe.crop_calls == 1
    after = docs(fake)
    assert after['car']['embedding_state'] == 'embedded'
    assert after['car']['pe_embedding'] == [0.0, 0.0, 1.0]
    for untouched in ('dog', 'bad', 'human'):
        assert 'pe_embedding' not in after[untouched], untouched
    assert resp.scopes[0].detail['crop_written'] == 1


@pytest.mark.asyncio
async def test_only_missing_skips_an_item_that_already_has_a_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, pe, service = _world(tmp_path, monkeypatch)
    await apply_reprocess(fake, _req(crops=['car', 'have']), service_factory=_factory(service))
    assert pe.crop_calls == 1
    assert docs(fake)['have']['pe_embedding'] == [9.0, 9.0, 9.0]  # kept
    assert pe.frame_calls == 0  # only-missing leaves the frame vector alone
    assert fake.docs(images_index())['img-1']['pe_embedding'] == [0.5, 0.5, 0.5]


@pytest.mark.asyncio
async def test_without_only_missing_the_selection_is_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, pe, service = _world(tmp_path, monkeypatch)
    await apply_reprocess(
        fake, _req(crops=['car', 'have'], only_missing=False), service_factory=_factory(service)
    )
    assert pe.crop_calls == 2
    assert docs(fake)['have']['pe_embedding'] == [0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_failed_items_are_retried_without_touching_the_rest_of_the_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, pe, service = _world(tmp_path, monkeypatch)
    await apply_reprocess(
        fake, _req(ReprocessFilter(embedding_state=['failed'])), service_factory=_factory(service)
    )
    assert pe.crop_calls == 1
    assert docs(fake)['bad']['embedding_state'] == 'embedded'
    assert docs(fake)['car']['embedding_state'] == 'not_selected'


@pytest.mark.asyncio
async def test_a_locked_item_is_embedded_but_its_class_fields_never_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _, service = _world(tmp_path, monkeypatch)
    before = dict(docs(fake)['human'])
    await apply_reprocess(fake, _req(crops=['human']), service_factory=_factory(service))
    after = docs(fake)['human']
    assert after['embedding_state'] == 'embedded'
    assert {k for k in after if after[k] != before.get(k)} == {'pe_embedding', 'embedding_state'}


class _Store:
    def assign_one_with_distance(self, _vec: Any) -> tuple[int, float]:
        return 4, 0.25


@pytest.mark.asyncio
async def test_a_newly_embedded_unclassed_item_is_placed_like_an_ingested_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reprocess_embed, 'get_ivf_ingest_store', lambda: _Store())
    fake, _, service = _world(tmp_path, monkeypatch)
    await apply_reprocess(fake, _req(crops=['car', 'human']), service_factory=_factory(service))
    car = docs(fake)['car']
    assert (car['cluster_id'], car['cluster_distance']) == (10_004, 0.25)
    assert car['cluster_distance_cluster_id'] == 10_004
    assert docs(fake)['human']['cluster_id'] == 1  # a class-labeled item keeps its class cluster


@pytest.mark.asyncio
async def test_an_item_failing_the_clustering_gate_is_parked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(reprocess_embed, 'get_ivf_ingest_store', lambda: _Store())
    monkeypatch.setattr('src.services.curation.ingest_index.ingest_passes_gate', lambda *_a: False)
    fake, _, service = _world(tmp_path, monkeypatch)
    await apply_reprocess(fake, _req(crops=['car']), service_factory=_factory(service))
    assert docs(fake)['car']['cluster_id'] == PARKED_CLUSTER_ID


@pytest.mark.asyncio
async def test_dry_run_counts_equal_the_applied_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _, service = _world(tmp_path, monkeypatch)
    filt = ReprocessFilter(origin=['detector'])
    dry = await apply_reprocess(
        fake,
        ReprocessRequest(
            targets=ReprocessTargets(filter=filt),
            scopes=['embed'],
            embed=EmbedOptions(only_missing=True),
            dry_run=True,
        ),
        service_factory=_factory(service),
    )
    planned = dry.scopes[0].detail
    assert (
        planned['to_embed'] == 3
    )  # car, dog, bad; 'have' has a vector, 'human' is not detector-made
    assert planned['without_vector'] == 3
    assert planned['estimated_vector_kb'] == 3 * 1024 * 4 // 1000
    assert planned['region_boxes_to_embed'] == 0  # none of these items has a box
    applied = await apply_reprocess(
        fake,
        _req(filt),
        service_factory=_factory(service),
    )
    assert applied.scopes[0].detail['crop_written'] == planned['to_embed']


@pytest.mark.asyncio
async def test_a_cap_embeds_only_the_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake, pe, service = _world(tmp_path, monkeypatch)
    req = ReprocessRequest(
        targets=ReprocessTargets(
            filter=ReprocessFilter(embedding_state=['not_selected', 'failed', 'deferred']),
            limit=2,
            sample='random',
            seed=3,
        ),
        scopes=['embed'],
        embed=EmbedOptions(only_missing=True),
        dry_run=False,
    )
    await apply_reprocess(fake, req, service_factory=_factory(service))
    assert pe.crop_calls == 2


@pytest.mark.asyncio
async def test_a_second_concurrent_job_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_REPROCESS_JOBS_DIR', str(tmp_path / 'jobs'))
    reprocess_job.create_job(request={}, scopes=['embed'], image_ids=['a'])
    with pytest.raises(reprocess_job.ReprocessBusyError):
        reprocess_job.create_job(request={}, scopes=['embed'], image_ids=['b'])


def test_limit_and_sample_apply_to_a_filter_only() -> None:
    with pytest.raises(ReprocessTargetsError):
        validate_targets(ReprocessTargets(crop_ids=['a'], limit=2))
    with pytest.raises(ReprocessTargetsError):
        validate_targets(ReprocessTargets(filter=ReprocessFilter(source='s'), sample='random'))
    with pytest.raises(ReprocessTargetsError):
        validate_targets(ReprocessTargets(filter=ReprocessFilter(conf_min=0.9, conf_max=0.1)))
