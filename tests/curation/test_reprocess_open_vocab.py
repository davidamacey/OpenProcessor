"""Wave 5: the ``open_vocab`` reprocess scope -- dry-run estimate, locked
count, active-set requirement, image-level selectors, outage handling."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest

from curation.open_vocab_fixtures import FakeSegmenter, StatefulRegistry, cand, ingested_world
from curation.reprocess_fixtures import docs, images_index
from src.services.config_store.store import StoredConfig, get_config_store, reset_config_stores
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_models import (
    ReprocessFilter,
    ReprocessRequest,
    ReprocessTargets,
)
from src.services.curation.reprocess_targets import ReprocessTargetsError


if TYPE_CHECKING:
    from pathlib import Path

    from curation.query_fakes import QueryFakeOpenSearch
    from src.services.curation.ingest import CurationIngestService


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Any:
    reset_config_stores()
    registry = StatefulRegistry()
    monkeypatch.setattr('src.services.curation.open_vocab_run.get_class_registry', lambda: registry)
    monkeypatch.delenv('OP_SEGMENTER_URL', raising=False)
    yield
    reset_config_stores()


def _activate(targets: list[dict[str, Any]], revision: int = 4) -> None:
    store = get_config_store()
    body = {'targets': targets}
    stored = StoredConfig(kind='open_vocab_set', name='street', revision=revision, body=body)
    store.apply_local(
        config_revision=1,
        open_vocab_set=stored,
        active_open_vocab=('street', revision),
        active_open_vocab_body=stored,
    )


async def _world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_images: int = 2
) -> tuple[QueryFakeOpenSearch, CurationIngestService, list[str]]:
    return await ingested_world(tmp_path, monkeypatch, n=n_images)


def _req(
    *, image_ids: list[str] | None = None, flt: ReprocessFilter | None = None, dry_run: bool = False
) -> ReprocessRequest:
    return ReprocessRequest(
        targets=ReprocessTargets(image_ids=image_ids, filter=flt),
        scopes=['open_vocab'],
        dry_run=dry_run,
    )


def _factory(service: Any) -> Any:
    async def build() -> Any:
        return service

    return build


@pytest.mark.asyncio
async def test_a_dry_run_estimates_calls_and_minutes_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _service, ids = await _world(tmp_path, monkeypatch)
    _activate([{'prompt': 'a', 'class_name': 'a'}, {'prompt': 'b', 'class_name': 'b'}])
    docs(fake)['human-1'] = {
        'crop_id': 'human-1',
        'image_id': ids[0],
        'bbox_norm': [0, 0, 1, 1],
        'class_source': 'human',
        'class_validated': True,
    }
    before = copy.deepcopy(docs(fake))

    resp = await apply_reprocess(fake, _req(image_ids=ids, dry_run=True))

    (res,) = resp.scopes
    assert (res.scope, res.selected, res.locked_skipped) == ('open_vocab', 2, 1)
    assert res.detail['enabled_targets'] == 2
    assert res.detail['estimated_calls'] == 4
    assert res.detail['segmenter_reachable'] == 0
    assert res.detail['estimated_minutes'] == 1  # 4 calls x 3 s on one instance, rounded up
    assert docs(fake) == before


@pytest.mark.asyncio
async def test_the_estimate_divides_by_the_segmenters_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _service, _ids = await _world(tmp_path, monkeypatch, n_images=2)
    _activate([{'prompt': 'a', 'class_name': 'a'}])
    monkeypatch.setenv('OP_SEGMENTER_URL', 'http://seg.invalid:8000')

    async def instances(_url: str) -> int:
        return 2

    monkeypatch.setattr('src.services.curation.reprocess_open_vocab.segmenter_instances', instances)
    many = [f'img-{i}' for i in range(200)]  # 200 calls x 3 s / 2 instances = 300 s = 5 min
    resp = await apply_reprocess(fake, _req(image_ids=many, dry_run=True))
    detail = resp.scopes[0].detail
    assert (detail['segmenter_instances'], detail['segmenter_reachable']) == (2, 1)
    assert detail['estimated_minutes'] == 5


@pytest.mark.asyncio
async def test_no_active_set_is_a_refusal_not_an_empty_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _service, ids = await _world(tmp_path, monkeypatch)
    with pytest.raises(ReprocessTargetsError, match='no open-vocabulary set is active'):
        await apply_reprocess(fake, _req(image_ids=ids, dry_run=True))


@pytest.mark.asyncio
async def test_a_run_writes_items_and_reports_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, ids = await _world(tmp_path, monkeypatch)
    _activate([{'prompt': 'traffic cone', 'class_name': 'cone'}])
    seg = FakeSegmenter()
    seg.default = [cand()]
    monkeypatch.setattr('src.services.curation.reprocess_images.segment_image_http', seg)

    resp = await apply_reprocess(fake, _req(image_ids=ids), service_factory=_factory(service))

    (res,) = resp.scopes
    assert (res.queued, res.failed) == (2, 0)
    assert res.detail['calls'] == 2
    assert res.detail['written'] == 2
    assert {d['image_id'] for d in docs(fake).values()} == set(ids)
    assert {d['open_vocab_revision'] for d in docs(fake).values()} == {4}
    assert len(seg.calls) == 2


@pytest.mark.asyncio
async def test_the_segmenter_being_down_trips_after_three_images_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, service, ids = await _world(tmp_path, monkeypatch, n_images=5)
    _activate([{'prompt': 'traffic cone', 'class_name': 'cone'}])
    seg = FakeSegmenter()
    seg.default = [cand()]
    seg.down = True
    monkeypatch.setattr('src.services.curation.reprocess_images.segment_image_http', seg)

    resp = await apply_reprocess(fake, _req(image_ids=ids), service_factory=_factory(service))

    (res,) = resp.scopes
    assert (res.queued, res.failed) == (0, 3)
    assert res.detail['failed_segmenter_unavailable'] == 3
    assert res.detail['not_attempted_segmenter_down'] == 2
    assert docs(fake) == {}
    assert len(seg.calls) == 3


@pytest.mark.asyncio
async def test_all_images_reaches_images_that_have_no_item_yet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _service, ids = await _world(tmp_path, monkeypatch)
    _activate([{'prompt': 'a', 'class_name': 'a'}])
    assert docs(fake) == {}  # no item anywhere

    resp = await apply_reprocess(fake, _req(flt=ReprocessFilter(all_images=True), dry_run=True))
    assert resp.scopes[0].selected == len(ids)


@pytest.mark.asyncio
async def test_open_vocab_status_selects_only_the_images_in_that_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _service, ids = await _world(tmp_path, monkeypatch)
    _activate([{'prompt': 'a', 'class_name': 'a'}])
    fake.docs(images_index())[ids[0]]['open_vocab_status'] = 'pending'
    fake.docs(images_index())[ids[1]]['open_vocab_status'] = 'done'

    resp = await apply_reprocess(
        fake, _req(flt=ReprocessFilter(open_vocab_status=['pending']), dry_run=True)
    )
    assert resp.scopes[0].selected == 1


@pytest.mark.asyncio
async def test_image_selectors_do_not_combine_with_item_selectors_or_item_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, _service, _ids = await _world(tmp_path, monkeypatch)
    _activate([{'prompt': 'a', 'class_name': 'a'}])
    with pytest.raises(ReprocessTargetsError, match='cannot be combined'):
        await apply_reprocess(
            fake, _req(flt=ReprocessFilter(all_images=True, class_id=1), dry_run=True)
        )
    region = ReprocessRequest(
        targets=ReprocessTargets(filter=ReprocessFilter(all_images=True)),
        scopes=['region'],
        dry_run=True,
    )
    with pytest.raises(ReprocessTargetsError, match='select images'):
        await apply_reprocess(fake, region)
