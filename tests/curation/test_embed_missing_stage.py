"""The auto-label run's ``embed_missing`` stage: first in the run, limited to
the run's scope, skipped when nothing is missing, cancellable at its boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from curation.reprocess_fixtures import FakePE, docs, item, jpeg_bytes, make_fake, servable_root
from curation.test_auto_label_selection import client, job_dir, packs  # noqa: F401 - fixtures
from src.services.curation.autolabel.embed_stage import run_embed_missing_stage, stage_request
from src.services.curation.autolabel.job import STAGES
from src.services.curation.autolabel.selection import unvalidated_count_query, vlm_selection_query
from src.services.curation.item_filter import ItemFilter


if TYPE_CHECKING:
    from pathlib import Path

BOX = (0.1, 0.1, 0.5, 0.5)


def _world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    p = str(path)
    items = [
        item(
            'car1',
            image_path=p,
            bbox_norm=BOX,
            proposal_name='car',
            class_id=3,
            embedding_state='deferred',
        ),
        item(
            'car2',
            image_path=p,
            bbox_norm=BOX,
            proposal_name='car',
            class_id=3,
            embedding_state='deferred',
        ),
        item(
            'dog1',
            image_path=p,
            bbox_norm=BOX,
            proposal_name='dog',
            class_id=7,
            embedding_state='deferred',
        ),
        item(
            'done',
            image_path=p,
            bbox_norm=BOX,
            proposal_name='car',
            class_id=3,
            embedding_state='embedded',
            pe_embedding=[9.0, 9.0, 9.0],
        ),
    ]
    return make_fake(items, [{'image_id': 'img-1', 'image_path': p}])


def test_the_stage_runs_first() -> None:
    assert STAGES[0] == 'embed_missing'


@pytest.mark.asyncio
async def test_the_stage_embeds_only_the_run_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _world(tmp_path, monkeypatch)
    pe = FakePE()
    out = await run_embed_missing_stage(
        fake, class_id=3, cluster_id=None, item_filter=None, encoder=pe
    )
    assert out == {'images': 1, 'images_failed': 0, 'embedded': 2}
    after = docs(fake)
    assert after['car1']['embedding_state'] == after['car2']['embedding_state'] == 'embedded'
    assert 'pe_embedding' not in after['dog1']  # other class: out of scope
    assert after['done']['pe_embedding'] == [9.0, 9.0, 9.0]  # already embedded: left alone
    assert pe.crop_calls == 2


@pytest.mark.asyncio
async def test_the_item_filter_narrows_the_scope_further(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _world(tmp_path, monkeypatch)
    out = await run_embed_missing_stage(
        fake,
        class_id=None,
        cluster_id=None,
        item_filter={'class_names': ['dog']},
        encoder=FakePE(),
    )
    assert out['embedded'] == 1
    assert docs(fake)['dog1']['embedding_state'] == 'embedded'
    assert 'pe_embedding' not in docs(fake)['car1']


@pytest.mark.asyncio
async def test_nothing_missing_is_skipped_without_building_an_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _world(tmp_path, monkeypatch)

    def _no_pool(*_: Any, **__: Any) -> None:
        raise AssertionError('an encoder must not be built when nothing is missing')

    monkeypatch.setattr('src.clients.triton_pool.AsyncTritonPool', _no_pool)
    out = await run_embed_missing_stage(
        fake, class_id=None, cluster_id=None, item_filter={'class_names': ['bird']}
    )
    assert out == {'skipped': True, 'reason': 'nothing to embed'}


@pytest.mark.asyncio
async def test_cancel_at_the_stage_boundary_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Cancelled:
        def update(self, **_: Any) -> None:
            return None

        def raise_if_cancelled(self) -> None:
            raise asyncio.CancelledError

    fake = _world(tmp_path, monkeypatch)
    pe = FakePE()
    with pytest.raises(asyncio.CancelledError):
        await run_embed_missing_stage(
            fake, class_id=None, cluster_id=None, item_filter=None, progress=Cancelled(), encoder=pe
        )
    assert pe.crop_calls == 0


@pytest.mark.asyncio
async def test_an_encoder_outage_is_reported_and_does_not_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Down(FakePE):
        async def embed_crops(self, crops: list[Any], max_batch: int = 32) -> Any:  # noqa: ARG002
            raise RuntimeError('triton down')

    fake = _world(tmp_path, monkeypatch)
    out = await run_embed_missing_stage(
        fake, class_id=None, cluster_id=None, item_filter=None, encoder=Down()
    )
    assert out['embedded'] == 0
    assert out['images_failed'] == 1


def test_the_vlm_selection_and_counts_honor_the_item_filter() -> None:
    flt = ItemFilter(class_names=['car'])
    selection = vlm_selection_query(
        class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8, item_filter=flt
    )
    assert 'case_insensitive' in str(selection)
    assert 'case_insensitive' in str(
        unvalidated_count_query(class_id=None, cluster_id=None, item_filter=flt)
    )
    plain = vlm_selection_query(class_id=None, cluster_id=None, classifier_confidence_skip_vlm=0.8)
    assert 'case_insensitive' not in str(plain)


def test_the_stage_request_selects_only_vectorless_items_in_scope() -> None:
    request = stage_request(3, None, ItemFilter(class_names=['car']))
    flt = request.targets.filter
    assert flt is not None
    assert (flt.class_id, flt.class_names) == (3, ['car'])
    assert flt.embedding_state == ['not_selected', 'deferred', 'failed']
    assert request.embed.only_missing is True
    assert request.dry_run is False


@pytest.mark.usefixtures('packs', 'job_dir')
def test_start_carries_the_scope_and_embed_flag_in_the_job_trigger(
    request: pytest.FixtureRequest,
) -> None:
    http = request.getfixturevalue('client')
    r = http.post(
        '/curation/projects/default/pipeline/auto_label/start',
        params={'embed_missing': 'true', 'class_name': 'car', 'min_area': '0.1', 'class_id': 3},
    )
    assert r.status_code == 200, r.text
    args = r.json()['args']
    assert args['embed_missing'] is True
    assert args['class_id'] == 3
    assert args['item_filter'] == {'class_names': ['car'], 'min_area': 0.1}
