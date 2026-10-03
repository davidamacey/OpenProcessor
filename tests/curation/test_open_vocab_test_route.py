"""Wave 5: ``POST /open_vocab/test`` -- shows what a pass would keep, writes nothing."""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _curation_app import mount_curation_routers
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.open_vocab_fixtures import BOX, FakeSegmenter, cand
from curation.query_fakes import QueryFakeOpenSearch
from curation.reprocess_fixtures import images_index, items_index, jpeg_bytes, servable_root


URL = '/curation/projects/default/open_vocab/test'


def _seg() -> FakeSegmenter:
    seg = FakeSegmenter()
    seg.default = [
        cand(BOX, 0.9),
        cand((0.5, 0.5, 0.6, 0.6), 0.2),
    ]
    return seg


@pytest.fixture
def env(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    fake = QueryFakeOpenSearch(
        {
            items_index(): {
                'locked-1': {
                    'crop_id': 'locked-1',
                    'image_id': 'img-1',
                    'bbox_norm': list(BOX),
                    'class_source': 'human',
                    'class_validated': True,
                }
            },
            images_index(): {'img-1': {'image_id': 'img-1', 'image_path': str(path)}},
        }
    )
    seg = _seg()
    monkeypatch.setattr('src.routers.curation.open_vocab_test.segment_image_http', seg)
    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr(
        'src.config.ingest_profiles.ingest_primary_profile',
        lambda: type('P', (), {'detector_model': ''})(),
    )
    monkeypatch.delenv('OP_SEGMENTER_URL', raising=False)
    app = FastAPI()
    mount_curation_routers(app, curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return {'client': TestClient(app), 'fake': fake, 'seg': seg, 'path': path}


def _body(**kw: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        'image_id': 'img-1',
        'target': {'prompt': 'traffic cone', 'class_name': 'cone'},
    }
    body.update(kw)
    return body


def test_it_shows_every_candidate_and_why_it_was_dropped_and_writes_nothing(
    env: dict[str, Any],
) -> None:
    fake = env['fake']
    before = {
        i: {k: dict(v) for k, v in fake.docs(i).items()} for i in (items_index(), images_index())
    }

    r = env['client'].post(URL, json=_body())

    assert r.status_code == 200, r.text
    out = r.json()
    assert out['image'] == {'width': 400, 'height': 300}
    by_score = {h['score']: h for h in out['hits']}
    # The strong hit sits on a locked item: shown, with the reason.
    assert by_score[0.9]['selected'] is False
    assert by_score[0.9]['drop_reason'] == 'skipped_locked'
    assert by_score[0.9]['mask_polygon'] == [[0.1, 0.2], [0.3, 0.2], [0.3, 0.5]]
    assert by_score[0.2]['drop_reason'] == 'below_min_score'
    assert out['prompt'] == 'traffic cone'
    assert env['seg'].calls[0]['return_masks'] is True
    after = {
        i: {k: dict(v) for k, v in fake.docs(i).items()} for i in (items_index(), images_index())
    }
    assert after == before


def test_an_uploaded_image_works_without_a_stored_one(env: dict[str, Any]) -> None:
    payload = base64.b64encode(env['path'].read_bytes()).decode()
    r = env['client'].post(URL, json={'image_base64': payload, 'target': {'prompt': 'cup'}})
    assert r.status_code == 200, r.text
    assert all(h['selected'] for h in r.json()['hits'] if h['score'] == 0.9)


def test_exactly_one_image_source_is_required(env: dict[str, Any]) -> None:
    c = env['client']
    assert c.post(URL, json={'target': {'prompt': 'cup'}}).status_code == 422
    assert c.post(URL, json=_body(image_base64='AAAA')).status_code == 422


def test_garbage_upload_and_unknown_image_are_refused(env: dict[str, Any]) -> None:
    c = env['client']
    assert c.post(URL, json={'image_base64': '!!!', 'target': {'prompt': 'cup'}}).status_code == 422
    bad = base64.b64encode(b'not an image').decode()
    assert c.post(URL, json={'image_base64': bad, 'target': {'prompt': 'cup'}}).status_code == 422
    r = c.post(URL, json=_body(image_id='nope'))
    assert (r.status_code, r.json()['detail']['error']) == (404, 'image_not_found')
    assert env['seg'].calls == []


def test_an_invalid_target_is_refused_before_any_call(env: dict[str, Any]) -> None:
    r = env['client'].post(URL, json=_body(target={'prompt': '', 'class_name': 'cone'}))
    assert r.status_code == 422
    assert r.json()['detail']['report']['errors'][0]['code'] == 'segmenter_prompt_empty'
    assert env['seg'].calls == []


def test_a_segmenter_outage_is_a_502_not_no_hits(env: dict[str, Any]) -> None:
    env['seg'].down = True
    r = env['client'].post(URL, json=_body())
    assert (r.status_code, r.json()['detail']['error']) == (502, 'segmenter_error')
