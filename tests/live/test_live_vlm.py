"""Live cohort-(c) scenarios: the VLM-backed write paths, plus the
restart-durability check.

What the fake VLM proves — and what it does not
-----------------------------------------------
The fake returns a **fixed** answer per prompt kind. That makes these
tests a proof of *wiring and persistence*: the router builds a
well-formed OpenAI request, the labeler's parser understands the reply,
and the parsed verdict reaches OpenSearch under the documented field
names. It proves **nothing about label quality** — no real model is
involved anywhere in this file.

This module runs last in the package (alphabetically), which is
deliberate: its final test recreates the API container, so nothing may
depend on in-process state afterwards.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from .conftest import INDEXES, VERIFY_DATA_DIR, compose, crop_ids_in, get_doc, refresh, wait_until


pytestmark = pytest.mark.live

FAKE_CLASS = 'box'
FAKE_REGION_TEXT = 'FAKE-LABEL-0042'


def _source(opensearch: Any, crop_id: str) -> dict[str, Any]:
    return get_doc(opensearch, INDEXES['items'], crop_id)['_source']


@pytest.fixture(scope='module')
def sample_jpeg_b64() -> str:
    """One of the seeded source JPEGs, base64-encoded.

    The batch VLM endpoints take caller-supplied JPEG bytes rather than
    re-deriving a crop from OpenSearch, so the test supplies real pixels.
    """
    images = sorted((VERIFY_DATA_DIR / 'images').glob('*.jpg'))
    assert images, f'no seeded images under {VERIFY_DATA_DIR / "images"}'
    return base64.b64encode(images[0].read_bytes()).decode()


def test_vlm_label_batch_persists_the_exact_fake_answer(
    kb: Any, opensearch: Any, fake_vlm: Any
) -> None:
    crop_ids = crop_ids_in(opensearch, 'cnd10001', limit=4)
    assert len(crop_ids) == 4

    resp = kb.post('/vlm/label_batch', json={'crop_ids': crop_ids})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['predicted'] == len(crop_ids), body
    assert body['updated'] == len(crop_ids), body
    assert body['new_class_proposals'] == []

    refresh(opensearch, INDEXES['items'])
    for crop_id in crop_ids:
        src = _source(opensearch, crop_id)
        assert src['class_name'] == FAKE_CLASS, crop_id
        assert src['class_source'] == 'vlm'
        assert src['label_source'] == 'vlm'
        assert src['vlm_confidence'] == 'high'
        # The raw answer is captured verbatim on every branch so a terms
        # agg can quantify what the model actually said.
        assert src['vlm_raw_label'] == FAKE_CLASS
        assert src['class_labeler'] == 'vlm'
        # A VLM write must never claim human validation.
        assert src.get('class_validated') is not True

    counts = fake_vlm.get('/__stats').json()['counts']
    assert counts.get('class_open', 0) >= 1, counts


def test_vlm_label_batch_leaves_human_owned_crops_alone(kb: Any, opensearch: Any) -> None:
    human_ids = [
        h['_id']
        for h in opensearch.post(
            f'/{INDEXES["items"]}/_search',
            json={
                'size': 2,
                '_source': False,
                'query': {'term': {'class_source': 'human'}},
            },
        ).json()['hits']['hits']
    ]
    assert human_ids, 'precondition: human-labelled crops must exist'
    before = {cid: _source(opensearch, cid)['class_id'] for cid in human_ids}

    resp = kb.post('/vlm/label_batch', json={'crop_ids': human_ids})
    assert resp.status_code == 200, resp.text
    assert resp.json()['updated'] == 0, resp.text

    for cid, class_id in before.items():
        assert _source(opensearch, cid)['class_id'] == class_id


def test_vlm_label_batch_caps_the_request_size(kb: Any) -> None:
    resp = kb.post('/vlm/label_batch', json={'crop_ids': [f'x{i}' for i in range(65)]})
    assert resp.status_code == 400, resp.text


def test_vlm_verify_regions_persists_the_verdict(kb: Any, opensearch: Any, fake_vlm: Any) -> None:
    crop_ids = crop_ids_in(opensearch, 'rgn', limit=40)[-3:]

    resp = kb.post('/vlm/verify_regions', json={'crop_ids': crop_ids})
    assert resp.status_code == 200, resp.text
    assert resp.json()['verified'] == len(crop_ids), resp.text

    refresh(opensearch, INDEXES['items'])
    for crop_id in crop_ids:
        src = _source(opensearch, crop_id)
        assert src['region_verified'] is True, crop_id
        assert src['region_reason'], crop_id


def test_vlm_verify_region_batch_returns_ordered_verdicts(
    kb: Any, opensearch: Any, fake_vlm: Any, sample_jpeg_b64: str
) -> None:
    crop_ids = crop_ids_in(opensearch, 'rgnfp', limit=3)
    payload = {
        'items': [
            {
                'crop_id': crop_id,
                'region_image_b64': sample_jpeg_b64,
                'candidate_text': f'candidate-{i}',
            }
            for i, crop_id in enumerate(crop_ids)
        ]
    }
    resp = kb.post('/vlm/verify_region_batch', json=payload)
    assert resp.status_code == 200, resp.text
    results = resp.json()['results']
    assert [r['crop_id'] for r in results] == crop_ids
    for i, result in enumerate(results):
        assert result['is_region'] is True, result
        assert result['confidence'] == 'high'
        assert result['candidate_text'] == f'candidate-{i}'


def test_vlm_verify_region_batch_rejects_bad_input(
    kb: Any, opensearch: Any, sample_jpeg_b64: str
) -> None:
    crop_id = crop_ids_in(opensearch, 'rgnfp', limit=1)[0]
    bad_b64 = kb.post(
        '/vlm/verify_region_batch',
        json={'items': [{'crop_id': crop_id, 'region_image_b64': 'not base64!!'}]},
    )
    assert bad_b64.status_code == 400, bad_b64.text

    duplicate = kb.post(
        '/vlm/verify_region_batch',
        json={
            'items': [
                {'crop_id': crop_id, 'region_image_b64': sample_jpeg_b64},
                {'crop_id': crop_id, 'region_image_b64': sample_jpeg_b64},
            ]
        },
    )
    assert duplicate.status_code == 400, duplicate.text


def test_vlm_verdict_key_matches_the_shipped_prompt(
    kb: Any, opensearch: Any, fake_vlm: Any, sample_jpeg_b64: str
) -> None:
    """The built-in prompt pack asks the model to answer with `is_region`
    (vlm_prompts.GENERIC_ITEM_PACK.region_user / region_batch_user), and
    both region parsers in vlm_labeler.py now read `is_region` too — a
    model that follows the shipped prompt exactly is parsed correctly.

    The fake defaults to `is_region` already; setting it explicitly here
    just documents the invariant this test exists to protect.
    """
    fake_vlm.post('/__control', json={'region_verdict_key': 'is_region'})
    crop_ids = crop_ids_in(opensearch, 'rgnfp', limit=2)
    resp = kb.post(
        '/vlm/verify_region_batch',
        json={
            'items': [
                {'crop_id': crop_id, 'region_image_b64': sample_jpeg_b64} for crop_id in crop_ids
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()['results']
    assert all(r['is_region'] is True for r in results), results


def test_vlm_region_visible_batch_honours_an_explicit_negative(
    kb: Any, opensearch: Any, fake_vlm: Any, sample_jpeg_b64: str
) -> None:
    """The visibility endpoint fails *open* to True on any RPC/parse
    failure, so only an explicit negative distinguishes "the fake was
    understood" from "the fake was ignored"."""
    fake_vlm.post('/__control', json={'region_visible': False})
    crop_ids = crop_ids_in(opensearch, 'cnd10002', limit=3)

    resp = kb.post(
        '/vlm/region_visible_batch',
        json={
            'items': [{'crop_id': crop_id, 'image_b64': sample_jpeg_b64} for crop_id in crop_ids]
        },
    )
    assert resp.status_code == 200, resp.text
    visible = resp.json()['visible']
    assert set(visible) == set(crop_ids)
    assert all(v is False for v in visible.values()), visible

    fake_vlm.post('/__control', json={'region_visible': True})
    resp = kb.post(
        '/vlm/region_visible_batch',
        json={'items': [{'crop_id': crop_ids[0], 'image_b64': sample_jpeg_b64}]},
    )
    assert resp.json()['visible'][crop_ids[0]] is True


def test_state_survives_an_api_container_restart(kb: Any, opensearch: Any) -> None:
    """Recreate ONLY the API container and re-read everything.

    Every write this suite made lives in OpenSearch or on a mounted
    volume, never in process memory — a restarted API must observe an
    identical world.
    """
    labelled = crop_ids_in(opensearch, 'cnd10001', limit=4)
    before_docs = {cid: _source(opensearch, cid)['class_name'] for cid in labelled}
    before_classes = kb.get('/classes').json()['classes']
    before_holdout = kb.get('/test_holdout/stats').json()['total']
    before_exports = kb.get('/export/datasets').json()['count']

    compose('up', '-d', '--force-recreate', '--no-deps', 'api', timeout=600)

    import httpx

    ready = wait_until(
        lambda: _probe(httpx, 'http://localhost:14701/live'), timeout=180, interval=2.0
    )
    assert ready, 'the API container did not come back up'

    after_classes = kb.get('/classes').json()['classes']
    assert [c['class_id'] for c in after_classes] == [c['class_id'] for c in before_classes]
    assert [c['class_name'] for c in after_classes] == [c['class_name'] for c in before_classes]
    assert kb.get('/test_holdout/stats').json()['total'] == before_holdout
    assert kb.get('/export/datasets').json()['count'] == before_exports
    for crop_id, class_name in before_docs.items():
        assert kb.get(f'/crops/{crop_id}').json()['class_name'] == class_name


def _probe(httpx_module: Any, url: str) -> bool:
    try:
        return httpx_module.get(url, timeout=5.0).status_code == 200
    except Exception:
        return False
