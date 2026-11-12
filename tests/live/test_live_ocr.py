"""Live: reading text and finding items by region text.

Two halves, split by what the harness can serve:

* ``POST /ocr/predict`` runs the real route, client and response mapping against the fake
  Triton's ``ocr_pipeline`` (``docker/test/fake_triton.py``: one fixed line, ``FAKE-PLATE-0042``).
* The region text a box carries and the filters over it (``/crops?region_text=`` exact, fuzzy,
  fragment, and ``same_text_as``) run against the real OpenSearch mapping. The text is written
  through ``PATCH /crops/{id}/regions/{box}`` (a human reading). The harness has no region-stage
  worker (the cascade needs a detector plus the OCR leg in the ``curation-detection-worker``), so
  OCR-written text in the cascade is not covered here.
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from PIL import Image

from .conftest import INDEXES, crop_ids_in, get_doc, refresh


pytestmark = pytest.mark.live

FAKE_OCR_TEXT = 'FAKE-PLATE-0042'


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (320, 160), (240, 240, 240)).save(buf, format='JPEG')
    return buf.getvalue()


def test_ocr_predict_returns_the_pipeline_text_and_boxes(api: Any) -> None:
    resp = api.post('/ocr/predict', files={'image': ('plate.jpg', _jpeg(), 'image/jpeg')})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body['status'] == 'success'
    assert body['num_texts'] == 1, body
    assert body['texts'] == [FAKE_OCR_TEXT]
    assert body['full_text'] == FAKE_OCR_TEXT
    (region,) = body['regions']
    assert region['text'] == FAKE_OCR_TEXT
    assert region['box_normalized'] == pytest.approx([0.1, 0.4, 0.9, 0.6])
    assert region['rec_score'] == pytest.approx(0.97, abs=1e-3)
    assert body['image'] == {'width': 320, 'height': 160}


def test_ocr_predict_filters_by_recognition_score_and_refuses_an_empty_upload(api: Any) -> None:
    kept = api.post(
        '/ocr/predict',
        params={'min_rec_score': 0.99},
        files={'image': ('plate.jpg', _jpeg(), 'image/jpeg')},
    )
    assert kept.status_code == 200, kept.text
    assert kept.json()['num_texts'] == 0, kept.text
    unfiltered = api.post(
        '/ocr/predict',
        params={'min_rec_score': 0.99, 'filter_by_score': False},
        files={'image': ('plate.jpg', _jpeg(), 'image/jpeg')},
    )
    assert unfiltered.json()['num_texts'] == 1, unfiltered.text
    assert (
        api.post('/ocr/predict', files={'image': ('e.jpg', b'', 'image/jpeg')}).status_code == 400
    )


@pytest.fixture(scope='module')
def text_items(api_client: Any, opensearch: Any) -> dict[str, str]:
    """Three region-cohort items given plate text; returns text -> crop id."""
    ids = crop_ids_in(opensearch, 'rgn', limit=40)[30:33]
    assert len(ids) == 3
    texts = ['KLM4821', 'KLM4822', 'ZQX9000']
    for crop_id, text in zip(ids, texts, strict=True):
        box = get_doc(opensearch, INDEXES['items'], crop_id)['_source']['region_boxes'][0]
        resp = api_client.patch(f'/crops/{crop_id}/regions/{box["box_id"]}', json={'text': text})
        assert resp.status_code == 200, resp.text
    refresh(opensearch, INDEXES['items'])
    return dict(zip(texts, ids, strict=True))


def _found(api_client: Any, **params: Any) -> set[str]:
    resp = api_client.get('/crops', params={'limit': 100, **params})
    assert resp.status_code == 200, resp.text
    return {item['crop_id'] for item in resp.json()['crops']}


def test_region_text_exact_fuzzy_and_fragment_filters(
    api_client: Any, text_items: dict[str, str]
) -> None:
    exact = _found(api_client, region_text='KLM4821', region_text_match='exact')
    assert exact == {text_items['KLM4821']}

    # One edit away: the neighbouring plate is found, the unrelated one is not.
    fuzzy = _found(api_client, region_text='KLM4821', region_text_match='fuzzy')
    assert text_items['KLM4821'] in fuzzy
    assert text_items['KLM4822'] in fuzzy
    assert text_items['ZQX9000'] not in fuzzy

    fragment = _found(api_client, region_text='LM482', region_text_match='fragment')
    assert {text_items['KLM4821'], text_items['KLM4822']} <= fragment
    assert text_items['ZQX9000'] not in fragment

    assert _found(api_client, region_text='NOSUCH1', region_text_match='exact') == set()


def test_same_text_as_finds_the_boxes_sharing_a_plate(
    api_client: Any, opensearch: Any, text_items: dict[str, str]
) -> None:
    crop_id = text_items['KLM4821']
    box = get_doc(opensearch, INDEXES['items'], crop_id)['_source']['region_boxes'][0]
    found = _found(api_client, same_text_as=f'{crop_id}:{box["box_id"]}')
    assert found == {crop_id}

    both = api_client.get(
        '/crops',
        params={'same_text_as': f'{crop_id}:{box["box_id"]}', 'region_text': 'KLM4821'},
    )
    assert both.status_code == 400, both.text
