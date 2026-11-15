"""The baseline client names every photo that failed and says why a request was refused.

The 2026-11-13 speed chain recorded only counts: six failed photos and two refused requests could
be explained only from the API's log, afterwards."""

from __future__ import annotations

import httpx
import pytest

from scripts.bench import baseline_report as rep, run_baseline as rb


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url='http://api')


def test_a_failed_photo_is_recorded_with_its_name_kind_and_message() -> None:
    body = {
        'status': 'partial',
        'summary': {
            'successful': 1,
            'duplicates': 0,
            'failed': 1,
            'crops_indexed': 2,
            'n_embedded': 2,
        },
        'results': [
            {'status': 'success', 'image_path': '/d/a.jpg'},
            {
                'status': 'failed',
                'image_path': '/d/b.jpg',
                'error_kind': 'engine_unavailable',
                'error': '[StatusCode.INTERNAL] Stream removed',
            },
        ],
    }
    out = rb._ingest_batch(
        _client(lambda _r: httpx.Response(200, json=body)),
        's',
        ['/d/a.jpg', '/d/b.jpg'],
        iter([0.0, 1.0]).__next__,
    )
    assert out.failed == 1
    assert out.failures == (
        {
            'name': '/d/b.jpg',
            'error_kind': 'engine_unavailable',
            'error': '[StatusCode.INTERNAL] Stream removed',
        },
    )


def test_a_refused_request_names_the_status_the_request_id_and_the_answer() -> None:
    refused = httpx.Response(
        503,
        headers={'x-request-id': 'req-6223bb3f'},
        json={'code': 'engine_unavailable', 'detail': 'the image engine is not ready'},
    )
    with pytest.raises(RuntimeError) as caught:
        rb._ingest_batch(_client(lambda _r: refused), 's', ['/d/a.jpg'], iter([0.0, 1.0]).__next__)
    text = str(caught.value)
    assert '503' in text
    assert 'req-6223bb3f' in text
    assert 'engine_unavailable' in text


def test_the_name_of_an_uploaded_photo_is_the_name_the_client_gave() -> None:
    [one] = rep.failed_photos(
        [
            {
                'status': 'failed',
                'image_path': '/state/uploads/ab12.jpg',
                'source_identifier': 'holiday/IMG_0001.jpg',
                'error_kind': 'decode_failed',
                'error': 'x',
            }
        ]
    )
    assert one['name'] == 'holiday/IMG_0001.jpg'
