"""POST /curation/projects/{project}/search/image: query by a picture (#402)."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from _msearch_fake import install_msearch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image


URL = '/curation/projects/default/search/image'
PE_DIM = 1024
QUERY_VECTOR = [1.0] + [0.0] * (PE_DIM - 1)

_HIT = {
    '_id': 'crop_1',
    '_score': 0.9,
    '_source': {
        'crop_id': 'crop_1',
        'image_path': '/data/foo.jpg',
        'region_boxes': [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.5, 0.5], 'state': 'accepted'}],
        'region_box_embeddings': [{'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.5, 0.5]}],
    },
    'inner_hits': {'matched_box': {'hits': {'hits': [{'_nested': {'offset': 0}}]}}},
}


def _image(fmt='PNG', size=(40, 20), mode='RGB', orientation=None) -> bytes:
    img = Image.new(mode, size, (200, 30, 30, 255)[: len(mode)] if mode != 'L' else 128)
    buf = io.BytesIO()
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs['exif'] = exif
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _encoder() -> MagicMock:
    enc = MagicMock()
    enc.embed_crops = AsyncMock(return_value=np.asarray([QUERY_VECTOR], dtype=np.float32))
    return enc


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    import src.config.curation_accessors as accessors
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr(
        'src.services.curation.index_bootstrap.ensure_indexes', AsyncMock(return_value=None)
    )
    monkeypatch.setenv('OP_SEMANTIC_SEARCH_ENABLED', 'true')
    monkeypatch.setattr(accessors, '_default_curation_config', None)

    fake_os = install_msearch(AsyncMock())
    fake_os.search = AsyncMock(return_value={'hits': {'hits': [_HIT], 'total': {'value': 1}}})
    fake_os.count = AsyncMock(return_value={'count': 4})
    fake_os.get = AsyncMock(return_value={'found': False})

    app = FastAPI()
    from _curation_app import mount_curation_routers

    mount_curation_routers(app, curation_router)
    app.state.pe_encoder = _encoder()
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake_os
    c = TestClient(app)
    c.fake_os = fake_os  # type: ignore[attr-defined]
    c.encoder = app.state.pe_encoder  # type: ignore[attr-defined]
    yield c
    monkeypatch.setattr(accessors, '_default_curation_config', None)


def _post(client, data: bytes | None = None, *, form=None, params=None):
    files = {'image': ('q.png', data, 'image/png')} if data is not None else None
    return client.post(URL, files=files, data=form or {}, params=params or {})


def _knn_body(client) -> dict:
    return client.fake_os.search.call_args_list[0].kwargs['body']


def test_disabled_flag_is_400(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('OP_SEMANTIC_SEARCH_ENABLED', 'false')
    resp = _post(client, _image())
    assert resp.status_code == 400
    assert 'OP_SEMANTIC_SEARCH_ENABLED' in resp.json()['detail']
    client.encoder.embed_crops.assert_not_called()


def test_happy_path_encodes_once_as_a_crop_and_returns_the_text_shape(client):
    resp = _post(client, _image())
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {'items', 'total', 'page', 'page_size', 'unembedded_in_scope'}
    assert body['items'][0]['semantic_score'] == 0.9
    assert body['unembedded_in_scope'] == 4
    client.encoder.embed_crops.assert_awaited_once()
    (crops,), _ = client.encoder.embed_crops.call_args
    assert len(crops) == 1
    assert crops[0].shape == (20, 40, 3)
    assert not hasattr(client.encoder, 'embed_whole_frame_image') or (
        not client.encoder.embed_whole_frame_image.called
    )
    assert _knn_body(client)['query']['knn']['pe_embedding']['vector'][0] == 1.0


def test_item_target_does_not_report_a_matched_box(client):
    item = _post(client, _image()).json()['items'][0]
    assert 'matched_box_id' not in item


def test_box_target_reports_the_matched_box(client):
    resp = _post(client, _image(), params={'target': 'box'})
    assert resp.status_code == 200
    assert resp.json()['items'][0]['matched_box_id'] == 'b1'
    assert 'nested' in _knn_body(client)['query']['bool']['must'][0]


def test_unknown_target_is_422(client):
    assert _post(client, _image(), params={'target': 'frame'}).status_code == 422


def test_box_crops_before_encoding(client):
    resp = _post(client, _image(size=(100, 50)), form={'box': '0.5,0,1,0.5'})
    assert resp.status_code == 200
    (crops,), _ = client.encoder.embed_crops.call_args
    assert crops[0].shape == (25, 50, 3)


@pytest.mark.parametrize(
    'box',
    [
        '0.5,0.5,0.4,0.9',
        '0,0,1.5,1',
        '-0.1,0,1,1',
        'a,b,c,d',
        '0.1,0.1,0.5',
        '0,0,0.001,1',
        'nan,0,1,1',
    ],
)
def test_bad_box_is_400(client, box: str):
    resp = _post(client, _image(size=(100, 50)), form={'box': box})
    assert resp.status_code == 400
    assert resp.json()['detail']['error'] == 'bad_box'
    client.encoder.embed_crops.assert_not_called()


def test_undecodable_bytes_are_400(client):
    resp = _post(client, b'this is not a picture')
    assert resp.status_code == 400
    assert resp.json()['detail']['error'] == 'image_unreadable'
    client.encoder.embed_crops.assert_not_called()


def test_oversize_bytes_are_413_before_any_decode(client, monkeypatch: pytest.MonkeyPatch):
    import src.config.curation_accessors as accessors

    monkeypatch.setenv('OP_SEARCH_IMAGE_MAX_BYTES', '100')
    monkeypatch.setattr(accessors, '_default_curation_config', None)
    decode = MagicMock()
    monkeypatch.setattr('src.services.curation.semantic_search_image.decode_image', decode)
    resp = _post(client, _image(size=(200, 200)))
    assert resp.status_code == 413
    assert resp.json()['detail']['error'] == 'upload_too_large'
    decode.assert_not_called()
    client.encoder.embed_crops.assert_not_called()


def test_too_many_pixels_are_413_before_full_decode(client, monkeypatch: pytest.MonkeyPatch):
    import src.config.curation_accessors as accessors

    monkeypatch.setenv('OP_SEARCH_IMAGE_MAX_PIXELS', '100')
    monkeypatch.setattr(accessors, '_default_curation_config', None)
    decode = MagicMock()
    monkeypatch.setattr('src.services.curation.semantic_search_image.decode_image', decode)
    resp = _post(client, _image(size=(20, 20)))
    assert resp.status_code == 413
    decode.assert_not_called()


def test_encoder_failure_is_503_never_an_empty_200(client):
    client.encoder.embed_crops = AsyncMock(side_effect=RuntimeError('triton down'))
    resp = _post(client, _image())
    assert resp.status_code == 503
    assert resp.json()['detail']['error'] == 'image_encoder_unavailable'
    client.fake_os.search.assert_not_called()


def test_missing_encoder_is_503(client):
    client.app.state.pe_encoder = None
    resp = _post(client, _image())
    assert resp.status_code == 503
    assert resp.json()['detail']['error'] == 'image_encoder_unavailable'


def test_exif_orientation_is_applied_before_the_crop(client):
    data = _image(fmt='JPEG', size=(40, 20), orientation=6)
    assert _post(client, data).status_code == 200
    (crops,), _ = client.encoder.embed_crops.call_args
    assert crops[0].shape == (40, 20, 3)


def test_rgba_and_grayscale_become_three_channels(client):
    for mode in ('RGBA', 'L'):
        client.encoder.embed_crops.reset_mock()
        assert _post(client, _image(mode=mode)).status_code == 200
        (crops,), _ = client.encoder.embed_crops.call_args
        assert crops[0].shape[2] == 3


def test_filters_reach_the_knn_filter(client):
    resp = _post(
        client,
        _image(),
        params={'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11', 'min_score': 0.3},
    )
    assert resp.status_code == 200
    body = _knn_body(client)
    assert {'term': {'cluster_uid': '6f1c1d3e-0a7e-4c55-9d5e-2f0f8f2a9b11'}} in body['query'][
        'knn'
    ]['pe_embedding']['filter']['bool']['filter']
    assert body['min_score'] == 0.3


def test_malformed_item_filter_band_is_400(client):
    resp = _post(client, _image(), params={'conf_min': 0.9, 'conf_max': 0.1})
    assert resp.status_code == 400


def test_only_the_bound_projects_indexes_are_read(client):
    _post(client, _image())
    seen = {c.kwargs['index'] for c in client.fake_os.search.call_args_list}
    seen |= {c.kwargs['index'] for c in client.fake_os.count.call_args_list}
    assert seen
    assert all(i.endswith('__items') for i in seen)
    resp = client.post(
        '/curation/projects/nope/search/image', files={'image': ('q.png', _image(), 'image/png')}
    )
    assert resp.status_code == 404
    assert resp.json()['detail']['error'] == 'project_not_found'


def test_nothing_sent_is_400(client):
    resp = _post(client)
    assert resp.status_code == 400
    assert resp.json()['detail']['error'] == 'invalid_request'


def test_both_image_and_image_id_is_400(client):
    resp = _post(client, _image(), form={'image_id': 'img1'})
    assert resp.status_code == 400
    assert resp.json()['detail']['error'] == 'invalid_request'


# --- referenced image (image_id) ------------------------------------------------------------


def test_unknown_image_id_is_404(client):
    resp = _post(client, form={'image_id': 'nope'})
    assert resp.status_code == 404
    assert resp.json()['detail']['error'] == 'image_not_found'


def test_referenced_image_is_read_from_the_project_and_boxed(
    client, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import src.config.curation_accessors as accessors

    (tmp_path / 'a.png').write_bytes(_image(size=(100, 50)))
    monkeypatch.setenv('OP_SOURCE_ROOT', str(tmp_path))
    monkeypatch.setattr(accessors, '_default_curation_config', None)
    client.fake_os.get = AsyncMock(
        return_value={'found': True, '_source': {'image_path': str(tmp_path / 'a.png')}}
    )
    resp = _post(client, form={'image_id': 'img1', 'box': '0,0,0.5,1'})
    assert resp.status_code == 200
    (crops,), _ = client.encoder.embed_crops.call_args
    assert crops[0].shape == (50, 50, 3)
    assert client.fake_os.get.call_args.kwargs['id'] == 'img1'


def test_referenced_image_outside_the_source_roots_is_refused(
    client, tmp_path, monkeypatch: pytest.MonkeyPatch
):
    import src.config.curation_accessors as accessors

    root = tmp_path / 'root'
    root.mkdir()
    (tmp_path / 'secret.png').write_bytes(_image())
    monkeypatch.setenv('OP_SOURCE_ROOT', str(root))
    monkeypatch.setattr(accessors, '_default_curation_config', None)
    client.fake_os.get = AsyncMock(
        return_value={'found': True, '_source': {'image_path': str(tmp_path / 'secret.png')}}
    )
    resp = _post(client, form={'image_id': 'img1'})
    assert resp.status_code == 400
    client.encoder.embed_crops.assert_not_called()
