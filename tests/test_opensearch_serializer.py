"""The OpenSearch body serializer writes the same JSON values the stock one did (#211)."""

from __future__ import annotations

import datetime as dt
import json
import uuid

import numpy as np
import pytest
from opensearchpy.exceptions import SerializationError
from opensearchpy.serializer import JSONSerializer

from src.clients.opensearch.serializer import OrjsonSerializer
from src.utils.vectors import float_list


def _same_values(doc: object) -> None:
    assert json.loads(OrjsonSerializer().dumps(doc)) == json.loads(JSONSerializer().dumps(doc))


def test_an_items_document_with_a_vector() -> None:
    vector = np.random.default_rng(0).normal(size=1024).astype(np.float32)
    doc = {
        'crop_id': 'c1',
        'bbox_norm': [0.1, 0.2, 0.5, 0.75],
        'confidence': 0.9123456789,
        'pe_embedding': float_list(vector),
        'region_boxes': [{'box_id': 'b', 'state': 'accepted', 'score': 1e-7}],
        'note': 'naïve – ünïcode',
        'flag': True,
        'gone': None,
    }
    _same_values(doc)


def test_float32_values_survive_exactly() -> None:
    vector = np.random.default_rng(1).normal(size=512).astype(np.float32)
    body = OrjsonSerializer().dumps({'v': float_list(vector)})
    back = np.asarray(json.loads(body)['v'], dtype=np.float32)
    assert np.array_equal(back, vector)


def test_numpy_scalars_and_arrays_go_through_the_default_hook() -> None:
    doc = {
        'f32': np.float32(0.1),
        'i64': np.int64(7),
        'flag': np.bool_(True),
        'arr': np.arange(3, dtype=np.float32),
    }
    _same_values(doc)


def test_dates_and_uuids_and_integer_keys_match() -> None:
    doc = {
        'when': dt.datetime(2026, 10, 10, 12, 30, 5, 123456, tzinfo=dt.UTC),
        'naive': dt.datetime(2026, 1, 2, 3, 4, 5),
        'day': dt.date(2026, 10, 10),
        'id': uuid.UUID('12345678-1234-5678-1234-567812345678'),
        7: 'seven',
    }
    _same_values(doc)


def test_strings_pass_through_as_the_bulk_body_lines() -> None:
    assert OrjsonSerializer().dumps('{"already":"json"}') == '{"already":"json"}'


def test_an_integer_beyond_64_bits_falls_back_to_the_stock_encoder() -> None:
    doc = {'big': 2**70}
    assert OrjsonSerializer().dumps(doc) == JSONSerializer().dumps(doc)


def test_an_unserializable_value_raises_the_stock_error() -> None:
    with pytest.raises(SerializationError):
        OrjsonSerializer().dumps({'bad': object()})


def test_a_float_list_is_python_floats_with_the_same_values_as_list() -> None:
    vector = np.random.default_rng(2).normal(size=64).astype(np.float32)
    converted = float_list(vector)
    assert all(type(v) is float for v in converted)
    assert converted == [float(v) for v in vector]
    assert float_list([1, 2.5]) == [1, 2.5]


def test_documents_carry_vectors_as_python_floats() -> None:
    from src.services.curation.item_doc import DetectedItem, build_image_doc, build_item_doc

    vector = np.random.default_rng(3).normal(size=1024).astype(np.float32)
    item = DetectedItem(bbox_pixel=(0.0, 0.0, 10.0, 10.0), score=0.5, pe_embedding=vector)
    item.backbone_embedding = vector[:16]
    doc = build_item_doc(
        crop_id='c',
        image_id='i',
        image_path='/p.jpg',
        source='s',
        request_id='r',
        bbox_norm=[0.0, 0.0, 0.1, 0.1],
        item=item,
        now='2026-10-10T00:00:00+00:00',
        crop_area_norm=0.01,
        crop_rank_in_image=1,
        blur_full_var=None,
        blur_lap_var=None,
        blur_lap_ratio=None,
    )
    image_doc = build_image_doc(
        image_id='i',
        image_path='/p.jpg',
        source='s',
        width=1,
        height=1,
        imohash='h',
        now='n',
        whole_frame_embedding=vector,
    )
    for stored in (doc['pe_embedding'], image_doc['pe_embedding']):
        assert type(stored) is list
        assert all(type(v) is float for v in stored)
        assert stored == [float(v) for v in vector]
    assert all(type(v) is float for v in doc['backbone_embedding'])
