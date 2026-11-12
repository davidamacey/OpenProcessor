"""The fields the core-ingest live test reads exist in the served index mappings and the
``/search/ocr`` response model, so a rename fails here instead of in a wave run."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.clients.opensearch.indexes import IndexesMixin
from src.routers.search.models import OCRSearchResult
from tests.live.test_live_core_ingest import OCR_RESULT_KEYS, QUERIED_FIELDS, VECTOR_FIELD


class _Capture(IndexesMixin):
    embedding_dim = 512

    def __init__(self) -> None:
        self.bodies: dict[str, dict[str, Any]] = {}
        self.client: Any = SimpleNamespace(
            indices=SimpleNamespace(put_mapping=AsyncMock(return_value=None))
        )

    async def _create_index(
        self, index_name: str, index_body: dict[Any, Any], force_recreate: bool = False
    ) -> bool:
        self.bodies[index_name] = index_body
        return True


@pytest.fixture(scope='module')
def mappings() -> dict[str, dict[str, Any]]:
    capture = _Capture()
    for create in (
        capture.create_global_index,
        capture.create_faces_index,
        capture.create_ocr_index,
    ):
        asyncio.run(create(False))
    return {name: body['mappings']['properties'] for name, body in capture.bodies.items()}


def test_every_field_the_live_test_queries_is_in_the_index_mapping(mappings: dict) -> None:
    missing = {
        index: [f for f in fields if f not in mappings[index]]
        for index, fields in QUERIED_FIELDS.items()
    }
    assert {k: v for k, v in missing.items() if v} == {}


def test_the_vector_fields_are_512_dim_knn_vectors(mappings: dict) -> None:
    for index, field in VECTOR_FIELD.items():
        assert mappings[index][field]['type'] == 'knn_vector'
        assert mappings[index][field]['dimension'] == 512


def test_every_ocr_result_key_the_live_test_reads_is_in_the_response_model() -> None:
    assert [k for k in OCR_RESULT_KEYS if k not in OCRSearchResult.model_fields] == []
