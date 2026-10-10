"""#153: FastAPI deprecates its orjson response class. Nothing in ``src/`` may use
it, no request may emit a ``FastAPIDeprecationWarning``, and the bytes on the wire
must be identical to what the old serializer produced.

The old serializer lives here only as the reference: ``_old_model_bytes`` (validate
against the response model, dump in JSON mode, render with orjson) and
``_old_dict_bytes`` (``jsonable_encoder`` then orjson, the path of a route with
``response_model=None``), both with the options the old response class used.
"""

from __future__ import annotations

import datetime as dt
import decimal
import enum
import math
import pathlib
import re
import uuid
from typing import Any

import numpy as np
import orjson
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.datastructures import DefaultPlaceholder
from fastapi.encoders import jsonable_encoder
from fastapi.responses import Response
from fastapi.testclient import TestClient
from pydantic import BaseModel, TypeAdapter

from src.clients.occ import OCCFinalConflictError
from src.core.wire_json import WireJSONResponse, WireRoute
from src.main import app as main_app, create_app
from src.routers.embed import BoxEmbeddingsResponse
from src.routers.ingest.models import IngestResponse
from src.schemas.detection import InferenceResult


SRC = pathlib.Path(__file__).resolve().parents[1] / 'src'
OLD_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
NO_DEPRECATION = pytest.mark.filterwarnings('error::fastapi.exceptions.FastAPIDeprecationWarning')


def _old_model_bytes(model: Any, payload: Any) -> bytes:
    adapter = TypeAdapter(model)
    return orjson.dumps(
        adapter.dump_python(adapter.validate_python(payload), mode='json', by_alias=True),
        option=OLD_OPTIONS,
    )


def _old_dict_bytes(payload: Any) -> bytes:
    return orjson.dumps(jsonable_encoder(payload), option=OLD_OPTIONS)


def _iter_routes(routes: Any) -> Any:
    for route in routes:
        if type(route).__name__ == '_IncludedRouter':
            yield from _iter_routes(route.original_router.routes)
        elif hasattr(route, 'response_class') and hasattr(route, 'methods'):
            yield route


# --- the deprecated class is gone ---------------------------------------------


def test_src_has_no_orjson_response() -> None:
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob('*.py')
        if re.search(r'\bORJSONResponse\b', path.read_text())
    ]
    assert offenders == []


def test_model_routes_use_the_default_response_class() -> None:
    """A response class override disables FastAPI's Pydantic direct-to-bytes path."""
    routes = list(_iter_routes(main_app.routes))
    assert routes
    for route in routes:
        cls = getattr(route.response_class, 'value', route.response_class)
        assert 'ORJSON' not in [c.__name__.upper() for c in cls.__mro__], route.path
        if route.response_model is not None:
            assert isinstance(route.response_class, DefaultPlaceholder), route.path


# --- no deprecation warning on the paths that built the old class --------------


@NO_DEPRECATION
def test_error_handlers_emit_no_deprecation_warning() -> None:
    from src.utils.retry import RetryExhaustedError

    application = create_app()

    @application.get('/__t409')
    async def _conflict() -> None:
        raise OCCFinalConflictError(doc_id='crop-1', retries=2)

    @application.get('/__t503')
    async def _unavailable() -> None:
        raise RetryExhaustedError('down')

    @application.get('/__t500')
    async def _boom() -> None:
        raise RuntimeError('boom')

    client = TestClient(application, raise_server_exceptions=False)
    r409 = client.get('/__t409')
    assert r409.status_code == 409
    assert r409.headers['content-type'] == 'application/json'
    assert set(r409.json()) == {'detail', 'doc_id', 'retries', 'request_id'}
    r503 = client.get('/__t503')
    assert r503.status_code == 503
    assert r503.headers['retry-after'] == '5'
    assert set(r503.json()) == {'detail', 'request_id', 'error_type'}
    r500 = client.get('/__t500')
    assert r500.status_code == 500
    assert set(r500.json()) == {'detail', 'request_id', 'error_type'}


@NO_DEPRECATION
def test_curation_wire_route_emits_no_deprecation_warning() -> None:
    from curation.test_crop_context import _client, _fake

    r = _client(_fake()).get('/curation/projects/default/crops/a1/context')
    assert r.status_code == 200, r.text
    assert [i['crop_id'] for i in r.json()['items']] == ['a1', 'a2']
    assert r.headers['content-type'] == 'application/json'


def test_ingest_duplicate_branch_bytes_are_the_old_exclude_none_bytes() -> None:
    model = IngestResponse(
        status='duplicate', image_id='i1', imohash='h', message='dup', total_time_ms=1.5
    )
    new = model.model_dump_json(by_alias=True, exclude_none=True).encode()
    assert new == orjson.dumps(model.model_dump(by_alias=True, exclude_none=True))
    assert b'null' not in new


# --- golden bodies: model routes ------------------------------------------------


def _boxes(n: int) -> list[dict[str, Any]]:
    return [
        {'box': [0.1 + i * 1e-4, 0.2, 0.3, 1e-05], 'embedding': [i * 0.001, -0.5, 1e-7, 0.0]}
        for i in range(n)
    ]


GOLDEN: list[tuple[type[BaseModel], dict[str, Any]]] = [
    (
        IngestResponse,
        {
            'status': 'success',
            'image_id': 'ünï-1',
            'num_faces': 2,
            'embedding_norm': 1.0,
            'indexed': {'global': True, 'faces': 2},
            'ocr': {'num_texts': 1, 'full_text': 'a\nb "q"', 'indexed': True},
            'total_time_ms': 12.5,
        },
    ),
    (IngestResponse, {'status': 'error', 'image_id': 'x', 'error': 'bad', 'errors': ['a', 'b']}),
    (BoxEmbeddingsResponse, {'boxes': _boxes(25), 'num_boxes': 25}),
    (
        InferenceResult,
        {
            'detections': [
                {'x1': 0.1, 'y1': 1e-05, 'x2': 0.3, 'y2': 0.4, 'confidence': 0.91, 'class': 2}
            ],
            'num_detections': 1,
            'image': {'width': 4, 'height': 2},
            'model': {'name': 'm', 'backend': 'triton'},
            'total_time_ms': math.nan,
        },
    ),
]


@pytest.mark.parametrize(('model', 'payload'), GOLDEN, ids=lambda v: getattr(v, '__name__', ''))
@NO_DEPRECATION
def test_model_route_body_matches_old_serializer(
    model: type[BaseModel], payload: dict[str, Any]
) -> None:
    mini = FastAPI()

    @mini.get('/m', response_model=model)
    async def _m() -> Any:
        return payload

    response = TestClient(mini).get('/m')
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'application/json'
    assert response.content == _old_model_bytes(model, payload)


# --- golden bodies: wire-dict routes --------------------------------------------


class _Color(enum.Enum):
    RED = 'red'


class _Doc(BaseModel):
    name: str
    score: float


def _wire_payloads() -> dict[str, Any]:
    from src.services.curation.wire import serialize_item

    items = [
        serialize_item(
            {
                'crop_id': f'c{i}',
                'image_id': 'img',
                'confidence': 0.9 - i * 1e-3,
                'bbox_norm': [0.1, 0.2, 0.3, 0.4],
                'region_boxes': [
                    {'box_id': 'b1', 'bbox_norm': [0.1, 0.1, 0.2, 0.2], 'state': 'accepted'}
                ],
            },
            f'c{i}',
            api_prefix='/curation/projects/default',
        )
        for i in range(100)
    ]
    return {
        'review_page': {
            'total': 1234,
            'page': 1,
            'page_size': 100,
            'items': items,
            'sort_applied': 'recent',
            'sort_fallback_reason': None,
            'empty_reason': None,
        },
        'edge_types': {
            'nan': math.nan,
            'small': 1e-05,
            'text': 'é "q"\n ',
            'by_int_key': {1: 'a', 2: 'b'},
            'when': dt.datetime(2026, 1, 2, 3, 4, 5, 600, tzinfo=dt.UTC),
            'day': dt.date(2026, 1, 2),
            'color': _Color.RED,
            'id': uuid.UUID(int=7),
            'dec': decimal.Decimal('1.50'),
            'path': pathlib.PurePosixPath('/a/b'),
            'set': {3},
            'model': _Doc(name='n', score=0.5),
            'tuple': (1, 2),
            'raw': b'bytes',
        },
    }


@pytest.mark.parametrize('name', ['review_page', 'edge_types'])
@NO_DEPRECATION
def test_wire_route_body_matches_old_serializer(name: str) -> None:
    payload = _wire_payloads()[name]
    router = APIRouter(route_class=WireRoute)

    @router.get('/w', response_model=None)
    async def _w() -> dict[str, Any]:
        return payload

    mini = FastAPI()
    mini.include_router(router)
    response = TestClient(mini).get('/w')
    assert response.status_code == 200, response.text
    assert response.headers['content-type'] == 'application/json'
    assert response.content == _old_dict_bytes(payload)


def test_wire_response_keeps_numpy_non_str_keys_and_nan() -> None:
    payload = {
        'vec': np.array([0.25, 0.5], dtype=np.float32),
        'arr': np.arange(3),
        'by_id': {1: 'a'},
        'nan': math.nan,
    }
    response = WireJSONResponse(payload)
    assert response.body == orjson.dumps(payload, option=OLD_OPTIONS)
    assert response.body == b'{"vec":[0.25,0.5],"arr":[0,1,2],"by_id":{"1":"a"},"nan":null}'


def test_wire_route_only_touches_explicit_response_model_none() -> None:
    router = APIRouter(route_class=WireRoute)

    @router.get('/plain')
    async def _plain() -> dict[str, Any]:
        return {'a': 1}

    @router.post('/created', response_model=None, status_code=201)
    async def _created() -> dict[str, Any]:
        return {'a': 1}

    @router.get('/passthrough', response_model=None)
    async def _passthrough() -> Response:
        return Response(b'x', media_type='text/plain', status_code=202)

    @router.delete('/gone', response_model=None, status_code=204)
    async def _gone() -> None:
        return None

    mini = FastAPI()
    mini.include_router(router)
    client = TestClient(mini)
    assert client.get('/plain').json() == {'a': 1}
    created = client.post('/created')
    assert (created.status_code, created.content) == (201, b'{"a":1}')
    passthrough = client.get('/passthrough')
    assert (passthrough.status_code, passthrough.content) == (202, b'x')
    gone = client.delete('/gone')
    assert (gone.status_code, gone.content) == (204, b'')
