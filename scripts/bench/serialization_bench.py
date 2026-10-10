"""Response serialization micro-benchmark (#153): the orjson response class path
against the path that replaced it.

Offline and in-process: each case mounts the same handler twice on a bare app
(the old response-class route and the new route) and times full requests through
``TestClient``, so request overhead is a constant on both sides. The legacy
class below is a local stand-in for FastAPI's deprecated orjson class (same
``render``); it exists only to give the comparison a reference.

Cases:

* ``review_page``: a 100-item ``GET /review/{tab}`` page (a wire dict).
  Old: ``jsonable_encoder`` then orjson. New: ``WireRoute`` (orjson with a
  ``jsonable_encoder`` fallback only for non-JSON values). ``stdlib`` is what
  dropping the class and changing nothing else would have cost.
* ``box_embeddings_1000``: a 1000-box ``/embed/boxes`` response (512-dim vectors).
  Old: validate, dump to Python, orjson. New: validate, Pydantic to bytes.
* ``batch_ingest_1000``: an ``/ingest/batch`` response with 1000 detail rows.

Usage: ``.venv/bin/python -m scripts.bench.serialization_bench [--iterations N]``
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import orjson
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient


if TYPE_CHECKING:
    from collections.abc import Callable

_LEGACY_OPTIONS = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY


class _LegacyOrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=_LEGACY_OPTIONS)


@dataclass(frozen=True)
class Row:
    case: str
    path: str
    ms_median: float
    ms_min: float
    body_bytes: int


def _review_page(n_items: int) -> dict[str, Any]:
    from src.services.curation.wire import serialize_item

    items = [
        serialize_item(
            {
                'crop_id': f'c{i}',
                'image_id': f'img{i // 4}',
                'class_id': i % 7,
                'confidence': 0.5 + (i % 50) / 100,
                'bbox_norm': [0.1, 0.2, 0.3, 0.4],
                'region_boxes': [
                    {
                        'box_id': f'b{k}',
                        'bbox_norm': [0.1, 0.1, 0.2, 0.2],
                        'state': 'accepted',
                        'score': 0.8,
                        'detector': 'det',
                    }
                    for k in range(5)
                ],
            },
            f'c{i}',
            api_prefix='/curation/projects/bench',
        )
        for i in range(n_items)
    ]
    return {
        'total': n_items * 10,
        'page': 1,
        'page_size': n_items,
        'items': items,
        'sort_applied': 'recent',
        'sort_fallback_reason': None,
        'empty_reason': None,
    }


def _box_embeddings(n_boxes: int, dim: int) -> dict[str, Any]:
    return {
        'boxes': [
            {
                'box': [0.1, 0.2, 0.3 + i * 1e-4, 0.4],
                'embedding': [((i + j) % 97) / 97 for j in range(dim)],
            }
            for i in range(n_boxes)
        ],
        'num_boxes': n_boxes,
    }


def _batch_ingest(n_rows: int) -> dict[str, Any]:
    return {
        'status': 'partial',
        'total': n_rows * 3,
        'processed': n_rows,
        'duplicates': n_rows,
        'errors_count': n_rows,
        'duplicate_details': [
            {'image_id': f'i{i}', 'existing_image_id': f'e{i}', 'imohash': f'{i:032x}'}
            for i in range(n_rows)
        ],
        'error_details': [{'image_id': f'x{i}', 'error': 'decode failed'} for i in range(n_rows)],
        'total_time_ms': 1234.5,
    }


def _time(client: TestClient, url: str, iterations: int) -> tuple[list[float], bytes]:
    body = client.get(url).content
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        client.get(url)
        samples.append((time.perf_counter() - start) * 1000)
    return samples, body


def _app(add: Callable[[FastAPI], None]) -> TestClient:
    app = FastAPI()
    add(app)
    return TestClient(app)


def _wire_dict_case(payload: dict[str, Any]) -> dict[str, Callable[[FastAPI], None]]:
    from src.core.wire_json import WireRoute

    def legacy(app: FastAPI) -> None:
        @app.get('/p', response_model=None, response_class=_LegacyOrjsonResponse)
        async def _p() -> dict[str, Any]:
            return payload

    def stdlib(app: FastAPI) -> None:
        @app.get('/p', response_model=None)
        async def _p() -> dict[str, Any]:
            return payload

    def new(app: FastAPI) -> None:
        router = APIRouter(route_class=WireRoute)

        @router.get('/p', response_model=None)
        async def _p() -> dict[str, Any]:
            return payload

        app.include_router(router)

    return {
        'old (jsonable_encoder + orjson)': legacy,
        'stdlib (jsonable_encoder + json)': stdlib,
        'new (WireRoute)': new,
    }


def _model_case(model: type, payload: dict[str, Any]) -> dict[str, Callable[[FastAPI], None]]:
    def legacy(app: FastAPI) -> None:
        @app.get('/p', response_model=model, response_class=_LegacyOrjsonResponse)
        async def _p() -> Any:
            return payload

    def new(app: FastAPI) -> None:
        @app.get('/p', response_model=model)
        async def _p() -> Any:
            return payload

    return {'old (validate + dump_python + orjson)': legacy, 'new (validate + Pydantic bytes)': new}


def run(
    iterations: int = 30, review_items: int = 100, boxes: int = 1000, dim: int = 512
) -> list[Row]:
    from src.routers.embed import BoxEmbeddingsResponse
    from src.routers.ingest.models import BatchIngestResponse

    cases: dict[str, dict[str, Callable[[FastAPI], None]]] = {
        f'review_page_{review_items}': _wire_dict_case(_review_page(review_items)),
        f'box_embeddings_{boxes}': _model_case(BoxEmbeddingsResponse, _box_embeddings(boxes, dim)),
        f'batch_ingest_{boxes}': _model_case(BatchIngestResponse, _batch_ingest(boxes)),
    }
    rows: list[Row] = []
    for case, paths in cases.items():
        reference: bytes | None = None
        for path, add in paths.items():
            samples, body = _time(_app(add), '/p', iterations)
            if path.startswith('old'):
                reference = body
            elif path.startswith('new'):
                assert body == reference, f'{case}: new body differs from the old serializer'
            rows.append(Row(case, path, statistics.median(samples), min(samples), len(body)))
    return rows


def format_rows(rows: list[Row]) -> str:
    lines = [f'{"case":<24}{"path":<40}{"median ms":>11}{"min ms":>10}{"bytes":>12}']
    lines += [
        f'{r.case:<24}{r.path:<40}{r.ms_median:>11.2f}{r.ms_min:>10.2f}{r.body_bytes:>12}'
        for r in rows
    ]
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    print(format_rows(run(iterations=args.iterations)))


if __name__ == '__main__':
    main()
