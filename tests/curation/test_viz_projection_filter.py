"""``GET /viz/projection`` under the shared item filter: the points the filter leaves out are
served dimmed (``matched: false``) rather than dropped, a box drawn on the scatter is a plain
range query, and the legend of classes and colour slots is served."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from _embedding_registry_fake import DEFAULT_WRITE_FIELDS
from config_store.memory_settings import current

from curation.query_fakes import QueryFakeOpenSearch
from src.config.curation_accessors import items_index
from src.services.curation import embedding_viz_read
from src.services.curation.item_filter import ItemFilter, item_filter_clauses


@pytest.fixture(autouse=True)
def _projection_state(monkeypatch: pytest.MonkeyPatch) -> None:
    async def space(_target: Any) -> Any:
        return SimpleNamespace(vector_field='pe_embedding')

    monkeypatch.setattr('src.services.curation.embedding_viz_read.embedding_space', space)
    current().projection = {'projection_version': 'umap_viz_v1', 'fitted_at': 't0'}


def _item(crop_id: str, x: float, y: float, class_name: str | None, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'viz_x': x,
        'viz_y': y,
        'viz_projection_version': 'umap_viz_v1',
        'pe_embedding': [0.1],
        'class_name': class_name,
        **extra,
    }


def _world() -> QueryFakeOpenSearch:
    return QueryFakeOpenSearch(
        {
            items_index(): {
                'a': _item('a', 0.0, 0.0, 'sedan'),
                'b': _item('b', 1.0, 1.0, 'sedan'),
                'c': _item('c', 5.0, 5.0, 'truck'),
                'd': _item('d', 6.0, 6.0, None),
            }
        }
    )


def _clauses(**filter_fields: Any) -> list[dict[str, Any]]:
    return item_filter_clauses(ItemFilter(**filter_fields), embedding=DEFAULT_WRITE_FIELDS)


@pytest.mark.asyncio
async def test_unmatched_points_are_dimmed_not_dropped() -> None:
    got = await embedding_viz_read.get_cached_projection(
        _world(),
        filter_clauses=_clauses(class_names=['sedan']),
    )
    assert {p['crop_id']: p['matched'] for p in got['points']} == {
        'a': True,
        'b': True,
        'c': False,
        'd': False,
    }


@pytest.mark.asyncio
async def test_no_filter_matches_every_point() -> None:
    got = await embedding_viz_read.get_cached_projection(_world())
    assert all(p['matched'] for p in got['points'])


@pytest.mark.asyncio
async def test_viz_box_count_equals_a_range_query() -> None:
    world = _world()
    box = (-0.5, -0.5, 2.0, 2.0)
    got = await embedding_viz_read.get_cached_projection(
        world, filter_clauses=_clauses(viz_box=box)
    )
    matched = {p['crop_id'] for p in got['points'] if p['matched']}
    plain = await world.search(
        index=items_index(),
        body={
            'size': 100,
            'query': {
                'bool': {
                    'filter': [
                        {'range': {'viz_x': {'gte': box[0], 'lte': box[2]}}},
                        {'range': {'viz_y': {'gte': box[1], 'lte': box[3]}}},
                    ]
                }
            },
        },
    )
    assert matched == {h['_id'] for h in plain['hits']['hits']} == {'a', 'b'}


@pytest.mark.asyncio
async def test_legend_is_served() -> None:
    got = await embedding_viz_read.get_cached_projection(_world())
    assert got['legend'] == [
        {'class_name': 'sedan', 'colour_index': 0, 'count': 2},
        {'class_name': 'truck', 'colour_index': 1, 'count': 1},
        {'class_name': None, 'colour_index': 2, 'count': 1},
    ]
    by_id = {p['crop_id']: p['colour_index'] for p in got['points']}
    assert by_id == {'a': 0, 'b': 0, 'c': 1, 'd': 2}


def test_an_inverted_viz_box_is_refused() -> None:
    with pytest.raises(ValueError, match='viz_box'):
        ItemFilter(viz_box=(2.0, 0.0, 1.0, 1.0))
